import os
import time
import io
import base64
import json
import zipfile
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from astropy.io import fits
from scipy.ndimage import median_filter
import astroscrappy
import pywt
from PIL import Image

# --- CORE PROCESSING FUNCTIONS (from your scripts) ---
# NOTE: Functions are slightly adapted to accept parameters and data directly

def correct_bad_pixels(image, hot_threshold=5.0, external_mask=None):
    if image.ndim != 2: raise ValueError("Input image must be 2D.")
    mask = external_mask.astype(bool) if isinstance(external_mask, np.ndarray) else image > (np.median(image) + hot_threshold * np.std(image))
    median_img = median_filter(image, size=3)
    corrected = image.copy()
    corrected[mask] = median_img[mask]
    return corrected, mask

def remove_bad_columns(image, sigma_threshold=3.0):
    img = image.copy()
    col_medians = np.median(img, axis=0)
    median, std = np.median(col_medians), np.std(col_medians)
    bad_cols = np.where(np.abs(col_medians - median) > sigma_threshold * std)[0]
    for col in bad_cols:
        if 1 <= col < img.shape[1] - 1: img[:, col] = 0.5 * (img[:, col - 1] + img[:, col + 1])
        elif col == 0: img[:, col] = img[:, col + 1]
        elif col == img.shape[1] - 1: img[:, col] = img[:, col - 1]
    return img, bad_cols

def crop_frame_fraction(image, frac=0.02):
    y, x = image.shape
    dy, dx = int(y * frac), int(x * frac)
    return image[dy:y - dy, dx:x - dx]

def wavelet_denoise(image, wavelet='db1', level=1, thresholding='soft'):
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-1][0])) / 0.6745 * np.sqrt(2 * np.log(image.size))
    new_coeffs = [coeffs[0]] + [tuple(pywt.threshold(d, threshold, mode=thresholding) for d in detail) for detail in coeffs[1:]]
    return pywt.waverec2(new_coeffs, wavelet)

def full_pipeline(image_data, params):
    """Single image processing pipeline, returns dict of image stages."""
    results = {'original': image_data.astype(float)}
    
    # 1. Bad Pixel/Column Correction
    processed = results['original']
    if params.get('use_bp_correction', False):
        mask = fits.getdata(io.BytesIO(params['mask_file'])).astype(bool) if 'mask_file' in params else None
        processed, _ = correct_bad_pixels(processed, params['hot_threshold'], external_mask=mask)
        processed, _ = remove_bad_columns(processed, params['column_sigma'])
        results['bad_pixel_corrected'] = processed
        
    # 2. AstroScrappy
    mask_astro, clean_astro = astroscrappy.detect_cosmics(
        processed, gain=params['gain1'], readnoise=params['readnoise1'], sigclip=params['sigclip1'],
        sigfrac=params['sigfrac1'], objlim=params['objlim1'], psffwhm=params['psffwhm1'], niter=params['niter1']
    )
    results['astroscrappy_cleaned'] = clean_astro
    
    # 3. Conservative AstroScrappy
    mask_cons, clean_cons = astroscrappy.detect_cosmics(
        processed, gain=params['gain2'], readnoise=params['readnoise2'], sigclip=params['sigclip2'],
        sigfrac=params['sigfrac2'], objlim=params['objlim2'], psffwhm=params['psffwhm2'], niter=params['niter2']
    )
    results['conservative_cleaned'] = clean_cons
    
    # 4. Denoising
    denoised_img = results[params['denoise_source']]
    results['denoised'] = wavelet_denoise(denoised_img, params['wavelet'], params['wavelet_level'], params['thresholding'])

    # 5. Cropping
    final_img = results[params['crop_source']]
    results['final_cropped'] = crop_frame_fraction(final_img, params['crop_frac'])
    
    return results

def normalize_to_png_bytes(data):
    vmin, vmax = np.percentile(data, [1, 99])
    scaled = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    img = Image.fromarray((scaled * 255).astype(np.uint8), 'L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

def create_rgb_composite(r, g, b, stretch):
    def normalize(data):
        vmin, vmax = np.percentile(data, [1, 99])
        return np.clip((data - vmin) / (vmax - vmin), 0, 1)**stretch
    rgb = np.dstack((normalize(r), normalize(g), normalize(b)))
    img = Image.fromarray((rgb * 255).astype(np.uint8), 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

def get_params_from_form(form):
    params = {}
    for key in form:
        try: params[key] = float(form[key])
        except ValueError: params[key] = form[key]
    params['use_bp_correction'] = 'maskFile' in request.files or params.get('hot_threshold', 0) > 0
    if 'maskFile' in request.files: params['mask_file'] = request.files['maskFile'].read()
    # Ensure integer types
    for int_key in ['niter1', 'niter2', 'wavelet_level']:
        if int_key in params:
            params[int_key] = int(params[int_key])
    return params

@app.route('/process-single', methods=['POST'])
def process_single_file():
    if 'fitsFile' not in request.files: return jsonify({'error': 'No FITS file provided'}), 400
    
    with fits.open(io.BytesIO(request.files['fitsFile'].read())) as hdul:
        image_data = hdul[0].data
        
    params = get_params_from_form(request.form)
    results = full_pipeline(image_data, params)

    # Prepare response
    response_data = {
        'images': {k: f"data:image/png;base64,{base64.b64encode(normalize_to_png_bytes(v)).decode('utf-8')}" for k, v in results.items()},
        'fits_files': {k: base64.b64encode(fits.HDUList([fits.PrimaryHDU(data=v)]).writeto(io.BytesIO(), overwrite=True) or b'').decode('utf-8') for k, v in results.items()}
    }
    return jsonify(response_data)

@app.route('/process-batch', methods=['POST'])
def process_batch_files():
    files = request.files.getlist('fitsFiles')
    if not files: return jsonify({'error': 'No FITS files provided for batch processing'}), 400

    params = get_params_from_form(request.form)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            with fits.open(io.BytesIO(f.read())) as hdul:
                image_data = hdul[0].data
            results = full_pipeline(image_data, params)
            
            # Save final cropped image to zip
            hdu = fits.PrimaryHDU(data=results['final_cropped'])
            hdulist = fits.HDUList([hdu])
            fits_io = io.BytesIO()
            hdulist.writeto(fits_io, overwrite=True)
            fits_io.seek(0)
            zf.writestr(f"processed_{f.filename}", fits_io.getvalue())

    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='processed_batch.zip')

@app.route('/process-rgb', methods=['POST'])
def process_rgb_files():
    if not all(k in request.files for k in ['r_file', 'g_file', 'b_file']):
        return jsonify({'error': 'Please provide all three FITS files for RGB composite.'}), 400
    
    with fits.open(io.BytesIO(request.files['r_file'].read())) as h: r_data = np.squeeze(h[0].data)
    with fits.open(io.BytesIO(request.files['g_file'].read())) as h: g_data = np.squeeze(h[0].data)
    with fits.open(io.BytesIO(request.files['b_file'].read())) as h: b_data = np.squeeze(h[0].data)
    
    stretch = float(request.form.get('stretch', 0.8))
    rgb_png_bytes = create_rgb_composite(r_data, g_data, b_data, stretch)
    
    return jsonify({'rgb_image': f"data:image/png;base64,{base64.b64encode(rgb_png_bytes).decode('utf-8')}"})

if __name__ == "__main__":
    # The port will be set by the hosting service, so we read it from the environment.
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)