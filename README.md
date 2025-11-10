FOC is a custom image processing pipeline with advanced capabilities that AIJ doesn't offer. It's designed to clean astronomical optical images in FITS format. The pipeline implements key steps including bad pixel correction, bad column removal, cosmic ray cleaning with AstroScrappy, wavelet-based denoising, and image cropping. The system supports single image processing, batch processing, and RGB composite generation via a Flask API backend ready for frontend integration.

## Features

- Correction of hot/bad pixels with optional external masks.
- Identification and replacement of bad columns.
- Cosmic ray detection and removal using AstroScrappy with configurable parameters.
- Wavelet denoising with adjustable wavelet type and thresholding.
- Cropping of image edges as a fraction of total size.
- Creation of RGB composite PNG images from FITS files.
- RESTful Flask API for single, batch, and RGB processing with CORS enabled for frontend apps.

----------###Link to the website###-------------
  
  
  https://foc-pipeline.web.app/

