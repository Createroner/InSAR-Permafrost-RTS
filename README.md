# InSAR-Permafrost-RTS
# Data-Driven Atmospheric Phase Identification and Removal for Enhanced InSAR Monitoring of Permafrost Thaw Subsidence

This repository contains the Python implementation of the methodologies described in the paper "Data-Driven Atmospheric Phase Identification and Removal for Enhanced InSAR Monitoring of Permafrost Thaw Subsidence". The software is designed to improve the accuracy of InSAR (Interferometric Synthetic Aperture Radar) data analysis by identifying and removing atmospheric phase components, enhancing the monitoring capabilities for permafrost thaw subsidence.

## Features

- **Atmospheric Phase Identification**: Automatically identifies atmospheric phases in InSAR data using data-driven techniques.
- **Phase Removal**: Implements algorithms to remove the identified atmospheric phase effects, leading to clearer and more accurate InSAR images.
- **Permafrost Monitoring**: Specifically tailored for monitoring permafrost thaw subsidence, crucial for understanding climate change impacts.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt

## Installation

To get started with this software, follow these simple steps:

1. **Prepare your data**: Ensure your input files are in the correct format as described in the `Data` section.

2. **Run the main script**: Use the following command to process your InSAR images:

   ```python
   from osgeo import gdal
   import numpy as np

   # Example function call
   calculate_statistics_and_remove_trend(
       "path_to_mask.tif",
       "path_to_data.tif",
       "path_to_output.tif"
   )
