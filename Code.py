from osgeo import gdal
import numpy as np

def calculate_statistics_and_remove_trend(input_tif_a, input_tif_b, output_tif):
    # Open TIF image A (mask file)
    try:
        dataset_a = gdal.Open(input_tif_a, gdal.GA_ReadOnly)
        if dataset_a is None:
            raise FileNotFoundError(f"Failed to open {input_tif_a}")
    except FileNotFoundError as e:
        print(e)
        return

    band_a = dataset_a.GetRasterBand(1)
    mask_array = band_a.ReadAsArray()
    nodata_a = band_a.GetNoDataValue()

    # Open TIF image B (data file)
    try:
        dataset_b = gdal.Open(input_tif_b, gdal.GA_ReadOnly)
        if dataset_b is None:
            raise FileNotFoundError(f"Failed to open {input_tif_b}")
    except FileNotFoundError as e:
        print(e)
        return

    band_b = dataset_b.GetRasterBand(1)
    data_array = band_b.ReadAsArray()
    nodata_b = band_b.GetNoDataValue()

    # Ensure arrays have consistent dimensions
    if mask_array.shape != data_array.shape:
        raise ValueError("Mask and data arrays have inconsistent shapes!")

    # Initialize statistics storage
    class_stats = {}

    # Prepare x, y coordinates for trend fitting
    x_coords, y_coords = np.meshgrid(
        np.arange(data_array.shape[1]),  # X-coordinate
        np.arange(data_array.shape[0])   # Y-coordinate
    )

    # Loop through classes (1, 2, 3)
    for cls in [1, 2, 3]:
        # Create mask for the current class
        mask = (mask_array == cls) & (mask_array != nodata_a) & (data_array != nodata_b)
        data_class = data_array[mask]
        x_class = x_coords[mask]
        y_class = y_coords[mask]

        # Remove NaN values from data
        valid_mask = ~np.isnan(data_class)
        data_class = data_class[valid_mask]
        x_class = x_class[valid_mask]
        y_class = y_class[valid_mask]

        if data_class.size == 0:
            print(f"No valid data for class {cls}. Skipping.")
            continue

        # Calculate mean for the current class
        mean_value = np.mean(data_class)
        class_stats[cls] = {"mean": mean_value}

        print(f"Class {cls}: Mean={mean_value}")

        # Subtract class mean
        data_array[mask] -= mean_value
        data_class -= mean_value  # Subtract mean from the class-specific array for trend fitting

        # Fit a trend for this class
        if data_class.size > 3:  # At least 3 points needed for fitting
            # Create design matrix for linear regression (x, y -> z)
            A = np.vstack([x_class, y_class, np.ones_like(x_class)]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, data_class, rcond=None)  # Linear regression coefficients

            # Calculate trend: z = a*x + b*y + c
            trend = coeffs[0] * x_coords + coeffs[1] * y_coords + coeffs[2]

            # Subtract trend from data array for this class
            data_array[mask] -= trend[mask]

            print(f"Class {cls}: Trend coefficients a={coeffs[0]}, b={coeffs[1]}, c={coeffs[2]}")

    # Mask out NoData values in the output
    if nodata_b is not None:
        data_array[data_array == nodata_b] = np.nan  # Optional: Set nodata as NaN for clarity
        nodata_value = nodata_b
    else:
        nodata_value = -9999  # Default NoData value if not provided
        data_array[mask_array == nodata_a] = nodata_value

    # Create the output dataset
    driver = gdal.GetDriverByName("GTiff")
    dataset_out = driver.Create(output_tif, dataset_b.RasterXSize, dataset_b.RasterYSize, 1, gdal.GDT_Float32)
    dataset_out.SetProjection(dataset_b.GetProjection())
    dataset_out.SetGeoTransform(dataset_b.GetGeoTransform())

    # Write the updated data array
    band_out = dataset_out.GetRasterBand(1)
    band_out.WriteArray(data_array)
    band_out.SetNoDataValue(nodata_value)  # Set NoData value for the output file
    band_out.FlushCache()

    # Release resources
    dataset_a = None
    dataset_b = None
    dataset_out = None

    print(f"Output written to {output_tif}")

# Example usage
calculate_statistics_and_remove_trend(
    r"G:\code\arcpy\ganshetufenlei\20220301_20220524.tif",
    r"G:\code\arcpy\quanbuganshtu-nodata\20220301_20220524.tif",
    r"G:\code\arcpy\ganshetuquchu\20220301_20220524-correct.tif"
)