import os
import numpy as np
import matplotlib.pyplot as plt

# --- Plot config ---
font_size = 14
microns_per_pixel = 1 / 21.9373  # µm per pixel
plt.rcParams.update({'font.size': font_size})
plt.rcParams['font.family'] = 'Helvetica'


def read_header_info(filepath):
    """Reads and parses the Elmitec LEEM file header, returning key metadata fields."""

    with open(filepath, 'rb') as f:
        header = f.read(512)  # Read enough for fileHeader and imageHeader

    total_size = os.path.getsize(filepath)

    # --- FileHeader fields ---
    uk_version = int.from_bytes(header[22:24], byteorder='little')

    image_width = int.from_bytes(header[40:42], byteorder='little') if uk_version >= 2 else None
    image_height = int.from_bytes(header[42:44], byteorder='little') if uk_version >= 2 else None

    attached_recipe_size = int.from_bytes(header[46:48], byteorder='little') if uk_version >= 7 else 0
    file_header_size = 104 + (128 if attached_recipe_size > 0 else 0)

    # --- ImageHeader fields (start after fileHeader) ---
    image_headersize_offset = file_header_size
    image_headersize = int.from_bytes(header[image_headersize_offset:image_headersize_offset + 2], byteorder='little')

    attached_markup_size = int.from_bytes(header[image_headersize_offset + 22:image_headersize_offset + 24],
                                          byteorder='little')
    attached_markup_size = 128 * ((attached_markup_size // 128) + 1) if attached_markup_size else 0

    leem_data_version = int.from_bytes(header[image_headersize_offset + 26:image_headersize_offset + 28],
                                       byteorder='little')
    # --- Total header size ---
    total_header_size = file_header_size + image_headersize + attached_markup_size + leem_data_version

    # --- Print summary ---
    print(f" UK Version: {uk_version}")
    print(f" Image size: {image_width} × {image_height} pixels")
    print(f" File size: {total_size} bytes")
    print(f" FileHeader size: {file_header_size} bytes")
    print(f" ImageHeader size: {image_headersize} bytes")
    print(f" Attached Markup size: {attached_markup_size} bytes")
    print(f" LEEM Data Version block: {leem_data_version} bytes")
    print(f" Header Size: {total_header_size} bytes")

    # --- Return all fields as a dictionary ---
    return {
        "uk_version": uk_version,
        "image_width": image_width,
        "image_height": image_height,
        "file_size": total_size,
        "file_header_size": file_header_size,
        "image_header_size": image_headersize,
        "attached_markup_size": attached_markup_size,
        "leem_data_version": leem_data_version,
        "total_header_size": total_header_size
    }


def read_dat_file(file_path, header_size, image_shape, dtype):
    with open(file_path, 'rb') as f:
        data = f.read()

    # Header bytes extracted but no printing of ASCII strings now
    header_bytes = data[:header_size]

    # Image data
    image_bytes = data[header_size:]
    if len(image_bytes) % 2 != 0:
        print(" Warning: Odd number of image bytes; trimming last byte.")
        image_bytes = image_bytes[:-1]

    expected_size = np.prod(image_shape)
    image_array = np.frombuffer(image_bytes, dtype=dtype)
    if image_array.size != expected_size:
        raise ValueError(f" Image shape {image_shape} requires {expected_size} pixels, "
                         f"but got {image_array.size} pixels.")

    # Flip vertically
    image = image_array.reshape(image_shape)[::-1, :]

    # Generate micron axes
    height, width = image.shape
    x_microns = np.arange(width) * microns_per_pixel
    y_microns = np.arange(height) * microns_per_pixel

    return header_bytes, image, x_microns, y_microns


def show_image(image, x_microns, y_microns):
    plt.figure(figsize=(8, 6))
    extent = [x_microns[0], x_microns[-1], y_microns[-1], y_microns[0]]  # Invert Y for vertical flip
    plt.imshow(image, cmap='hot', extent=extent, aspect='equal')
    plt.colorbar(label="Intensity [a.u.]")
    plt.xlabel("X [µm]")
    plt.ylabel("Y [µm]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #  Change to your .dat file
    file_path = "2025_06_04_Bi_on_40_nm_Ti2O3_on_Al2O3(0001)_LEEM_30mkm_0p50_eV_No__CA_No_ILA_Exp_1s_Ave_S_000001.dat"
    dtype = np.uint16  # 16-bit unsigned

    header_info = read_header_info(file_path)
    header_size = header_info["total_header_size"]

    #  Automatically use image dimensions from header
    image_shape = (header_info["image_height"], header_info["image_width"])

    if None in image_shape:
        raise ValueError(" Image dimensions are missing from header. Cannot proceed.")

    header_bytes, image, x_microns, y_microns = read_dat_file(
        file_path,
        header_size=header_size,
        image_shape=image_shape,
        dtype=dtype
    )

    show_image(image, x_microns, y_microns)
