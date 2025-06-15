import os
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.colors import ListedColormap, BoundaryNorm

# Plot config
font_size = 14
microns_per_pixel = 1 / 21.9373  # µm per pixel
plt.rcParams.update({'font.size': font_size})
plt.rcParams['font.family'] = 'Helvetica'

colors = [
    "#4E79A7",  # modern muted blue
    "#F28E2B",  # modern orange
    "#E15759",  # modern red
    "#76B7B2",  # modern teal
    "#59A14F",  # modern green
    "#EDC948",  # modern yellow
    "#B07AA1",  # modern purple
    "#FF9DA7",  # soft pink
    "#9C755F",  # warm brown
    "#BAB0AC"   # light gray
]
boundaries = np.arange(0, 141, 14)  # 0, 14, ..., 140
custom_cmap = ListedColormap(colors)
norm = BoundaryNorm(boundaries, ncolors=len(colors), clip=True)

def get_header_size(filepath, width=1024, height=1024, bytes_per_pixel=2):
    total_size = os.path.getsize(filepath)
    image_data_size = width * height * bytes_per_pixel
    header_size = total_size - image_data_size
    print(f" Total file size: {total_size} bytes")
    print(f" Expected image data size: {image_data_size} bytes")
    print(f" Header size: {header_size} bytes")
    return header_size

def read_dat_file(file_path, header_size, image_shape, dtype):
    with open(file_path, 'rb') as f:
        data = f.read()

    # Header
    header_bytes = data[:header_size]
    header_ascii = ''.join([chr(b) if 32 <= b < 127 else '?' for b in header_bytes])
    print("\n Printable ASCII strings in header:\n")
    for match in re.finditer(r"[A-Za-z0-9\.\'\& ]{5,}", header_ascii):
        print(f"{match.start():04d}: {match.group()}")

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

    return header_ascii, image, x_microns, y_microns

def show_image(image, x_microns, y_microns):
    plt.figure(figsize=(8, 6))
    extent = [x_microns[0], x_microns[-1], y_microns[-1], y_microns[0]]  # Invert Y
    plt.imshow(image, cmap=custom_cmap, norm=norm, extent=extent, aspect='equal')
    plt.colorbar(label="Intensity [a.u.]")
    plt.xlabel("X [µm]")
    plt.ylabel("Y [µm]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "2025_06_04_Bi_on_40_nm_Ti2O3_on_Al2O3(0001)_LEEM_30mkm_0p50_eV_No__CA_No_ILA_Exp_1s_Ave_S_000001.dat"
    image_shape = (1024, 1024)
    dtype = np.uint16  # 16-bit unsigned

    header_size = get_header_size(file_path, width=image_shape[1], height=image_shape[0], bytes_per_pixel=2)

    header_ascii, image, x_microns, y_microns = read_dat_file(
        file_path,
        header_size=header_size,
        image_shape=image_shape,
        dtype=dtype
    )

    show_image(image, x_microns, y_microns)
