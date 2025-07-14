import os
import struct
import bisect as bs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# --- Plot config ---
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
    "#BAB0AC"  # light gray
]
boundaries = np.arange(0, 141, 14)  # 0, 14, ..., 140
custom_cmap = ListedColormap(colors)
norm = BoundaryNorm(boundaries, ncolors=len(colors), clip=True)

# --- Device numbers to display on plot ---
important_device_numbers = [2, 3, 11, 30, 31, 38, 39]


def read_header_info(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(512)

    total_size = os.path.getsize(filepath)
    uk_version = int.from_bytes(header[22:24], byteorder='little')

    image_width = int.from_bytes(header[40:42], byteorder='little') if uk_version >= 2 else None
    image_height = int.from_bytes(header[42:44], byteorder='little') if uk_version >= 2 else None
    attached_recipe_size = int.from_bytes(header[46:48], byteorder='little') if uk_version >= 7 else 0
    file_header_size = 104 + (128 if attached_recipe_size > 0 else 0)

    image_headersize_offset = file_header_size
    image_headersize = int.from_bytes(header[image_headersize_offset:image_headersize_offset + 2], byteorder='little')
    BG = int.from_bytes(header[image_headersize_offset + 2:image_headersize_offset + 4], byteorder='little')
    Lowest_Image_Intensity = int.from_bytes(header[image_headersize_offset + 4:image_headersize_offset + 6], byteorder='little')
    Highest_Image_Intensity = int.from_bytes(header[image_headersize_offset + 6:image_headersize_offset + 8], byteorder='little')
    attached_markup_size = int.from_bytes(header[image_headersize_offset + 22:image_headersize_offset + 24], byteorder='little')
    leem_data_version = int.from_bytes(header[image_headersize_offset + 26:image_headersize_offset + 28], byteorder='little')

    attached_markup_size = 128 * ((attached_markup_size // 128) + 1) if attached_markup_size else 0
    total_header_size = file_header_size + image_headersize + attached_markup_size + leem_data_version

    print(f" UK Version: {uk_version}")
    print(f" Image size: {image_width} × {image_height} pixels")
    print(f" File size: {total_size} bytes")
    print(f" FileHeader size: {file_header_size} bytes")
    print(f" ImageHeader size: {image_headersize} bytes")
    print(f" Attached Markup size: {attached_markup_size} bytes")
    print(f" LEEM Data Version block: {leem_data_version} bytes")
    print(f" Header Size: {total_header_size} bytes")
    print(f" Lowest Image Intensity {Lowest_Image_Intensity} a.u.")
    print(f" Highest Image Intensity: {Highest_Image_Intensity} a.u.")
    print(f" Background: {BG} a.u.")

    return {
        "uk_version": uk_version,
        "image_width": image_width,
        "image_height": image_height,
        "file_size": total_size,
        "file_header_size": file_header_size,
        "image_header_size": image_headersize,
        "attached_markup_size": attached_markup_size,
        "leem_data_version": leem_data_version,
        "total_header_size": total_header_size,
        "Lowest_Image_Intensity": Lowest_Image_Intensity,
        "Highest_Image_Intensity": Highest_Image_Intensity,
        "BG": BG
    }

# def extract_leem_param adapted from code by Dr. Helder Marchetto (https://github.com/heldermarchetto) in the repository:
# https://github.com/heldermarchetto/ElmitecReadUviewFileFormat
# License: Apache License
def get_units(val: str) -> str:
    return {
        '0': 'none', '1': 'V', '2': 'mA', '3': 'A', '4': 'C',
        '5': 'K', '6': 'mV', '7': 'pA', '8': 'nA', '9': 'uA',
    }.get(val, 'none')

# def extract_leem_param adapted from code by Dr. Helder Marchetto (https://github.com/heldermarchetto) in the repository:
#  https://github.com/heldermarchetto/ElmitecReadUviewFileFormat
#  License: Apache License
def extract_leem_param(data, zero_list, start_pos):
    dev_nr = data[start_pos] & 0x7F
    end_name = zero_list[bs.bisect_left(zero_list, start_pos + 1)]

    device = {'number': dev_nr}
    raw_name = bytes(data[start_pos:end_name]).decode('utf-8', errors='ignore')
    device['name'] = ''.join(c if c.isprintable() else '?' for c in raw_name).strip()

    unit_byte = bytes(data[end_name - 1:end_name]).decode('utf-8', errors='ignore')
    device['units'] = get_units(unit_byte)

    try:
        device['value'] = struct.unpack('f', data[end_name + 1:end_name + 5])[0]
    except struct.error:
        device['value'] = None

    return device, end_name + 5

# def parse_leem_parameters adapted from code by Dr. Helder Marchetto (https://github.com/heldermarchetto) in the repository:
#  https://github.com/heldermarchetto/ElmitecReadUviewFileFormat
#  License: Apache License
def parse_leem_parameters(data):
    zero_list = [i for i, b in enumerate(data) if b == 0]
    if not zero_list:
        return []

    end_leem_data = zero_list[-1]
    pos = 0
    params = []

    while pos <= end_leem_data:
        param, pos = extract_leem_param(data, zero_list, pos)
        params.append(param)

    return params

# def read_dat_file adapted from code by Dr. Helder Marchetto (https://github.com/heldermarchetto) in the repository:
#  https://github.com/heldermarchetto/ElmitecReadUviewFileFormat
#  License: Apache License
def read_dat_file(file_path, header_size, image_shape, dtype):
    with open(file_path, 'rb') as f:
        data = f.read()

    header_bytes = data[:header_size]
    image_bytes = data[header_size:]

    if len(image_bytes) % 2 != 0:
        image_bytes = image_bytes[:-1]

    expected_size = np.prod(image_shape)
    image_array = np.frombuffer(image_bytes, dtype=dtype)

    if image_array.size != expected_size:
        raise ValueError(f"Expected {expected_size} pixels, got {image_array.size}")

    image = image_array.reshape(image_shape)[::-1, :]

    height, width = image.shape
    x_microns = np.arange(width) * microns_per_pixel
    y_microns = np.arange(height) * microns_per_pixel

    return header_bytes, image, x_microns, y_microns


def show_image(image, x_microns, y_microns, header_info, leem_params):
    fig, (ax_text, ax_img) = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [2, 4]})

    ax_text.axis("off")
    info_str = "\n".join([
        f"UK Version: {header_info['uk_version']}",
        f"Image size: {header_info['image_width']} × {header_info['image_height']}",
        f"File size: {header_info['file_size']} bytes",
        f"FileHeader: {header_info['file_header_size']} bytes",
        f"ImageHeader: {header_info['image_header_size']} bytes",
        f"Markup size: {header_info['attached_markup_size']} bytes",
        f"LEEM Data Ver.: {header_info['leem_data_version']} bytes",
        f"Total Header: {header_info['total_header_size']} bytes",
        f"Lowest Intensity: {header_info['Lowest_Image_Intensity']} a.u.",
        f"Highest Intensity: {header_info['Highest_Image_Intensity']} a.u.",
        f"Background: {header_info['BG']} a.u.",
        "",
        "Selected LEEM Parameters "
    ])

    # Add selected LEEM parameters to display
   # for param in leem_params:
       # if param["number"] in important_device_numbers:
            #info_str += f"\n{param['name']:<20} {param['value']:10.4f} {param['units']}"
    CUSTOM_PARAM_NAMES = {
         "Illum.Defl. X2": "Illuminator Deflector X:",
         "Illum.Defl. Y2": "Illuminator Deflector Y:",
         "?Objective2": "Objective Lens:",
         "Illum.Equal.X2": "Illuminator Equalization X:",
         "Illum.Equal.Y2": "Illuminator Equalization Y:",
         "&Start Voltage1": "Start Voltage:",
         "'Sample Temp.4": "Sample Temperature:",
         "Manipulator XB ": "Manipulator X:",
         "Manipulator YB": "Manipulator Y:"

    }

    for param in leem_params:
        if param["number"] in important_device_numbers:
            name = CUSTOM_PARAM_NAMES.get(param['name'], param['name'])  # map name if available
            info_str += f"\n{name:<25} {param['value']:10.4f} {param['units']}"
    ax_text.text(1, 1, info_str, fontsize=12, va='top', ha='right', family='monospace')

    extent = [x_microns[0], x_microns[-1], y_microns[-1], y_microns[0]]
    im = ax_img.imshow(image, cmap=custom_cmap, norm=norm, extent=extent, aspect='equal')
    ax_img.set_xlabel("X [µm]")
    ax_img.set_ylabel("Y [µm]")
    ax_img.set_title("LEEM Image")
    fig.colorbar(im, ax=ax_img, label="Intensity [a.u.]")

    plt.tight_layout()
    plt.savefig("image_with_header.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    file_path = "2025_06_04_Bi_on_40_nm_Ti2O3_on_Al2O3(0001)_LEEM_30mkm_0p50_eV_No__CA_No_ILA_Exp_1s_Ave_S_000001.dat"
    dtype = np.uint16

    header_info = read_header_info(file_path)

    start_pos = header_info["file_header_size"] + header_info["image_header_size"] + header_info["attached_markup_size"]
    with open(file_path, 'rb') as f:
        f.seek(start_pos)
        leem_data = f.read(header_info["leem_data_version"])

    parsed_params = parse_leem_parameters(leem_data)
    sorted_params = sorted(parsed_params, key=lambda x: x['number'])

    print("\n--- Parsed LEEM Parameters (Sorted by Device Number) ---")
    for p in sorted_params:
        print(f"{p['number']:3}: {p['name']:20} = {p['value']:10.4f} {p['units']}")

    image_shape = (header_info["image_height"], header_info["image_width"])
    if None in image_shape:
        raise ValueError("Image dimensions missing in header.")

    header_bytes, image, x_microns, y_microns = read_dat_file(
        file_path,
        header_size=header_info["total_header_size"],
        image_shape=image_shape,
        dtype=dtype
    )

    show_image(image, x_microns, y_microns, header_info, sorted_params)
