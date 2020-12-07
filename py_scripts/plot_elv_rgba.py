import argparse
import numpy as np
import matplotlib.pyplot as plt

def convert_uint_to_rgba(arr):
    arr_conv = np.zeros((arr.shape[0], arr.shape[1], 4)).astype(np.uint8)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            r = arr[y,x] % 256
            g = arr[y,x] % (256 * 256) - r
            b = arr[y,x] % (256 * 256 * 256) - g - r
            a = arr[y,x] - r - g - b
            g = g / 256
            b = b / (256 * 256)
            a = a / (256 * 256 * 256)
            assert(r >= 0 and r < 256)
            assert(g >= 0 and g < 256)
            assert(b >= 0 and b < 256)
            assert(a >= 0 and a < 256)
            arr_conv[y, x, 0] = r
            arr_conv[y, x, 1] = g
            arr_conv[y, x, 2] = b
            arr_conv[y, x, 3] = a
    return arr_conv

def read_elv_rgba(filename):
    with open(filename, "r") as infile:
        arr = []
        for lnum, line in enumerate(infile):
            line = line.strip()
            line = line.split(" ")
            if len(line) > 0:
                if lnum == 0:
                    width, height = int(line[0]), int(line[1])
                else:
                    line = [float(el) for el in line]
                    #if lnum < height / 2:
                    #    line = [el for el in line]
                    arr.append(line)
        arr = np.array(arr).astype(np.uint32)
    return arr

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("elv_filenames", type=str, nargs="+")
    arg_parser.add_argument("--scales", type=str, nargs="+")
    arg_parser.add_argument("--mults", type=int, nargs="+", help="Multiplier for RGBA values (if there is not enough variation)")

    a = arg_parser.parse_args()

    nfiles = len(a.elv_filenames)

    if a.scales is None:
        a.scales = []
    while len(a.scales) < nfiles * 2:
        a.scales.append("None")

    if a.mults is None:
        a.mults = []
    while len(a.mults) < nfiles:
        a.mults.append(None)

    scales = a.scales[:]
    mults = a.mults[:]

    for i in range(nfiles * 2):
        try:
            num = float(scales[i])
            scales[i] = num
        except ValueError:
            scales[i] = None

    arrs = [_ for _ in range(nfiles)]

    for i in range(nfiles):
        elv_filename = a.elv_filenames[i]
        arrs[i] = read_elv_rgba(elv_filename)
        if (mults[i] is not None):
            arrs[i] = arrs[i] * mults[i]
        arrs[i] = convert_uint_to_rgba(arrs[i])

    fig, axes = plt.subplots(1, nfiles)

    if nfiles == 1:
        vmin, vmax = scales[0], scales[1]
        axes.imshow(arrs[0])
    else:
        for i in range(nfiles):
            vmin = scales[i * 2]
            vmax = scales[i * 2 + 1]
            axes[i].imshow(arrs[i], vmin=vmin, vmax=vmax)
    plt.show()
