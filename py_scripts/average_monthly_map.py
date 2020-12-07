import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_monthly_map(filename, notherparams=0):
    if notherparams < 0 or notherparams > 2:
        raise ValueError("notherparams argument must be between 0 and 2, inclusive")
    data = []
    with open(filename, "r") as infile:
        for lnum, line in enumerate(infile):
            line = line.strip()
            line = line.split(" ")
            if lnum == 0:
                width, height = int(line[0]), int(line[1])
                if notherparams == 1:
                    try:
                        step = float(line[2])
                    except IndexError:
                        raise ValueError("{} does not contain a step parameter".format(filename))
                elif len(line) > 2:
                    step = float(line[2])
                    print("NOTE: step parameter not requested, but present in file. step size: {}".format(step))
            else:
                #line = np.array([float(el) for el in line])
                line = [float(el.strip()) for idx, el in enumerate(line) if len(el.strip()) > 0]
                data.append(line)
                #arr = np.array(line).reshape(height, width, 12)
    #arr = np.concatenate(data)
    arr = np.array(data)
    arr = arr.reshape(height, width, 12)
    if notherparams == 0:
        return arr
    else:
        return arr, step

def write_monthly_map(filename, arr, step=None):
    h, w, _ = arr.shape
    arr = arr.reshape((arr.shape[0], arr.shape[1] * 12))
    with open(filename, "w+") as outfile:
        if step is None:
            wh_str = "{} {}\n".format(w, h)
        else:
            wh_str = "{} {} {}\n".format(w, h, step)
        outfile.write(wh_str)
        for rownum in range(h):
            rowvals = arr[rownum,:]
            vals_str = ["{} " for _ in range(w * 12)]
            vals_str[-1] = vals_str[-1][:-1]
            vals_str += "\n"
            vals_str = "".join(vals_str)
            vals_str = vals_str.format(*list(rowvals))
            outfile.write(vals_str)


def avg_map_monthly(filename):
    arr = read_monthly_map(filename)
    arr = np.mean(arr, axis=2)
    return arr

def write_arr_to_elv(out_filename, arr, step, lat):
    with open(out_filename, "w") as outfile:
        topline = "{} {} {} {}\n".format(arr.shape[1], arr.shape[0], step, lat)
        outfile.write(topline)
        for i in range(arr.shape[0]):
            currline = arr[i,:].tolist()
            currline = [str(el) for el in currline]
            currline = " ".join(currline)
            outfile.write(currline)
            outfile.write("\n")

def write_arr_to_elvlike(out_filename, arr):
    with open(out_filename, "w") as outfile:
        topline = "{} {}\n".format(arr.shape[1], arr.shape[0])
        outfile.write(topline)
        for i in range(arr.shape[0]):
            currline = arr[i,:].tolist()
            currline = [str(el) for el in currline]
            currline = " ".join(currline)
            outfile.write(currline)
            outfile.write("\n")

if __name__ == "__main__":


    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("monthmap")
    arg_parser.add_argument("output_filename")
    arg_parser.add_argument("--plot", action="store_true")
    a = arg_parser.parse_args()
    
    arr = avg_map_monthly(a.monthmap)
    
    write_arr_to_elvlike(a.output_filename, arr)

    print("Average monthly map from {} written to {}".format(a.monthmap, a.output_filename))

    #print(arr.shape)
    
    if a.plot:
        plt.imshow(arr)
        plt.show()
