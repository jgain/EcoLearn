from average_monthly_map import read_monthly_map, avg_map_monthly
import argparse
import matplotlib.pyplot as plt

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("filename")

a = arg_parser.parse_args()

arr = avg_map_monthly(a.filename)

plt.imshow(arr)
plt.show()
