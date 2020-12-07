import matplotlib.pyplot as plt
from read_pdb import pdb_content
import argparse
import png
import numpy as np
import copy

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("pdb_filenames", nargs="+")
arg_parser.add_argument("--chm_filename")
arg_parser.add_argument("--dem_filename")
arg_parser.add_argument("--marker_size", type=int)
arg_parser.add_argument("--color_list", nargs="+", type=str)
arg_parser.add_argument("--plot_stats", action="store_true")
arg_parser.add_argument("--real_distances", action="store_true")
arg_parser.add_argument("--titles", type=str, nargs="+")
arg_parser.add_argument("--plot_rect", type=int, nargs=4, help="Rectangle in which plants are allowed: X1 Y1 X2 Y2")
a = arg_parser.parse_args()

marker_size = a.marker_size if a.marker_size is not None else 1

if a.chm_filename is not None:
    if a.chm_filename.endswith(".png"):
        w, h, chm_data, info = png.Reader(a.chm_filename).read()
        chm_data = np.array(list(chm_data))
        chm_data = chm_data[:,::info["planes"]]
    else:
        with open(a.chm_filename, "r") as infile:
            lines = []
            for lnum, line in enumerate(infile):
                line = line.strip()
                if len(line) > 0:
                    line = line.split(" ")
                    if lnum == 0:
                        w, h = int(line[0]), int(line[1])
                    else:
                        line = [float(el) for el in line]
                        lines.append(line)
            chm_data = np.array(lines)
    if a.plot_rect is not None:
        chm_data = chm_data[a.plot_rect[1] : a.plot_rect[3], a.plot_rect[0] : a.plot_rect[2]]

else:
    chm_data = None

if a.dem_filename is not None:
    w, h, dem_data, info = png.Reader(a.dem_filename).read()
    dem_data = np.array(list(dem_data))
    dem_data = dem_data[:,::info["planes"]]
else:
    dem_data = None

pdbs = [pdb_content(pdbfname, normalize_distances=not a.real_distances) for pdbfname in a.pdb_filenames]
allspecies_list = [pdb.get_species_ids() for pdb in pdbs]
allspecies = set()
for allsp in allspecies_list:
    allspecies = allspecies.union(allsp)
allspecies = list(allspecies)
nspecies = len(allspecies)

notitle_count = 1
if a.titles:
    titles = copy.copy(a.titles)
    while len(titles) < len(a.pdb_filenames):
        titles.append("Untitled" + str(notitle_count))

color_list = a.color_list
if a.color_list is not None and nspecies != len(a.color_list):
    color_list = [a.color_list[i % len(a.color_list)] for i in range(nspecies)]
elif a.color_list is None:
    color_list = [(np.random.random(), np.random.random(), np.random.random()) for _ in range(nspecies)]

nplants = 0
maxr = 0
maxh = 0
radii = []
heights = []
plnt_xs, plnt_ys = [ [] for _ in range(len(pdbs))], [ [] for _ in range(len(pdbs))]
plnt_colors = [ [] for _ in range(len(pdbs))]
plotrect = [-float("inf"), -float("inf"), float("inf"), float("inf")]
if a.plot_rect:
    plotrect = a.plot_rect
for pdbidx, pdb in enumerate(pdbs):
    for plnt in pdb.all_plants():
        if plnt.x < plotrect[0] or plnt.x > plotrect[2] or plnt.y < plotrect[1] or plnt.y > plotrect[3]:
            continue
        plnt_xs[pdbidx].append(plnt.x)
        plnt_ys[pdbidx].append(plnt.y)
        plnt_colors[pdbidx].append(plnt.specie)
        if plnt.r > maxr:
            maxr = plnt.r
        if plnt.h > maxh:
            maxh = plnt.h
        radii.append(plnt.r)
        heights.append(plnt.h)
        nplants += 1
print("max radius: {}".format(maxr))
print("max height: {}".format(maxh))
print("Number of plants: {}".format(nplants))

if a.plot_stats:
    plt.hist(radii)
    plt.show()
    plt.hist(heights)
    plt.show()
#species = sorted(set(plnt_colors))

#print(plnt_colors[1])

#species = range(len(pdb.species))
#species = [specdict["spec_id"] for specdict in pdb.species]
print(len(allspecies))
print(allspecies)
specie_colors = {spec_id: color for spec_id, color in zip(allspecies, color_list)}

fig, ax = plt.subplots(1, len(pdbs))

for idx in range(len(pdbs)):
    colarr = np.array(plnt_colors[idx])
    unique, counts = np.unique(colarr, return_counts=True)
    for v, c in zip(unique, counts):
        print("Species {}, count: {}".format(v, c))
    plnt_plot_colors = [specie_colors[spec_id] for spec_id in plnt_colors[idx]]

    if chm_data is not None:
        extent = None
        if a.plot_rect is not None:
            extent = [a.plot_rect[0], a.plot_rect[2], a.plot_rect[3], a.plot_rect[1]]
        if len(pdbs) == 1:
            ax.imshow(chm_data, extent=extent)
        else:
            ax[idx].imshow(chm_data, extent=extent)
    if len(pdbs) == 1:
        ax.scatter(plnt_xs[idx], plnt_ys[idx], s=marker_size, c=plnt_plot_colors)
    else:
        ax[idx].scatter(plnt_xs[idx], plnt_ys[idx], s=marker_size, c=plnt_plot_colors)
    if chm_data is None:
        if len(pdbs) == 1:
            ax.set_ylim(ax.get_ylim()[::-1])
        else:
            ax[idx].set_ylim(ax[idx].get_ylim()[::-1])
    if a.titles:
        if len(pdbs) == 1:
            ax.title.set_text(titles[0])
        else:
            for i in range(len(pdbs)):
                ax[i].title.set_text(titles[i])
plt.show()
