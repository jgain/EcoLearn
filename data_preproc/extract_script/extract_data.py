import subprocess
import sys
import argparse
import os
import shutil
import sqlite3
import numpy as np
import sys

def generate_random_simspec(nsims, concurrent_subbiomes_perc, sql_db_filename, out_filename):
    def dict_factory(cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    def normalize_dict(d, dname):
        csum = 0
        for key, value in d.items():
            csum += value
        if csum <= 1e-5:
            raise ValueError("Sum of {} values must not be zero or negative".format(dname))
        for key in d:
            d[key] = d[key] / csum

    def create_cumul_dict(d):
        cumul_d = dict(d)
        keys = [k for k in d]
        csum = 0
        for k in keys:
            csum += d[k]
            cumul_d[k] = csum
        return keys, cumul_d

    def sample_from_probdict(probdict, keys_ordered):
        rnum = np.random.uniform()
        for k in keys_ordered:
            if rnum < sub_csums[k]:
                return k
        return None     # this indicates there is a problem with the probability dictionary

    conn = sqlite3.connect(sql_db_filename)
    conn.row_factory = dict_factory
    cursor = conn.cursor()
    biome_ids = []
    for row in cursor.execute("SELECT Sub_biome_ID FROM subBiomes"):
        biome_ids.append(row["Sub_biome_ID"])
    
    subbiomes = {i: [] for i in biome_ids}
    for row in cursor.execute("SELECT Sub_biome_ID, Tree_ID FROM subBiomesMapping WHERE Canopy = 1"):
        subbiomes[row["Sub_biome_ID"]].append(row["Tree_ID"])
    
    sub_cc = concurrent_subbiomes_perc
    normalize_dict(sub_cc, "concurrent_subbiomes_perc")
    keys, sub_csums = create_cumul_dict(sub_cc)

    nsbiomes = None
    filestr = ""
    for sim_idx in range(nsims):
        nsbiomes = sample_from_probdict(sub_csums, keys)
        assert(nsbiomes is not None)
        sim_subbiomes = list(np.random.choice(keys, size=nsbiomes, replace=False))
        sub_probs = {}
        for sb in sim_subbiomes:
            sub_probs[sb] = np.random.uniform() * 0.6 + 0.2
        normalize_dict(sub_probs, "sub_probs")
        for sb, prob in sub_probs.items():
            filestr += str(sb) + " " + str(prob) + " "
        filestr.strip()
        filestr += "\n"
        for sb in sim_subbiomes:
            species = subbiomes[sb]
            probs = np.array([np.random.uniform() * 0.6 + 0.2 for _ in species])
            probs = probs / np.sum(probs)
            probs = list(probs)
            for sp, p in zip(species, probs):
                filestr += str(sb) + " " + str(sp) + " " + str(p) + "\n"
    filestr = filestr[:-1]
    with open(out_filename, "w") as outfile:
        outfile.write(filestr)

# remove first six lines of file and write in a new header
def strip_header(stripfile, newheader):
    out_lines = [newheader]
    with open(stripfile, "r") as infile:
        for idx, line in enumerate(infile):
            if idx < 6:  # skip first six lines
                continue
            out_lines.append(line)

    with open(stripfile, "w+") as outfile:
        for line in out_lines:
            outfile.write(line)


def main():
    
    # parse command line arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("source_dem", type=str, help="Source GeoTiff file containing the DEM")
    arg_parser.add_argument("source_chm", type=str, help="Source GeoTiff file containing the CHM")
    arg_parser.add_argument("source_cdm", type=str, help="Source GeoTiff file containing the CDM")
    arg_parser.add_argument("start_x", type=int, help="starting x-coordinate of crop region")
    arg_parser.add_argument("start_y", type=int, help="starting y-coordinate of crop region")
    arg_parser.add_argument("extent_x", type=int, help="x-extent of crop region")
    arg_parser.add_argument("extent_y", type=int, help="y-extent of crop region")
    arg_parser.add_argument("db_filename", type=str, help="Filename of SQL database containing base data")
    arg_parser.add_argument("nsims", type=int, help="Number of simulations to run")
    arg_parser.add_argument("moisture_program", type=str, help="Program that does moisture flow simulation")
    arg_parser.add_argument("sunlight_program", type=str, help="Program that does sunlight simulation")
    arg_parser.add_argument("slope_program", type=str, help="Program that calculates slope on landscape")
    arg_parser.add_argument("temp_program", type=str, help="Program that calculates temperature on landscape")
    arg_parser.add_argument("species_assign_program", type=str, help="Program that assigns species to the canopy footprint")
    arg_parser.add_argument("canopy_placement_program", type=str, help="Program that does canopy placement on CHM")
    arg_parser.add_argument("undersim_program", type=str, help="Program that does undergrowth simulation")
    arg_parser.add_argument("fin_dir", type=str, help="Final output directory name")
    arg_parser.add_argument("subbiome_count_prob", type=float, nargs="+", help="Probabilities for having a given count of subbiomes. The first argument is \
            the probability to have one subbiome, the second, the probability to have 2, etc.")
    arg_parser.add_argument("nyears_sim", type=int, help="Number of years to simulate undergrowth")
    arg_parser.add_argument("--ignore_undersim", action="store_true")
    arg_parser.add_argument("--ignore_bio", action="store_true")
    a = arg_parser.parse_args()
    all_subbiome_sum = sum(a.subbiome_count_prob)
    if any([p < 0 for p in a.subbiome_count_prob]):
        raise ValueError("subbiome_count_prob arguments must be zero or positive")
    if all_subbiome_sum < 1e-5:
        raise ValueError("subbiome_count_prob arguments must sum to a nonzero, positive value")
    a.subbiome_count_prob = [v / all_subbiome_sum for v in a.subbiome_count_prob]
    subbiome_probs = {i + 1: prob for i, prob in enumerate(a.subbiome_count_prob)}
    if a.nsims <= 0:
        raise ValueError("nsims must be positive, nonzero")
    
    if a.start_x+a.extent_x > 22895 or a.start_x < 0:
        print("Out of bounds. Minimum x is 0 and maximum x is 22895")
        exit
    if a.start_y+a.extent_y > 19849 or a.start_y < 0:
        print("Out of bounds. Minimum y is 0 and maximum y is 19849")
        exit
    
    # make directory in current folder
    #fin_path = os.path.join(os.getcwd(),a.fin_dir)
    fin_path = a.fin_dir.rstrip(os.sep)
    if not os.path.exists(fin_path):
        os.mkdir(fin_path)
    else:
        while os.path.exists(fin_path):
            print("Error: desired output directory already exists as a file or a directory. Appending -1 to directory name")
            fin_path += "-1"
            print("Directory is now: {}".format(fin_path))
        os.mkdir(fin_path)
    directory = os.path.basename(fin_path)

    simspec_out = os.path.join(fin_path, directory + "_simspec.txt")
    generate_random_simspec(a.nsims, subbiome_probs, a.db_filename, simspec_out)
   
    # copy and rename biome and climate files
    base_biome = os.path.join(os.getcwd(),"base_biome.txt")
    if not os.path.exists(base_biome):
        print("base_biome.txt not found in base directory where script is executed")
        exit()
    fin_biome = os.path.join(fin_path, directory+"_biome.txt")
    shutil.copy(base_biome, fin_biome)

    base_clim = os.path.join(os.getcwd(),"base_clim.txt")
    if not os.path.exists(base_clim):
        print("base_clim.txt not found in base directory where script is executed")
        exit()
    fin_clim = os.path.join(fin_path, directory+"_clim.txt")
    shutil.copy(base_clim, fin_clim)

    base_species_params = os.path.join(os.getcwd(), "base_species_params.txt")
    if not os.path.exists(base_species_params):
        print("base_species_params.txt not found in base directory where script is executed")
        exit()
    fin_species_params = os.path.join(fin_path, directory + "_species_params.txt")
    shutil.copy(base_species_params, fin_species_params)

    # crop and convert elevation, canopy height, canopy density files
    base_elv = os.path.join(a.source_dem)
    base_chm = os.path.join(a.source_chm)
    base_cdm = os.path.join(a.source_cdm)
    #base_cdm = os.path.join(os.getcwd(),"../output_cd.tif")
    fin_elv = os.path.join(fin_path, directory+".elv")
    fin_chm = os.path.join(fin_path, directory+".chm")
    #fin_cdm = os.path.join(fin_path, directory+".cdm")
    #fin_cdm = os.path.join(fin_path, a.fin_dir+".cdm")
    # bottom left zero coordinates in Sonoma county elevation and plant data are
    # bx = 6212545, by = 1996315
    subprocess.run(['gdal_translate', '-of', 'AAIGrid', '-srcwin', str(a.start_x), str(a.start_y), str(a.extent_x), str(a.extent_y),  base_elv, fin_elv])
    subprocess.run(['gdal_translate', '-of', 'AAIGrid', '-srcwin', str(a.start_x), str(a.start_y), str(a.extent_x), str(a.extent_y), base_chm, fin_chm])
    #subprocess.run(['gdal_translate', '-of', 'AAIGrid', '-srcwin', str(a.start_x), str(a.start_y), str(a.extent_x), str(a.extent_y), base_cdm, fin_cdm])

    # clean up headers
    strip_header(fin_elv, str(a.extent_x) + " " + str(a.extent_y) + " 0.9144 38.5\n")
    strip_header(fin_chm, str(a.extent_x) + " " + str(a.extent_y) + "\n")
    #strip_header(fin_cdm, str(a.extent_x) + " " + str(a.extent_y) + "\n")

    # delete extraneous files
    subprocess.run(['rm', fin_elv+'.aux.xml'])
    subprocess.run(['rm', fin_chm+'.aux.xml'])
    #subprocess.run(['rm', fin_cdm+'.aux.xml'])
    fin_prj = os.path.join(fin_path, directory+".prj")
    subprocess.run(['rm', fin_prj])

    moisture_data = os.path.join(fin_path, directory + "_wet.txt")
    sunlight_data = os.path.join(fin_path, directory + "_sun_landscape.txt")
    slope_data = os.path.join(fin_path, directory + "_slope.txt")
    temp_data = os.path.join(fin_path, directory + "_temp.txt")
    """
    subprocess.run([a.moisture_program, fin_elv, moisture_data])
    subprocess.run([a.sunlight_program, fin_elv, sunlight_data])
    subprocess.run([a.slope_program, fin_elv, slope_data])
    subprocess.run([a.temp_program, fin_elv, temp_data])
    """
    print("Running moisture program", file=sys.stderr)
    subprocess.run([a.moisture_program, fin_path, a.db_filename], check=True)
    print("Running sunlight program", file=sys.stderr)
    subprocess.run([a.sunlight_program, fin_path, a.db_filename], check=True)
    print("Running slope program", file=sys.stderr)
    subprocess.run([a.slope_program, fin_path], check=True)
    print("Running temperature program", file=sys.stderr)
    subprocess.run([a.temp_program, fin_path, a.db_filename], check=True)
    if not a.ignore_bio:
        print("Running species assign program", file=sys.stderr)
        subprocess.run([a.species_assign_program, fin_path, a.db_filename], check=True)
        print("Running canopy_placement program", file=sys.stderr)
        subprocess.run([a.canopy_placement_program, fin_path, a.db_filename], check=True)
        if not a.ignore_undersim:
            print("Running understorey simulation program", file=sys.stderr)
            subprocess.run([a.undersim_program, fin_path, "0", str(a.nyears_sim)], check=True)


    # code here to run automated tree placement to derive pdb plant file
    
    # compensate for difference
    # top left corner as input, and x-y extent
    # SONOMA cdm origin bx = 6212545, by = 2055862 equiv to (0,0) at top left
    bx = 6212545
    by = 2055862
    
    # list of land types
    lndtype = ['Hardwood Forest','Conifer Forest', 'Mixed Conifer-Hardwood Forest', 'Riparian Forest', 'Non-native Forest', 'Forest Sliver',
            'Shrub', 'Riparian Shrub', 'Herbaceous', 'Herbaceous Wetland', 'Aquatic Vegetation', 'Salt Marsh', 'Barrren and Sparsely Vegetated',
            'Agriculture', 'Water', 'Developed']
    #lndtype = ['Hardwood Forest','Conifer Forest', 'Mixed Conifer-Hardwood Forest', 'Shrub'] 
    lndcode = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117] # unique greyscale colour for type

    # set bounds for extraction
    sx = bx+(a.start_x*3)    # left
    sy = by-(a.start_y*3)-(a.extent_y*3)    # start_y is top edge, convert to bottom edge
    ex = a.extent_x*3
    ey = a.extent_y*3

    # sub types needed but a good start
    
    """
    # rasterize and composite
    cnt = 0
    for t in lndtype:
        # rasterize shape for specific type to a tif file
        outfileTIF = './output'+str(cnt)+'.tif'
        base_plt = os.path.join(os.getcwd(),"Sonoma_Veg_Map_5_1.shp")
        fin_plt = os.path.join(fin_path, a.fin_dir+"_plt.png")
        
        subprocess.run(['gdal_rasterize', '-ot', 'Byte', '-l', 'Sonoma_Veg_Map_5_1',
                    '-where', r"LF_FOREST LIKE '"+t+r"'",
                    '-burn', str(lndcode[cnt]), '-te', str(sx), str(sy), str(sx+ex), str(sy+ey), '-tr', '3', '3',
                    base_plt, outfileTIF])
        # convert from tif to png
        if cnt == 0:
            outfilePNG = fin_plt
        else:
            outfilePNG = './output'+str(cnt)+'.png'
        subprocess.run(['gdal_translate', '-of', 'PNG', outfileTIF, outfilePNG])
        # make black background transparent  
        if cnt > 0:
             subprocess.run(['convert', outfilePNG, '-transparent', 'black', outfilePNG])
             # composite into final file output
             subprocess.run(['composite', outfilePNG, fin_plt, fin_plt])
             # delete intermediate files
             subprocess.run(['rm', outfilePNG])
        subprocess.run(['rm', outfilePNG+'.aux.xml'])
        subprocess.run(['rm', outfileTIF])
        cnt+=1
    """
    
if __name__ == "__main__":
    main()
    
    
    
