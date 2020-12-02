import argparse
import subprocess

basedir = "/home/konrad/EcoSynth"

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("start_x", type=int, help="starting x-coordinate of crop region")
arg_parser.add_argument("start_y", type=int, help="starting y-coordinate of crop region")
arg_parser.add_argument("extent_x", type=int, help="x-extent of crop region")
arg_parser.add_argument("extent_y", type=int, help="y-extent of crop region")
arg_parser.add_argument("fin_dir", type=str, help="Final output directory name")
arg_parser.add_argument("--ignore_undersim", action="store_true")
arg_parser.add_argument("--ignore_bio", action="store_true")
a = arg_parser.parse_args()

arglist = ["python3", "extract_data.py", 
                "/home/konrad/EcoLearn/Data/output_be.tif", 
                "/home/konrad/EcoLearn/Data/output_vh.tif",
                "/home/konrad/EcoLearn/Data/output_cd.tif",
                str(a.start_x),
                str(a.start_y),
                str(a.extent_x),
                str(a.extent_y),
                basedir + "/data_preproc/common_data/sonoma.db",
                "5",
                basedir + "/data_preproc/build-default/waterfill/waterfill",
                basedir + "/data_preproc/build-default/sunlight_sim/main",
                basedir + "/data_preproc/build-default/slope_compute/compute_slope",
                basedir + "/data_preproc/build-default/temp_compute/temp_compute",
                basedir + "/data_preproc/build-default/species_assign/assign_species_single",
                basedir + "/data_preproc/build-default/canopy_placement/canopy_placement",
                basedir + "/viewer/build-UnderSim-Desktop-Default/viewer/viewer",
                a.fin_dir,
                "0.6",
                "0.3",
                "0.1",
                "150"]

if a.ignore_undersim:
    arglist.append("--ignore_undersim")
if a.ignore_bio:
    arglist.append("--ignore_bio")

subprocess.run(arglist, check=True)

