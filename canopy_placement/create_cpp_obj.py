import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("obj_filename")
arg_parser.add_argument("output_basename")

a = arg_parser.parse_args()
output_hfile = a.output_basename + ".h"
output_cppfile = a.output_basename + ".cpp"

hfile_string = "extern std::string sphere_obj_contents;"

cppfile_string = "#include <string>\n"
cppfile_string += "std::string sphere_obj_contents = \n"

with open(a.obj_filename, "r") as infile:
    for line in infile:
        line = line[:-1]
        cppfile_string += "\"" + line + "\\n\"\n"
    cppfile_string += ";"

with open(output_hfile, "w+") as outfile:
    outfile.write(hfile_string)

with open(output_cppfile, "w+") as outfile:
    outfile.write(cppfile_string)
