import sqlite3
import pandas as pd
import os
import numpy as np
from create_sql_db import filenames

PRIMARY = 2
FOREIGN = 1
NONE = 0

class columns:
    def __init__(self, colstr_list, tablename):
        self.cols = []
        self.colstr_list = [el.strip() for el in colstr_list if len(el.strip()) > 0]
        self.colstr_list_proc = [colname.strip().replace(" ", "_") for colname in self.colstr_list]
        self.has_primary = False
        self.tablename = tablename
        for name in self.colstr_list_proc:
            if name.endswith("_PRIMARY"):
                if self.has_primary:
                    raise ValueError("TABLE {} in file {} has more than one primary key".format(self.tablename, filenames[self.tablename]))
                else:
                    self.has_primary = True
                self.cols.append((name[: -len("_PRIMARY")], PRIMARY))
            elif name.endswith("_FOREIGN"):
                self.cols.append((name[: -len("_FOREIGN")], FOREIGN))
            else:
                self.cols.append((name, NONE))

    def assert_eq(self, dbcols):
        colnames = [tup[0] for tup in self.cols]
        return dbcols == colnames
    
    def gen_csv_colstr(self):
        csvstr = ""
        for el in self.colstr_list:
            csvstr += el + ","
        if len(csvstr) > 0:
            return csvstr[: -1]

def make_columns_from_csv(tablename):
    prefix = "/home/konrad/EcoSynth/data_preproc/common_data/"
    filename = filenames[tablename]
    filename = prefix + filename
    with open(filename, "r") as infile:
        for line in infile:
            line = line.strip().split(",")
            break
    return columns(line, tablename)

def gen_line(values):
    csv_str = ""
    values = list(values)
    for idx, v in enumerate(values):
        csv_str += "{},"
        if v == "nan":
            values[idx] = ""
    csv_str = csv_str[:-1] + "\n"
    return csv_str.format(*values)

def gen_csv(cursor, tablename):
    cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='{}'".format(tablename))
    count = cursor.fetchone()[0]
    print("TABLE COUNT: {}".format(count))
    if count > 0:
        cursor.execute("SELECT * FROM {}".format(tablename))
        cols = make_columns_from_csv(tablename)
        csv_str = cols.gen_csv_colstr() + "\n"
        for val_line in cursor.fetchall():
            csv_str += gen_line(val_line)
        return csv_str
    else:
        print("WARNING: table {} does not exist in sql database".format(tablename))
        return None


conn = sqlite3.connect("/home/konrad/EcoSynth/ecodata/sonoma.db")
cursor = conn.cursor()

out_dir = "/home/konrad/EcoSynth/data_preproc/common_data/adapted_csvs"

if os.path.isfile(out_dir):
    raise ValueError("{} already exists as a file, not a directory. Aborting".format(out_dir))
elif not os.path.exists(out_dir):
    os.makedirs(out_dir)

for key, value in filenames.items():
    out_filename = os.path.join(out_dir, value)
    csvstr = gen_csv(cursor, key)
    if csvstr is not None:
        print(csvstr)
        if os.path.exists(out_filename) and not os.path.isdir(out_filename):
            print("Removing existing file {}".format(out_filename))
            os.remove(out_filename)
        with open(out_filename, "w+") as outfile:
            print("Writing {}".format(out_filename))
            outfile.write(csvstr)
    else:
        print("Skipping file {}, because table {} does not exist in DB".format(out_filename, value))
    print("----------------------------------------------------------------")

"""

cursor.execute("PRAGMA table_info(species)")

print(cursor.fetchall())

print(filenames)

exit(0)

cursor.execute("SELECT * FROM species")
print(cursor.lastrowid)
for n in dir(cursor):
    print(n)
print(cursor.description)
exit(0)
print(cursor.description)
for line in cursor.fetchall():
    print(line)
"""
