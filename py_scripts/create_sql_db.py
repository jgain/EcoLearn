import argparse
import sqlite3
import pandas as pd
import os
import numpy as np
from create_specratio_csv import create_specratios

class keyref:
    def __init__(self, keyname, ftype, reftable_name, srctable_name):
        self.name = keyname
        self.reftable_name = reftable_name    # the table being referenced by the foreign key (in which the foreign key is primary key)
        self.srctable_name = srctable_name    # the table in which the foreign key resides
        self.ftype = ftype

    def __str__(self):
        return "in TABLE {}: FOREIGN KEY {} REFERENCES TABLE {}. TYPE: {}".format(self.srctable_name, self.name, self.reftable_name, self.ftype)

class field:
    def __init__(self, fname, ftype):
        self.name = fname
        self.ftype = ftype

class db_dataframe:
    def __init__(self, dfname, all_dfs):
        self.primstr = "_PRIMARY"
        self.frstr = "_FOREIGN"
        self.dfname = dfname
        df = all_dfs[dfname]
        self.cols = df.columns
        self.cols = [colname.strip().replace(" ", "_") for colname in self.cols]
        self.df = all_dfs[dfname]
        self.nrows, self.ncols = self.df.shape
        self.prim_key, self.foreign_key_names, self.other_fields = None, [], []
        for i, cname in enumerate(self.cols):
            if cname.endswith("_PRIMARY"):
                if self.prim_key is not None:
                    raise KeyError("Primary key not unique in table {}".format(self.dfname))
                else:
                    self.cols[i] = cname[:-len("_PRIMARY")]
                    #self.df.columns[i] = self.cols[i][:]
                    self.df.rename(columns={self.df.columns[i]: self.cols[i]}, inplace=True)
                    keytype = self.get_coltype(self.cols[i])
                    self.prim_key = field(self.cols[i], keytype)
            elif cname.endswith("_FOREIGN"):
                cname = cname[:-len(self.frstr)]
                self.cols[i] = cname
                #self.df.columns[i] = self.cols[i][:]
                self.df.rename(columns={self.df.columns[i]: self.cols[i]}, inplace=True)
                self.foreign_key_names.append(cname)
                ftype = self.get_coltype(cname)
                self.other_fields.append(field(cname, ftype))
            else:
                self.df.rename(columns={self.df.columns[i]: cname}, inplace=True)
                ftype = self.get_coltype(cname)
                self.other_fields.append(field(cname, ftype))

    def resolve_foreign_keys(self, all_df_dbs):
        self.foreign_keys = []
        for fkey_name in self.foreign_key_names:
            self.resolve_key(fkey_name, all_df_dbs)

    def resolve_key(self, keyname, all_df_dbs):
        keytype = self.get_coltype(keyname)
        found = False
        for dfname, df in all_df_dbs.items():
            if dfname != self.dfname:
                if keyname == df.prim_key.name:
                    self.foreign_keys.append(keyref(keyname, keytype, dfname, self.dfname))
                    found = True
                    break
        if not found:
            raise KeyError("Could not find foreign key {} in table {} in any other table".format(keyname, self.dfname))

    def get_coltype(self, colname):
        coltypes = self.df.dtypes.to_dict()
        if coltypes[colname].kind in "ib":
            return "INTEGER"
        elif coltypes[colname].kind in "f":
            return "REAL"
        else:
            return "TEXT"

    def create_table(self, sql_cursor):
        sqlstr = "CREATE TABLE {} ({})"

        cols_str = ""
        if self.prim_key is not None:
            cols_str += "{} {} PRIMARY KEY".format(self.prim_key.name, self.prim_key.ftype)
        for ofield in self.other_fields:
            if len(cols_str) > 0:
                cols_str += ", "
            cols_str += "{} {}".format(ofield.name, ofield.ftype)
        for fkey in self.foreign_keys:
            if len(cols_str) > 0:
                cols_str += ", "
            cols_str += "FOREIGN KEY ({}) REFERENCES {}({})".format(fkey.name, fkey.reftable_name, fkey.name)
        sqlstr = sqlstr.format(self.dfname, cols_str)

        print("Executing: {}".format(sqlstr))
        sql_cursor.execute(sqlstr)

    def insert_row(self, rowdict, sql_cursor):
        all_fields = self.other_fields
        #all_fields = [self.prim_key] + self.other_fields
        if self.prim_key is not None:
            all_fields = [self.prim_key] + all_fields
        sqlstr = "INSERT INTO {} ({}) VALUES ({})"
        value_list = [None for _ in range(len(rowdict))]
        cols = [f.name for f in all_fields]
        for key, value in rowdict.items():
            try:
                idx = cols.index(key)
            except ValueError as e:
                print(cols)
                raise e
            ftype = all_fields[idx].ftype
            if ftype != "TEXT":
                value_list[idx] = str(value)
            else:
                if not isinstance(value, float):     # TODO: FIX THIS!!!
                    value = value.strip()
                value_list[idx] = "'" + str(value) + "'"
            cols[idx] = all_fields[idx].name
        value_sqlstr = ",".join(value_list)
        cols_str = ",".join(cols)
        sqlstr = sqlstr.format(self.dfname, cols_str, value_sqlstr)
        print("Executing: {}".format(sqlstr))
        sql_cursor.execute(sqlstr)

    def populate_table(self, sql_cursor):
        for i in range(self.nrows):
            rowdict = self.df.iloc[i].to_dict()
            self.insert_row(rowdict, sql_cursor)
        
def create_subbiome_junction(filename, sql_cursor):
    df = pd.read_csv(filename)
    cols = list(df.columns)
    print(cols)

    sqlstr = "CREATE TABLE subBiomesMapping (Sub_biome_ID INTEGER, Tree_ID INTEGER, Canopy INTEGER, FOREIGN KEY (Tree_ID) REFERENCES species(Tree_ID), \
    FOREIGN KEY (Sub_biome_ID) REFERENCES subBiome(Sub_biome_ID))"
    print("Executing: {}".format(sqlstr))
    sql_cursor.execute(sqlstr)

    nrows, ncols = df.shape
    df.rename(columns={"Sub biome ID PRIMARY": "Sub biome ID"}, inplace=True)
    cols = ["Sub biome ID", "canopy list", "co-species list"]
    all_cols = list(df.columns)
    begin_idx = all_cols.index("canopy list")
    types = ["INTEGER", "INTEGER", "INTEGER"]
    for i in range(nrows):
        values = []
        rowdict = df.iloc[i].to_dict()
        biome_id = rowdict["Sub biome ID"]
        canopy = True
        canopy_plants, all_plants = [], []
        for colidx in range(begin_idx, len(all_cols)):
            colname = all_cols[colidx]
            if colname == "co-species list":
                canopy = False
            if np.isnan(rowdict[colname]):
                if not canopy:
                    break
                else:
                    continue
            else:
                treeid = rowdict[colname]
                if canopy:
                    canopy_plants.append(treeid)
                else:
                    all_plants.append(treeid)

        for treeid in all_plants:
            if treeid in canopy_plants:
                canopy = "1"
            else:
                canopy = "0"
            sqlstr = "INSERT INTO subBiomesMapping (Sub_biome_ID, Tree_ID, Canopy) VALUES ({}, {} ,{})".format(biome_id, treeid, canopy)
            print("Executing: {}".format(sqlstr))
            sql_cursor.execute(sqlstr)


def create_table(df, foreign_keys, name, sql_cursor):
    sqlstr = "CREATE TABLE {} ({})"
    cols = list(df.columns)
    cols = [colname.strip().replace(" ", "_") for colname in cols]
    df.columns = cols
    cols_sqlstr = ""
    coltypes = df.dtypes.to_dict()
    is_numeric = [coltypes[col].kind in "bifc" for col in cols]
    is_int = [coltypes[col].kind in "ib" for col in cols]
    is_float = [coltypes[col].kind in "f" for col in cols]
    types = ["INTEGER" if is_int[i] else "REAL" if is_float[i] else "TEXT" if not is_numeric[i] else "NULL" for i in range(len(cols))]
    cols_and_types = []
    for c, t in zip(cols, types):
        cols_and_types.append(c)
        cols_and_types.append(t)
    #is_bool = [coltypes[col].kind in "b" for col in cols]
    if len(cols) > 0:
        cols_sqlstr += "{} {} PRIMARY KEY"
        
        for i in range(1, len(cols)):
            cols_sqlstr += ", {} {}"
    cols_sqlstr = cols_sqlstr.format(*cols_and_types)
    sqlstr = sqlstr.format(name, cols_sqlstr)

    print("Executing: {}".format(sqlstr))
    sql_cursor.execute(sqlstr)

    nrows, ncols = df.shape
    for i in range(nrows):
        sqlstr = "INSERT INTO {} ({}) VALUES ({})"
        rowdict = df.iloc[i].to_dict()
        value_list = [None for _ in range(len(rowdict))]
        for key, value in rowdict.items():
            idx = cols.index(key)
            isnum = is_numeric[idx]
            if isnum:
                value_list[idx] = str(value)
            else:
                value_list[idx] = "'" + str(value) + "'"
        value_sqlstr = ",".join(value_list)
        cols_str = ",".join(cols)
        sqlstr = sqlstr.format(name, cols_str, value_sqlstr)
        print("Executing: {}".format(sqlstr))
        sql_cursor.execute(sqlstr)

    
filenames = {}
filenames["biome_stats"] = "biomeStatsTable.csv"
filenames["allometry"] = "allometryTable.csv"
filenames["growth"] = "growthMonthTable.csv"
filenames["species"] = "speciesTable.csv"
filenames["subBiomes"] = "subBiomeTable.csv"
filenames["monthlies"] = "monthlies.csv"
filenames["shadeTolLow"] = "shadeTolLow.csv"
filenames["sunTolUpper"] = "sunTolUpper.csv"
filenames["coldTolLow"] = "coldTolLow.csv"
filenames["droughtTolLow"] = "droughtTolLow.csv"
filenames["floodTolUpper"] = "floodTolUpper.csv"
filenames["slopeTolUpper"] = "slopeTolUpper.csv"
filenames["modelDetails"] = "modelDetails.csv"
filenames["modelMapping"] = "modelMapping.csv"
            
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("directory", type=str, help="Directory containing info files for db")

    a = arg_parser.parse_args()

    while a.directory.endswith("/"):
        a.directory = a.directory[:-1]

    dbfile = a.directory + "/" + "sonoma.db"
    if os.path.isfile(dbfile):
        os.remove(dbfile)
    conn = sqlite3.connect(dbfile)
    cursor = conn.cursor()

    create_specratios(a.directory)      # creates the modelMapping csv file from modelDetails.csv

    for key, value in filenames.items():
        filenames[key] = a.directory + "/" + value

    data_tables = {}
    for key, value in filenames.items():
        data_tables[key] = pd.read_csv(value)

    df_dbs = {}
    for key in data_tables:
        df_dbs[key] = db_dataframe(key, data_tables)

    for key, value in df_dbs.items():
        value.resolve_foreign_keys(df_dbs)
        for k in value.foreign_keys:
            print(k)

    for key, value in df_dbs.items():
        value.create_table(cursor)
        value.populate_table(cursor)

    create_subbiome_junction(a.directory + "/" + "subBiomeMapping.csv", cursor)

    """
    def mktable(name):
        return create_table(data_tables[name], name, cursor)

    print(data_tables["biome_stats"])
    print(list(data_tables["biome_stats"].columns))

    mktable("biome_stats")
    mktable("allometry")
    mktable("growth")
    mktable("species")

    """
    conn.commit()
    conn.close()
