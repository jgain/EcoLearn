#!/bin/bash

python3 ~/PhDStuff/prototypes/repo/code/py_scripts/create_sql_db.py ~/EcoSynth/data_preproc/common_data

cp ~/EcoSynth/data_preproc/common_data/sonoma.db ~/EcoSynth/ecodata/sonoma.db
