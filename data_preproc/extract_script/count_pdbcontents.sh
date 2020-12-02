count=0

if [ "$1" == "" ] || [ "$2" == "" ]; then
	echo "Usage: count_pdbcontents.sh <base dirname> <ndirs>"
	exit 1
fi

base_dirname=$1
ndirs=$2

dbname="/home/konrad/EcoSynth/data_preproc/common_data/sonoma.db"

for count in $(eval echo "{0..$(($ndirs - 1))}"); do
	dirname="$base_dirname$count"
	datasetname="$(basename $dirname)"
	for run_id in {0..3}; do
		#/home/konrad/EcoSynth/viewer/build-UnderSim-Qt_5_10_0_GCC_64bit-Default/viewer/viewer "$dirname" $run_id
		filename=$dirname/"$datasetname"_undergrowth$run_id.pdb
		#echo "$filename"
		python3 /home/konrad/PhDStuff/prototypes/repo/code/py_scripts/pdb_counts.py $filename
		#echo "$(dirname $dirname)"
		#echo "$(basename $dirname)"
	done
done
