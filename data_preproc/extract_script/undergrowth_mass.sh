count=0

if [ "$1" == "" ] || [ "$2" == "" ]; then
	echo "Usage: undergrowth_mass.sh <base dirname> <ndirs>"
	exit 1
fi

base_dirname=$1
ndirs=$2

dbname="/home/konrad/EcoSynth/data_preproc/common_data/sonoma.db"

for count in $(eval echo "{0..$(($ndirs - 1))}"); do
	dirname="$base_dirname$count"
	for run_id in {0..3}; do
		/home/konrad/EcoSynth/viewer/build-UnderSim-Desktop-Default/viewer/viewer "$dirname" $run_id
		#/home/konrad/EcoSynth/viewer/build-UnderSim-Desktop-Default/viewer/viewer "$dirname" 0
	done
done
