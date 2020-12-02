count=0

if [ "$1" == "" ] || [ "$2" == "" ]; then
	echo "Usage: sunsim_canopy_mass.sh <base dirname> <ndirs>"
	exit 1
fi

base_dirname=$1
ndirs=$2

for count in $(eval echo "{0..$(($ndirs - 1))}"); do
	dirname="$base_dirname$count"
	/home/konrad/EcoLearn/Code/build-UnderSim-Desktop-Default/viewer/viewer "$dirname" 0
done
