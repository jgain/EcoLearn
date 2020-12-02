UBUNTU INSTALLATION:
--------------------
These are requirements for Ubuntu. In each case, I've listed where to get the software 
for Ubuntu 18.10. For newer versions of Ubuntu you might not need all the PPAs.

CMake 2.8.7+, Eigen 3.x, make, automake 1.9, pkg-config, doxygen, GLUT, OpenEXR, GDAL, QT5, Boost, at least 1.49 (earlier versions don't play nice with C++11)::

sudo apt-get install cmake libeigen3-dev make automake1.9 pkg-config doxygen freeglut3-dev libopenexr-dev openexr openexr-viewers exrtools libgdal1-dev libglm-dev

sudo apt-get install cmake libeigen3-dev automake doxygen freeglut3-dev libopenexr-dev openexr openexr-viewers libgdal-dev libglm-dev qtbase5-dev libboost-all-dev libglew-dev libcppunit-dev libomp-dev qtcreator sqlite3 sqlitebrowser libzmq3-dev

To install a particular version of g++ (in this case v7, note that version 9 does not work):

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt install gcc-8 g++-8
then make sure to set the correct version in the build script (buildsim.sh)

OpenCL: (for NVIDIA cards)

sudo apt-get install nvidia-opencl-dev 


ADDED PACKAGES:

sudo apt-get install libgl1-mesa-dev libqtcore4 libqtgui4 libqt4-opengl libqt4-opengl-dev //line used in 3rd year to fix opengl stuff 

sudo apt-get install libxmu-dev libxi-dev

ImageMagick:

I couldn't find a PPA for this, and 6.8 is required. I downloaded the source,
unpacked it, and ran ./configure, make, make install. Note, that this needs to be ImageMagick version 6 and not a later version.

ADDITIONAL REQUIREMENTS:

Modify the `basedir` variable in viewer/CMakeLists.txt to contain the location of this directory on your machine.

packages/libraries:
+ SDL2		(install: sudo apt install libsdl2-dev)
+ Assimp	(install: sudo apt install libassimp-dev)
+ CUDA 9+	(install: this can be complicated sometimes. Apparently you can install from the ubuntu repositories, but I think the recommended way is to download the installer from the NVIDIA website)
+ OpenCV	(install: sudo apt install libopencv-dev)
+ pthread, X11, libpng, libz (although I think the first two are usually installed by default)

wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run

sudo sh cuda_10.2.89_440.33.01_linux.run

The last requirement in the list (all 4 of them on the last line) is only because I used CImg to import the textures for the tree models. In the near future I will use the native Qt QImage class to import the textures instead, so this is a temporary requirement. The OpenCV requirement might also not be strictly necessary (but it is needed to compile), but might be used for some smoothing when doing canopy placement.

Compiling and Executing
-----------------------

There is a build script that you can run from the root: buildsim.sh

Alternatively, once all the requirements are running, create a subdirectory called build (or
anything starting with build - you can have multiple build directories), switch
into it, and run

cmake <options> ..

Some useful options
-DCMAKE_CXX_COMPILER=g++-4.8          (to force a specific compiler)
-DCMAKE_BUILD_TYPE=Debug|Release|RelWithDebInfo  (to select build type, only choose one option here)
-DCMAKE_CXX_FLAGS_DEBUG="-gdwarf-3"   (helps older versions of GDB interpret debug symbols)
-DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -DNDEBUG -g -gdwarf-3" (similar)
-DCMAKE_CXX_FLAGS="-fPIC" (if the compilation complains about position independent code)

Then run make to build. cmake options are sticky. The executable is ./sim/sim. System must be run from the build directory because there are some relative paths.

There are three possible usages of the undergrowth simulator depending on the command line parameters:
(1) Run the full simulator for a set number of years: ./sim/sim -sim <data directory> <run id> <number of years> (<data directory> will need to contain all the relevant simulation inputs such as terrain, abiotic maps, canopy plant positions, etc. Samples for this are provided in /datasamples. <run id> is used to differentiate between different simulation runs. <number of years> is the duration of the simulation. The undergrowth output is written in pdb file format to <data directory> as <data director>_undergrowth<run id>.pdb) 
(2) View the results of a previous simulation: ./sim/sim -view <data directory> (as before <data directory> should contain all simulation inputs and outputs. ./sim/sim must previously have been run in -sim mode for all of these to be available.
(3) Prepare abiotic maps but do not run the simulation: ./sim/sim -sun <data directory> [--ignore-canopy] (prepare abiotic maps for sunlight and moisture. Note that this does not overwrite any existing maps. Calling ./sim/sim -sim will prepare these maps as well. The difference here is that simulation is not performed. The optional --ignore-canopy flag prepares a monthly sunlight map without inscribing shade from the canopy layer.)

Data samples are available in the directory /data/simsamples.

GUI
---
Once a model has been loaded, it can be rotated by holding down the right mouse button and moving the cursor. Zooming is with the mouse wheel. Double click with the right mouse button to change the focal point on the terrain. The are two view panels: the first controls the terrain viewing (including rendering of the various abiotic maps) and the second controls the display of plants, such as selecting which species are currently displayed.

There are also various keyboard shortcuts which can be found in glwidgets.cpp



