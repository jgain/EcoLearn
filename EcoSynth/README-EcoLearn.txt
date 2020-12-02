POSSIBLE SMALLER SUBSET OF INSTALLS:

Boost, at least 1.49 (earlier versions don't play nice with C++11):
sudo apt-get update
sudo apt-get install libboost-all-dev


UBUNTU INSTALLATION:
--------------------
These are requirements for Ubuntu. In each case, I've listed where to get the software 
for Ubuntu 18.04. For newer versions of Ubuntu you might not need all the PPAs.

CMake 2.8.7+, Eigen 3.x, make, automake 1.9, pkg-config, doxygen, GLUT, OpenEXR, GDAL:

sudo apt-get install cmake libeigen3-dev make automake1.9 pkg-config doxygen freeglut3-dev libopenexr-dev openexr openexr-viewers exrtools libgdal1-dev libglm-dev

GCC 4.8:

sudo apt-add-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install g++-4.8

Boost, at least 1.49 (earlier versions don't play nice with C++11):
sudo apt-get update
sudo apt-get install libboost-all-dev

OpenCL: (for NVIDIA cards)

sudo apt-get install nvidia-opencl-dev 

Qt5:

sudo apt-add-repository ppa:ubuntu-sdk-team/ppa
sudo apt-get update
sudo apt-get install qtbase5-dev


ADDED PACKAGES:

sudo apt-get install libgl1-mesa-dev qtcreator libglew-dev libqtcore4 libqtgui4 libqt4-opengl libqt4-opengl-dev //line used in 3rd year to fix opengl stuff 
sudo apt-get install build-essential
sudo apt-get install libxmu-dev libxi-dev
sudo apt-get install libcppunit-dev //added in to get testing framework

ImageMagick:

I couldn't find a PPA for this, and 6.8 is required. I downloaded the source,
unpacked it, and run ./configure, make, make install. Note, that this needs to be ImageMagick version 6 and not a later version.

ADDITIONAL REQUIREMENTS:

Modify the `basedir` variable in viewer/CMakeLists.txt to contain the location of this directory on your machine.

packages/libraries:
+ SDL2		(install: sudo apt install libsdl2-dev)
+ Assimp	(install: sudo apt install libassimp-dev)
+ CUDA 9+	(install: this can be complicated sometimes. Apparently you can install from the ubuntu repositories, but I think the recommended way is to download the installer from the NVIDIA website)
+ OpenCV	(install: sudo apt install libopencv-dev)
+ pthread, X11, libpng, libz (although I think the first two are usually installed by default)

The last requirement in the list (all 4 of them on the last line) is only because I used CImg to import the textures for the tree models. In the near future I will use the native Qt QImage class to import the textures instead, so this is a temporary requirement. The OpenCV requirement might also not be strictly necessary (but it is needed to compile), but might be used for some smoothing when doing canopy placement.


Client-side git config
----------------------
In the top level of the repository, run the following commands:

git config core.whitespace tab-in-indent
cp .git/hooks/pre-commit.sample .git/hooks/pre-commit

This will prevent checkins with tab indentation or with trailing whitespace.


Compiling
---------

There is a build script that you can run from the root: buildecolearn.sh

Alternatively, once all the requirements are running, create a subdirectory called build (or
anything starting with build - you can have multiple build directories), switch
into it, and run

cmake <options> ..

Some useful options
-DCMAKE_CXX_COMPILER=g++-4.8          (to force a specific compiler)
-DCMAKE_BUILD_TYPE=Debug|Release|RelWithDebInfo  (to select build type, only choose one option here!!!)
-DCMAKE_CXX_FLAGS_DEBUG="-gdwarf-3"   (helps older versions of GDB interpret debug symbols)
-DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -DNDEBUG -g -gdwarf-3" (similar)
-DCMAKE_CXX_FLAGS="-fPIC" (if the compilation complains about position independent code)


Then run make to build. cmake options are sticky. The executable is ./viewer/viewer. System must be run from the build directory because there are some relative paths.

OSX INSTALLATION:
-----------------
1. Make sure that Xcode command line tools  and homebrew are installed.
2. Install QT and QTCreator
	Add CMAKE_PREFIX_PATH:~/Qt/5.11.0/clang_64 (or similar) to bash_profile
3. Install latest GCC via homebrew:
		from command line: brew install gcc
		point qtcreator to g++-8 in /usr/local/bin (preferences->build&run->manual add)
		Edit buildterviewer.sh to point to g++-8 if necessary
4. brew install: cppunit cmake doxygen pkg-config openexr eigen glew gdal libsvm glm
5. Brew install -cc=gcc-8 boost

6. 
	Download imagemagick version 6
	tar -xvf ImageMagick*
	cd ./ImageMagick*
	./configure -CXX g++-8
	make
	sudo make install
	sudo update_dyld_shared_cache

7. Revise paths in CMakeLists.txt in the places indicated.

8. in GLHeaders.h

	NOTE: the gl3.h file is missing and you need to copy it over manually. If you installed all packages the file should be 	somewhere on the machine. Navigate to root and search for gl3.h and copy it to this directory in the project: cgp1-	prep/khronos_headers/GL

	change:

	#include <GL/gl3.h> 
	#include <GL/glu.h> 

	To: 
	#include <OpenGL/gl3.h> 
	#include <OpenGL/glu.h> 
9. Compile and run as per Ubuntu instructions


GUI
---
Once a model has been loaded, it can be rotated by holding down the right mouse button and moving the cursor. Zooming is with the mouse wheel. Double click with the right mouse button to change the focal point on the terrain. There are also various keyboard shortcuts which can be found in glwidgets.cpp

Valgrind Usage
--------------

valgrind --leak-check=yes --track-origins=yes ./viewer/viewer



