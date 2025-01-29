# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build"

# Include any dependencies generated for this target.
include CMakeFiles/NeuralNetwork.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/NeuralNetwork.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/NeuralNetwork.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/NeuralNetwork.dir/flags.make

CMakeFiles/NeuralNetwork.dir/codegen:
.PHONY : CMakeFiles/NeuralNetwork.dir/codegen

CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.o: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.o: /Users/nicodevelopment/Desktop/C++\ Practice/FromScratch/NeuralNetwork/src/ActivationFunctions.cpp
CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.o: CMakeFiles/NeuralNetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.o -MF CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.o.d -o CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.o -c "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/ActivationFunctions.cpp"

CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/ActivationFunctions.cpp" > CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.i

CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/ActivationFunctions.cpp" -o CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.s

CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.o: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.o: /Users/nicodevelopment/Desktop/C++\ Practice/FromScratch/NeuralNetwork/src/DenseLayer.cpp
CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.o: CMakeFiles/NeuralNetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.o -MF CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.o.d -o CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.o -c "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/DenseLayer.cpp"

CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/DenseLayer.cpp" > CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.i

CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/DenseLayer.cpp" -o CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.s

CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.o: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.o: /Users/nicodevelopment/Desktop/C++\ Practice/FromScratch/NeuralNetwork/src/Layer.cpp
CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.o: CMakeFiles/NeuralNetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.o -MF CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.o.d -o CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.o -c "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/Layer.cpp"

CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/Layer.cpp" > CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.i

CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/Layer.cpp" -o CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.s

CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.o: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.o: /Users/nicodevelopment/Desktop/C++\ Practice/FromScratch/NeuralNetwork/src/LossFunction.cpp
CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.o: CMakeFiles/NeuralNetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.o -MF CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.o.d -o CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.o -c "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/LossFunction.cpp"

CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/LossFunction.cpp" > CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.i

CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/LossFunction.cpp" -o CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.s

CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.o: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.o: /Users/nicodevelopment/Desktop/C++\ Practice/FromScratch/NeuralNetwork/src/MNISTLoader.cpp
CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.o: CMakeFiles/NeuralNetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.o -MF CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.o.d -o CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.o -c "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/MNISTLoader.cpp"

CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/MNISTLoader.cpp" > CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.i

CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/MNISTLoader.cpp" -o CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.s

CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.o: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.o: /Users/nicodevelopment/Desktop/C++\ Practice/FromScratch/NeuralNetwork/src/Matrix.cpp
CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.o: CMakeFiles/NeuralNetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.o -MF CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.o.d -o CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.o -c "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/Matrix.cpp"

CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/Matrix.cpp" > CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.i

CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/Matrix.cpp" -o CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.s

CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.o: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.o: /Users/nicodevelopment/Desktop/C++\ Practice/FromScratch/NeuralNetwork/src/NeuralNetwork.cpp
CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.o: CMakeFiles/NeuralNetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.o -MF CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.o.d -o CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.o -c "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/NeuralNetwork.cpp"

CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/NeuralNetwork.cpp" > CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.i

CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/NeuralNetwork.cpp" -o CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.s

CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.o: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.o: /Users/nicodevelopment/Desktop/C++\ Practice/FromScratch/NeuralNetwork/src/Optimizer.cpp
CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.o: CMakeFiles/NeuralNetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.o -MF CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.o.d -o CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.o -c "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/Optimizer.cpp"

CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/Optimizer.cpp" > CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.i

CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/Optimizer.cpp" -o CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.s

CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.o: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.o: /Users/nicodevelopment/Desktop/C++\ Practice/FromScratch/NeuralNetwork/src/Trainer.cpp
CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.o: CMakeFiles/NeuralNetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.o -MF CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.o.d -o CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.o -c "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/Trainer.cpp"

CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/Trainer.cpp" > CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.i

CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/Trainer.cpp" -o CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.s

CMakeFiles/NeuralNetwork.dir/src/main.cpp.o: CMakeFiles/NeuralNetwork.dir/flags.make
CMakeFiles/NeuralNetwork.dir/src/main.cpp.o: /Users/nicodevelopment/Desktop/C++\ Practice/FromScratch/NeuralNetwork/src/main.cpp
CMakeFiles/NeuralNetwork.dir/src/main.cpp.o: CMakeFiles/NeuralNetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/NeuralNetwork.dir/src/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/NeuralNetwork.dir/src/main.cpp.o -MF CMakeFiles/NeuralNetwork.dir/src/main.cpp.o.d -o CMakeFiles/NeuralNetwork.dir/src/main.cpp.o -c "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/main.cpp"

CMakeFiles/NeuralNetwork.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/NeuralNetwork.dir/src/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/main.cpp" > CMakeFiles/NeuralNetwork.dir/src/main.cpp.i

CMakeFiles/NeuralNetwork.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/NeuralNetwork.dir/src/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/src/main.cpp" -o CMakeFiles/NeuralNetwork.dir/src/main.cpp.s

# Object files for target NeuralNetwork
NeuralNetwork_OBJECTS = \
"CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.o" \
"CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.o" \
"CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.o" \
"CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.o" \
"CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.o" \
"CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.o" \
"CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.o" \
"CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.o" \
"CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.o" \
"CMakeFiles/NeuralNetwork.dir/src/main.cpp.o"

# External object files for target NeuralNetwork
NeuralNetwork_EXTERNAL_OBJECTS =

NeuralNetwork: CMakeFiles/NeuralNetwork.dir/src/ActivationFunctions.cpp.o
NeuralNetwork: CMakeFiles/NeuralNetwork.dir/src/DenseLayer.cpp.o
NeuralNetwork: CMakeFiles/NeuralNetwork.dir/src/Layer.cpp.o
NeuralNetwork: CMakeFiles/NeuralNetwork.dir/src/LossFunction.cpp.o
NeuralNetwork: CMakeFiles/NeuralNetwork.dir/src/MNISTLoader.cpp.o
NeuralNetwork: CMakeFiles/NeuralNetwork.dir/src/Matrix.cpp.o
NeuralNetwork: CMakeFiles/NeuralNetwork.dir/src/NeuralNetwork.cpp.o
NeuralNetwork: CMakeFiles/NeuralNetwork.dir/src/Optimizer.cpp.o
NeuralNetwork: CMakeFiles/NeuralNetwork.dir/src/Trainer.cpp.o
NeuralNetwork: CMakeFiles/NeuralNetwork.dir/src/main.cpp.o
NeuralNetwork: CMakeFiles/NeuralNetwork.dir/build.make
NeuralNetwork: CMakeFiles/NeuralNetwork.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX executable NeuralNetwork"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/NeuralNetwork.dir/link.txt --verbose=$(VERBOSE)
	/opt/homebrew/bin/cmake -E copy_directory /Users/nicodevelopment/Desktop/C++\ Practice/FromScratch/NeuralNetwork/data /Users/nicodevelopment/Desktop/C++\ Practice/FromScratch/NeuralNetwork/build/data

# Rule to build all files generated by this target.
CMakeFiles/NeuralNetwork.dir/build: NeuralNetwork
.PHONY : CMakeFiles/NeuralNetwork.dir/build

CMakeFiles/NeuralNetwork.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/NeuralNetwork.dir/cmake_clean.cmake
.PHONY : CMakeFiles/NeuralNetwork.dir/clean

CMakeFiles/NeuralNetwork.dir/depend:
	cd "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork" "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork" "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build" "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build" "/Users/nicodevelopment/Desktop/C++ Practice/FromScratch/NeuralNetwork/build/CMakeFiles/NeuralNetwork.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : CMakeFiles/NeuralNetwork.dir/depend

