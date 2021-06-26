## dealii_practice
Test cases repo for the dealii C++ finite element library. Problems can be exported as C++ Eclipse project as follows:

- Create a CMakeLists.txt with 'target' the name of your problem.cc file, in case you have only one source file. 
- If you have more files with include/source folders and/or PETSC/Trilinos and other dependencies, then a good customized `CMakeLists.txt` file can be the one named `my_CMakeLists.txt` (of course you have to remove the `my` prefix if you intend to use it).

From command line, type
- `cmake .` to detect C/CXX compilers
- `cmake -G "Eclipse CDT4 - Unix Makefiles" .` (don't miss the `.` at the end)

Now move to Eclipse IDE:

- File -> Import -> Existing Project into Workspace
- Select the directory of current project


You can compile and then run using Eclipse targets or as usual from command line using make commands (see `make info` to know how to switch from debug/release mode)

# Parameter handler 

After that a `test_parameter_file.prm` has been generated, to run the executable with your new parameters, without recompiling all the project, type `./main test_parameter_file.prm`.
To run on multiple processors, just use `mpirun` as usual `mpirun -np 4 ./main test_parameter_file.prm` (here I run on 4 processors)


![Screenshot](adaptive_refinement_kelly_estimator/3D_fichera_corner_adaptive.png)

