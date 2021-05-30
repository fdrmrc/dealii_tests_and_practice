# dealii_practice
Test cases repo for the dealii C++ finite element library. Steps are exported as C++ Eclipse project as follows:

- `cmake -G "Eclipse CDT4 - Unix Makefiles" .` (don't miss the . at the end)

Now move to Eclipse IDE:

- New MakeFile project with existing code
- Select LinuxGCC or MacOSX GCC compiler
- Browse directory of current project

This allows you to compile. To run the resulting executable:
- Run configurations -> New launch configuration: write the name of the executable
