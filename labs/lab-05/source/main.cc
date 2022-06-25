#include "poisson.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  std::string par_name = "";
  if (argc > 1)
    par_name = argv[1];

  Poisson<2> laplace_problem;
  if (par_name != "")
    laplace_problem.initialize(par_name);
  laplace_problem.run();
  return 0;
}