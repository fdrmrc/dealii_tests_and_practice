/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 *          Luca Heltai, 2021
 */
#include "poisson.h"

using namespace dealii;

template <int dim>
Poisson<dim>::Poisson()
  : mpi_communicator{MPI_COMM_WORLD}
  , n_mpi_processes{Utilities::MPI::n_mpi_processes(mpi_communicator)}
  , this_mpi_process{Utilities::MPI::this_mpi_process(mpi_communicator)}
  , pcout{std::cout, (this_mpi_process == 0)}
  , dof_handler(triangulation)
{
  add_parameter("Finite element degree", fe_degree);
  add_parameter("Number of global refinements", n_refinements);
  add_parameter("Output filename", output_filename);
  add_parameter("Forcing term expression", forcing_term_expression);
  add_parameter("Dirichlet boundary condition expression",
                dirichlet_boundary_conditions_expression);
  add_parameter("Neumann boundary condition expression",
                neumann_boundary_conditions_expression);

  add_parameter("Dirichlet boundary ids", dirichlet_ids);
  add_parameter("Neumann boundary ids", neumann_ids);

  add_parameter("Problem constants", constants);
  add_parameter("Grid generator function", grid_generator_function);
  add_parameter("Grid generator arguments", grid_generator_arguments);
  add_parameter("Number of refinement cycles", n_refinement_cycles);

  this->prm.enter_subsection("Error table");
  error_table.add_parameters(this->prm);
  this->prm.leave_subsection();
}


template <int dim>
void
Poisson<dim>::initialize(const std::string &filename)
{
  ParameterAcceptor::initialize(filename);
}



template <int dim>
void
Poisson<dim>::parse_string(const std::string &parameters)
{
  ParameterAcceptor::prm.parse_input_from_string(parameters);
  ParameterAcceptor::parse_all_parameters();
}



template <int dim>
void
Poisson<dim>::make_grid()
{
  GridGenerator::generate_from_name_and_arguments(triangulation,
                                                  grid_generator_function,
                                                  grid_generator_arguments);
  triangulation.refine_global(n_refinements);
  pcout << "Number of active cells: " << triangulation.n_active_cells()
        << std::endl;
}



template <int dim>
void
Poisson<dim>::refine_grid()
{
  triangulation.refine_global(1);
}



template <int dim>
void
Poisson<dim>::setup_system()
{
  if (!fe)
    {
      fe = std::make_unique<FE_Q<dim>>(fe_degree);
      forcing_term.initialize(dim == 1 ? "x" :
                              dim == 2 ? "x,y" :
                                         "x,y,z",
                              forcing_term_expression,
                              constants);
      dirichlet_boundary_condition.initialize(
        dim == 1 ? "x" :
        dim == 2 ? "x,y" :
                   "x,y,z",
        dirichlet_boundary_conditions_expression,
        constants);

      neumann_boundary_condition.initialize(
        dim == 1 ? "x" :
        dim == 2 ? "x,y" :
                   "x,y,z",
        neumann_boundary_conditions_expression,
        constants);
    }
  GridTools::partition_triangulation(n_mpi_processes, triangulation);

  dof_handler.distribute_dofs(*fe);

  DoFRenumbering::subdomain_wise(dof_handler);
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  for (const auto &id : dirichlet_ids)
    VectorTools::interpolate_boundary_values(dof_handler,
                                             id,
                                             dirichlet_boundary_condition,
                                             constraints);
  constraints.close();


  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

  const std::vector<IndexSet> locally_owned_dofs_per_process =
    DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
  const IndexSet locally_owned_dofs =
    locally_owned_dofs_per_process[this_mpi_process];


  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       mpi_communicator);
  solution.reinit(locally_owned_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
}



template <int dim>
void
Poisson<dim>::assemble_system()
{
  QGauss<dim>     quadrature_formula(fe->degree + 1);
  QGauss<dim - 1> face_quadrature_formula(fe->degree + 1);

  FEValues<dim> fe_values(*fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(*fe,
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                     update_JxW_values);

  const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->subdomain_id() == this_mpi_process)
        {
          fe_values.reinit(cell);
          cell_matrix = 0;
          cell_rhs    = 0;
          for (const unsigned int q_index :
               fe_values.quadrature_point_indices())
            {
              for (const unsigned int i : fe_values.dof_indices())
                for (const unsigned int j : fe_values.dof_indices())
                  cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                     fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                     fe_values.JxW(q_index));           // dx
              for (const unsigned int i : fe_values.dof_indices())
                cell_rhs(i) +=
                  (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                   forcing_term.value(
                     fe_values.quadrature_point(q_index)) * // f(x_q)
                   fe_values.JxW(q_index));                 // dx
            }

          if (cell->at_boundary())
            //  for(const auto face: cell->face_indices())
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              if (neumann_ids.find(cell->face(f)->boundary_id()) !=
                  neumann_ids.end())
                {
                  fe_face_values.reinit(cell, f);
                  for (const unsigned int q_index :
                       fe_face_values.quadrature_point_indices())
                    for (const unsigned int i : fe_face_values.dof_indices())
                      cell_rhs(i) +=
                        fe_face_values.shape_value(i, q_index) *
                        neumann_boundary_condition.value(
                          fe_face_values.quadrature_point(q_index)) *
                        fe_face_values.JxW(q_index);
                }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }
    }
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}



template <int dim>
unsigned int
Poisson<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  PETScWrappers::SolverCG  solver(solver_control, mpi_communicator);
  PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  Vector<double> localized_solution(solution);
  constraints.distribute(localized_solution);
  solution = localized_solution;
  return solver_control.last_step();
}



template <int dim>
void
Poisson<dim>::output_results(const unsigned cycle) const
{
  const Vector<double> localized_solution(solution);

  if (this_mpi_process == 0)
    {
      std::ofstream fname{"solution" + std::to_string(cycle) + ".vtk"};
      DataOut<dim>  data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(localized_solution, "u");

      std::vector<unsigned int> partition_int(triangulation.n_active_cells());
      GridTools::get_subdomain_association(triangulation, partition_int);

      const Vector<double> partitioning(partition_int.begin(),
                                        partition_int.end());

      data_out.add_data_vector(partitioning, "partitioning");

      data_out.build_patches();
      data_out.write_vtk(fname);
    }
}



template <int dim>
void
Poisson<dim>::run()
{
  make_grid();
  for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle)
    {
      pcout << "Cycle: " << cycle << std::endl;
      setup_system();

      pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << " (by partition:";
      for (unsigned int p = 0; p < n_mpi_processes; ++p)
        pcout << (p == 0 ? ' ' : '+')
              << (DoFTools::count_dofs_with_subdomain_association(dof_handler,
                                                                  p));
      pcout << ')' << std::endl;

      assemble_system();
      const unsigned int n_iter = solve();
      pcout << "Solver converged in " << n_iter << " iterations" << std::endl;

      output_results(cycle);
      if (cycle < n_refinement_cycles - 1)
        refine_grid();
    }
}

template class Poisson<1>;
template class Poisson<2>;
template class Poisson<3>;