#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>


#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

using namespace dealii;



template <int dim>
class Adaptive_kelly
{
public:
  Adaptive_kelly(const unsigned int fe_degree, const unsigned int refinement_cycles = 8);

  void run();
  void observe_convergence();

private:
  void setup_system();
  void assemble_system();
  void solve();
  void refine_grid();
  void output_results(const unsigned int cycle) const;

  Triangulation<dim> triangulation;

  FE_Q<dim>       fe;
  DoFHandler<dim> dof_handler;


  AffineConstraints<double> constraints;

  SparseMatrix<double> system_matrix;
  SparsityPattern      sparsity_pattern;

  Vector<double> solution;
  Vector<double> system_rhs;
  const unsigned int n_refinement_cycles;

  std::vector<double> mean_values{};//{n_refinement_cycles}; //store mean values after refinements

};



template <int dim>
double coefficient(const Point<dim> &p)
{
  if (p.square() < 0.5 * 0.5)
    return 20;
  else
    return 1;
}





template <int dim>
Adaptive_kelly<dim>::Adaptive_kelly(const unsigned int fe_degree, const unsigned refinement_cycles)
  : fe{fe_degree}, dof_handler{triangulation}, n_refinement_cycles{refinement_cycles}
{}




template <int dim>
void Adaptive_kelly<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);


  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);

  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);

  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
}



template <int dim>
void Adaptive_kelly<dim>::assemble_system()
{
  const QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values.reinit(cell);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          const double current_coefficient =
            coefficient<dim>(fe_values.quadrature_point(q_index)); //a depends on the quadrature point
          for (const unsigned int i : fe_values.dof_indices())
            {
              for (const unsigned int j : fe_values.dof_indices())
                cell_matrix(i, j) +=
                  (current_coefficient *              // a(x_q)
                   fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                   fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                   fe_values.JxW(q_index));           // dx

              cell_rhs(i) += (1.0 *                               // f(x)
                              fe_values.shape_value(i, q_index) * // phi_i(x_q)
                              fe_values.JxW(q_index));            // dx
            }
        }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}






template <int dim>
void Adaptive_kelly<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);
//  std::cout << std::setprecision(8) << "Mean value: " << VectorTools::compute_mean_value (dof_handler,
//                                            QGauss<dim>(fe.degree + 1),
//                                            solution,
//                                            0) << std::endl;

  mean_values.push_back(VectorTools::compute_mean_value (dof_handler,
                                              QGauss<dim>(fe.degree + 1),
                                              solution,
                                              0));
}



template <int dim>
void Adaptive_kelly<dim>::refine_grid()
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells()); //for every cell, compute an estimate of the error (floats are okay, no need the estimator to be super accurate)

  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim - 1>(fe.degree + 1),
                                     {},
                                     solution,
                                     estimated_error_per_cell); //output is obtained by reference. It's a vector of values for all active cells

  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  0.5,
                                                  0.03); //flags some cells of the triangulation for refinement or coarsening

  triangulation.execute_coarsening_and_refinement(); //exectue the local coarsnening/refinement
}



template <int dim>
void Adaptive_kelly<dim>::output_results(const unsigned int cycle) const
{
  {
    GridOut               grid_out;
    std::ofstream         output("grid-" + std::to_string(cycle) + ".gnuplot");
    GridOutFlags::Gnuplot gnuplot_flags(false, 5);
    grid_out.set_flags(gnuplot_flags);
    MappingQGeneric<dim> mapping(3);
    grid_out.write_gnuplot(triangulation, output, &mapping);
  }

  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();

    std::ofstream output("solution-" + std::to_string(cycle) + ".vtu");
    data_out.write_vtu(output);
  }
}



template <int dim>
void Adaptive_kelly<dim>::run()
{
  for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
        {
          //GridGenerator::hyper_ball(triangulation);
    	  GridGenerator::hyper_L(triangulation, -1., +1.);
          triangulation.refine_global(2); //start from coarse grid
        }
      else{
       refine_grid();
      }

      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells() << std::endl;

      setup_system();

      std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                << std::endl;

      assemble_system();
      solve();
      output_results(cycle);
    }
}


template<int dim>
void Adaptive_kelly<dim>::observe_convergence(){
	for(unsigned int cycle{0}; cycle< n_refinement_cycles;++cycle)
		std::cout << "Mean value at " << cycle <<" cycle: \t"<< mean_values[cycle] << "\n";

}



int main()
{
  try
    {
      Adaptive_kelly<2> lapl(2);
      lapl.run();

      lapl.observe_convergence();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
