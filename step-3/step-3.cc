#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <iomanip>      // std::setprecision

using namespace dealii;



class Step3
{
public:
  Step3();

  void run();
  void observe_convergence();

private:
  void make_grid(const unsigned int n_refs);
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<2> triangulation;
  FE_Q<2>          fe; //Langrangian FE space, in 2D
  DoFHandler<2>    dof_handler; //global numbering of degrees of freedom

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};


Step3::Step3()
  : fe(1)
  , dof_handler(triangulation)
{}



void Step3::make_grid(const unsigned int n_refs= 5)
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
 // triangulation.begin_active()->face(0)->set_boundary_id(1); //first unique cell has one boundary with indicator 1, which will be propagated after refinements
  triangulation.refine_global(n_refs);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;

  // If you wanna know how many cells there are, namely the parent cell, its parent, etc., you have
  std::cout << "Number of cells (including parent cells): " << triangulation.n_cells()<<"\n";
}




void Step3::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp); //SparsityPattern object does not hold the values of the matrix, it only stores the places where entries are.

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}



void Step3::assemble_system()
{
  QGauss<2> quadrature_formula(fe.degree + 1); //two quadrature points in each direction, i.e. a total of four points since we are in 2D
  FEValues<2> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values);
  //provide you with information about values and gradients of shape functions at quadrature points on a real cell.

  const unsigned int dofs_per_cell = fe.dofs_per_cell; //ask the finite element to tell us about the number of degrees of freedom per cell

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell); //temporary array with the global numbers of DoFs

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs    = 0; //reset local cell's contributions

      for (const unsigned int q_index : fe_values.quadrature_point_indices()) // @suppress("Symbol is not resolved")
        {
          for (const unsigned int i : fe_values.dof_indices()) // @suppress("Symbol is not resolved")
            for (const unsigned int j : fe_values.dof_indices()) // @suppress("Symbol is not resolved")
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

          for (const unsigned int i : fe_values.dof_indices()) // @suppress("Symbol is not resolved")
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            1 *                                 // f(x_q)
                            fe_values.JxW(q_index));            // dx
        }
      //find out which GLOBAL numbers the degrees of freedom on this cell have
      cell->get_dof_indices(local_dof_indices);

      for (const unsigned int i : fe_values.dof_indices()) //@suppress("Symbol is not resolved")
        for (const unsigned int j : fe_values.dof_indices()) //@suppress("Symbol is not resolved")
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));

      for (const unsigned int i : fe_values.dof_indices()) //@suppress("Symbol is not resolved")
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }


  std::map<types::global_dof_index, double> boundary_values; //will be the output object after the call to interpolate_boundary_values
  //list of pairs of global degree of freedom numbers (i.e. the number of the degrees of freedom on the boundary) and their boundary values

  VectorTools::interpolate_boundary_values(dof_handler,
                                           0, //boundary indicator on Dirichlet boundary
                                           Functions::ZeroFunction<2>(), /*Functions::ConstantFunction<2,double>(1,1)*/
										   boundary_values);
//  for(auto x : boundary_values ){
//	  std::cout << x.first <<"\n";
//  }

  MatrixTools::apply_boundary_values(boundary_values, //modify the system of equations accordingly to boundary DoFs and their boundary values
                                     system_matrix,
                                     solution,
                                     system_rhs);
}



void Step3::solve()
{
  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);

  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  std::cout << std::setprecision(8) << "Mean value: " << VectorTools::compute_mean_value (dof_handler,
                                            QGauss<2>(fe.degree + 1),
                                            solution,
                                            0) << std::endl;
}



void Step3::output_results() const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  std::ofstream output("solution.vtk");
  data_out.write_vtk(output);
}

void Step3::observe_convergence(){
	const unsigned int n_of_runs{8};
	for(unsigned int it =1;it<n_of_runs;++it){
		Step3 laplace_problem;
		laplace_problem.make_grid(it);
		laplace_problem.setup_system();
		laplace_problem.assemble_system();
		laplace_problem.solve();
	}
}


void Step3::run()
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}



int main()
{
  deallog.depth_console(2);

  Step3 laplace_problem;
  laplace_problem.run();
  laplace_problem.observe_convergence();

  return 0;
}
