/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2006 - 2016 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2006
 * Author: Konstantin Ladutenko, ITMO University, 2016
 */

// This is an attempt to convert step-23 into time-domain Maxwell solver.

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

// The last step is as in all previous programs:
namespace Step23 {
 using namespace dealii;

// @sect3{The <code>MaxwellTD</code> class}

// Next comes the declaration of the main class. It's public interface of
// functions is like in most of the other tutorial programs. Worth
// mentioning is that we now have to store four matrices instead of one: the
// mass matrix $M$, the Laplace matrix $A$, the matrix $M+k^2\theta^2A$ used
// for solving for $U^n$, and a copy of the mass matrix with boundary
// conditions applied used for solving for $V^n$. Note that it is a bit
// wasteful to have an additional copy of the mass matrix around. We will
// discuss strategies for how to avoid this in the section on possible
// improvements.
//
// Likewise, we need solution vectors for $U^n,V^n$ as well as for the
// corresponding vectors at the previous time step, $U^{n-1},V^{n-1}$. The
// <code>system_rhs</code> will be used for whatever right hand side vector
// we have when solving one of the two linear systems in each time
// step. These will be solved in the two functions <code>solve_u</code> and
// <code>solve_v</code>.
//
// Finally, the variable <code>theta</code> is used to indicate the
// parameter $\theta$ that is used to define which time stepping scheme to
// use, as explained in the introduction. The rest is self-explanatory.
template <int dim>
class MaxwellTD {
public:
  MaxwellTD (const unsigned int degree);
  void run ();

private:
  void setup_system();
  void solve_u();
  void solve_v();
  void output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  FESystem<dim> mt_fe;
  DoFHandler<dim> mt_dof_handler;

  ConstraintMatrix constraints;

  SparsityPattern sparsity_pattern;
  SparsityPattern mt_sparsity_pattern;

  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> laplace_matrix;
  SparseMatrix<double> matrix_u;
  SparseMatrix<double> matrix_v;

  SparseMatrix<double> mt_matrix_e;
  SparseMatrix<double> mt_matrix_b;
  SparseMatrix<double> mt_G;
  SparseMatrix<double> mt_K;
  SparseMatrix<double> mt_P;
  SparseMatrix<double> mt_C;
  SparseMatrix<double> mt_Kt;
  SparseMatrix<double> mt_S;
  SparseMatrix<double> mt_Q;

  Vector<double> solution_u, solution_v;
  Vector<double> old_solution_u, old_solution_v;
  Vector<double> system_rhs;

  unsigned int p_degree;
  unsigned int quad_degree;

  int refine_num = 2;
  double time, time_step;
  unsigned int timestep_number;
  const double theta;
};


  // Set material parameters
template <int dim> double mu_rev (const Point<dim> &p) {return 1.0;}
template <int dim> double eps (const Point<dim> &p) {return 1.0;}
template <int dim>
FullMatrix<double> sigma_m (const Point<dim> &p) {
  FullMatrix<double>   sigma (dim,dim);
  sigma = 0.0;
  for (int i=0; i<dim; ++i){
    sigma(i,i) = 1.0;
  }
  return sigma;
}
template <int dim>
FullMatrix<double> sigma_e (const Point<dim> &p) {
  FullMatrix<double>   sigma (dim,dim);
  sigma = 0.0;
  for (int i=0; i<dim; ++i){
    sigma(i,i) = 1.0;
  }
  return sigma;
}



  
template <int dim>
class InitialValuesU : public Function<dim> {
public:
  InitialValuesU() : Function<dim>() {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const;
};

template <int dim>
class InitialValuesV : public Function<dim> {
public:
  InitialValuesV() : Function<dim>() {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const;
};

template <int dim>
double InitialValuesU<dim>::value(const Point<dim> & /*p*/,
                                  const unsigned int component) const {
  Assert(component == 0, ExcInternalError());
  return 0;
}

template <int dim>
double InitialValuesV<dim>::value(const Point<dim> & /*p*/,
                                  const unsigned int component) const {
  Assert(component == 0, ExcInternalError());
  return 0;
}

// Secondly, we have the right hand side forcing term. Boring as we are, we
// choose zero here as well:
template <int dim> class RightHandSide : public Function<dim> {
public:
  RightHandSide() : Function<dim>() {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const;
};

template <int dim>
double RightHandSide<dim>::value(const Point<dim> & /*p*/,
                                 const unsigned int component) const {
  Assert(component == 0, ExcInternalError());
  return 0;
}

// Finally, we have boundary values for $u$ and $v$. They are as described
// in the introduction, one being the time derivative of the other:
template <int dim> class BoundaryValuesU : public Function<dim> {
public:
  BoundaryValuesU() : Function<dim>() {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const;
};

template <int dim> class BoundaryValuesV : public Function<dim> {
public:
  BoundaryValuesV() : Function<dim>() {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const;
};

template <class T> inline T pow2(const T value) { return value * value; }
double dwidth = 0.3;
double get_amplitude(double time) {
  return 0.0 * time;
  // double width = 0.5;
  // double delay = 2.0*width;
  // return  std::exp( -pow2(time-delay) / (2*pow2(width)) );
}
template <int dim> double get_shape(const Point<dim> &p) {
  return 0.0 * p[0];
  // return std::cos(numbers::PI/2.0*p[1]/dwidth);
  // return 1.0;
}

template <int dim>
double BoundaryValuesU<dim>::value(const Point<dim> &p,
                                   const unsigned int component) const {
  Assert(component == 0, ExcInternalError());

  if ((p[0] <= 0))
    return get_shape(p) * get_amplitude(this->get_time()) *
           std::sin(this->get_time() * 4 * numbers::PI);
  else

    return 0;
}

template <int dim>
double BoundaryValuesV<dim>::value(const Point<dim> &p,
                                   const unsigned int component) const {
  Assert(component == 0, ExcInternalError());
  return 0;

  if ((p[0] <= 0) // &&
      // (p[1] < 1./3) &&
      // (p[1] > -1./3)
      )
    return get_shape(p) * get_amplitude(this->get_time()) *
           (std::cos(this->get_time() * 4 * numbers::PI) * 4 * numbers::PI);
  else
    return 0;
}

template <int dim>
MaxwellTD<dim>::MaxwellTD(const unsigned int degree)
  : fe(degree), mt_fe(FE_Nedelec<dim>(degree), 1, FE_RaviartThomas<dim>(degree), 1),dof_handler(triangulation), mt_dof_handler(triangulation), time_step(1. / 64), theta(0.5)
{
  p_degree = degree;
  quad_degree = p_degree+2;
}

// @sect4{MaxwellTD::setup_system}

// The next function is the one that sets up the mesh, DoFHandler, and
// matrices and vectors at the beginning of the program, i.e. before the
// first time step. The first few lines are pretty much standard if you've
// read through the tutorial programs at least up to step-6:
template <int dim> void MaxwellTD<dim>::setup_system() {
  Point<dim> left, right;
  int factor = 14;
  dwidth = 0.3;
  std::vector<unsigned int> repetitions;
  for (int i = 0; i < dim; ++i) {
    left[i] = -dwidth;
    right[i] = dwidth;
    repetitions.push_back(3);
  }
  left[0] *= 0;
  right[0] *= factor * 2.0;
  repetitions[0] *= factor;

  GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, left,
                                            right);
  triangulation.refine_global(refine_num);
  time_step = dwidth / std::pow(2, refine_num + 1);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;

  dof_handler.distribute_dofs(fe);
  mt_dof_handler.distribute_dofs(mt_fe);

  // solution.reinit (dof_handler.n_dofs());
  // system_rhs.reinit (dof_handler.n_dofs());
  // constraints.clear ();
  // DoFTools::make_hanging_node_constraints (dof_handler,
  //                                          constraints);
  // // FE_Nedelec boundary condition.
  // VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, ExactSolution<dim>(), 0, constraints);

  // constraints.close ();

  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << " and " << mt_dof_handler.n_dofs()
            << std::endl
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  DynamicSparsityPattern mt_dsp(mt_dof_handler.n_dofs(), mt_dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(mt_dof_handler, mt_dsp);
  mt_sparsity_pattern.copy_from(mt_dsp);

  std::vector<unsigned int> mt_block_component (2*dim,1);
  for (int i = 0; i < dim; ++i) mt_block_component[i] = 0;
  DoFRenumbering::component_wise (mt_dof_handler, mt_block_component);

  mass_matrix.reinit(sparsity_pattern);
  laplace_matrix.reinit(sparsity_pattern);
  matrix_u.reinit(sparsity_pattern);
  matrix_v.reinit(sparsity_pattern);

  MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(3), mass_matrix);
  MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(3),
                                       laplace_matrix);

  solution_u.reinit(dof_handler.n_dofs());
  solution_v.reinit(dof_handler.n_dofs());
  old_solution_u.reinit(dof_handler.n_dofs());
  old_solution_v.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.close();
}
  
template <int dim> void MaxwellTD<dim>::solve_u() {
  SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
  SolverCG<> cg(solver_control);

  cg.solve(matrix_u, solution_u, system_rhs, PreconditionIdentity());

  std::cout << "   u-equation: " << solver_control.last_step()
            << " CG iterations." << std::endl;
}
  
template <int dim> void MaxwellTD<dim>::solve_v() {
  SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
  SolverCG<> cg(solver_control);

  cg.solve(matrix_v, solution_v, system_rhs, PreconditionIdentity());

  std::cout << "   v-equation: " << solver_control.last_step()
            << " CG iterations." << std::endl;
}

template <int dim> void MaxwellTD<dim>::output_results() const {
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_u, "U");
  data_out.add_data_vector(solution_v, "V");

  data_out.build_patches();

  const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";
  std::ofstream output(filename.c_str());
  data_out.write_vtu(output);
}

template <int dim> void MaxwellTD<dim>::run() {
  setup_system();
  const FEValuesExtractors::Vector e_field (0);
  const FEValuesExtractors::Vector b_field (dim);
  
  VectorTools::project(dof_handler, constraints, QGauss<dim>(3),
                       InitialValuesU<dim>(), old_solution_u);
  VectorTools::project(dof_handler, constraints, QGauss<dim>(3),
                       InitialValuesV<dim>(), old_solution_v);

  // The next thing is to loop over all the time steps until we reach the
  // end time ($T=5$ in this case). In each time step, we first have to
  // solve for $U^n$, using the equation $(M^n + k^2\theta^2 A^n)U^n =$
  // $(M^{n,n-1} - k^2\theta(1-\theta) A^{n,n-1})U^{n-1} + kM^{n,n-1}V^{n-1}
  // +$ $k\theta \left[k \theta F^n + k(1-\theta) F^{n-1} \right]$. Note
  // that we use the same mesh for all time steps, so that $M^n=M^{n,n-1}=M$
  // and $A^n=A^{n,n-1}=A$. What we therefore have to do first is to add up
  // $MU^{n-1} - k^2\theta(1-\theta) AU^{n-1} + kMV^{n-1}$ and the forcing
  // terms, and put the result into the <code>system_rhs</code> vector. (For
  // these additions, we need a temporary vector that we declare before the
  // loop to avoid repeated memory allocations in each time step.)
  //
  // The one thing to realize here is how we communicate the time variable
  // to the object describing the right hand side: each object derived from
  // the Function class has a time field that can be set using the
  // Function::set_time and read by Function::get_time. In essence, using
  // this mechanism, all functions of space and time are therefore
  // considered functions of space evaluated at a particular time. This
  // matches well what we typically need in finite element programs, where
  // we almost always work on a single time step at a time, and where it
  // never happens that, for example, one would like to evaluate a
  // space-time function for all times at any given spatial location.
  Vector<double> tmp(solution_u.size());
  Vector<double> forcing_terms(solution_u.size());

  // for (
  timestep_number = 1, time = time_step;
  // time <= 25; time += time_step, ++timestep_number)
  {
    std::cout << "Time step " << timestep_number << " at t=" << time
              << " amp =" << get_amplitude(time) << std::endl;

    mass_matrix.vmult(system_rhs, old_solution_u);
    
    mass_matrix.vmult(tmp, old_solution_v);
    system_rhs.add(time_step, tmp);

    laplace_matrix.vmult(tmp, old_solution_u);
    system_rhs.add(-theta * (1 - theta) * time_step * time_step, tmp);

    RightHandSide<dim> rhs_function;
    rhs_function.set_time(time);
    VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(2),
                                        rhs_function, tmp);
    forcing_terms = tmp;
    forcing_terms *= theta * time_step;

    rhs_function.set_time(time - time_step);
    VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(2),
                                        rhs_function, tmp);

    forcing_terms.add((1 - theta) * time_step, tmp);

    system_rhs.add(theta * time_step, forcing_terms);

    // After so constructing the right hand side vector of the first
    // equation, all we have to do is apply the correct boundary
    // values. As for the right hand side, this is a space-time function
    // evaluated at a particular time, which we interpolate at boundary
    // nodes and then use the result to apply boundary values as we
    // usually do. The result is then handed off to the solve_u()
    // function:
    {
      BoundaryValuesU<dim> boundary_values_u_function;
      boundary_values_u_function.set_time(time);

      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(
          dof_handler, 0, boundary_values_u_function, boundary_values);

      matrix_u.copy_from(mass_matrix);
      matrix_u.add(theta * theta * time_step * time_step, laplace_matrix);
      MatrixTools::apply_boundary_values(boundary_values, matrix_u, solution_u,
                                         system_rhs);
    }
    solve_u();

    // The second step, i.e. solving for $V^n$, works similarly, except
    // that this time the matrix on the left is the mass matrix (which we
    // copy again in order to be able to apply boundary conditions, and
    // the right hand side is $MV^{n-1} - k\left[ \theta A U^n +
    // (1-\theta) AU^{n-1}\right]$ plus forcing terms. %Boundary values
    // are applied in the same way as before, except that now we have to
    // use the BoundaryValuesV class:
    laplace_matrix.vmult(system_rhs, solution_u);
    system_rhs *= -theta * time_step;

    mass_matrix.vmult(tmp, old_solution_v);
    system_rhs += tmp;

    laplace_matrix.vmult(tmp, old_solution_u);
    system_rhs.add(-time_step * (1 - theta), tmp);

    system_rhs += forcing_terms;

    {
      BoundaryValuesV<dim> boundary_values_v_function;
      boundary_values_v_function.set_time(time);

      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(
          dof_handler, 0, boundary_values_v_function, boundary_values);
      matrix_v.copy_from(mass_matrix);
      MatrixTools::apply_boundary_values(boundary_values, matrix_v, solution_v,
                                         system_rhs);
    }
    solve_v();

    // Finally, after both solution components have been computed, we
    // output the result, compute the energy in the solution, and go on to
    // the next time step after shifting the present solution into the
    // vectors that hold the solution at the previous time step. Note the
    // function SparseMatrix::matrix_norm_square that can compute
    // $\left<V^n,MV^n\right>$ and $\left<U^n,AU^n\right>$ in one step,
    // saving us the expense of a temporary vector and several lines of
    // code:
    output_results();

    std::cout << "   Total energy: "
              << (mass_matrix.matrix_norm_square(solution_v) +
                  laplace_matrix.matrix_norm_square(solution_u)) /
                     2
              << std::endl;

    old_solution_u = solution_u;
    old_solution_v = solution_v;
  }
}
}

// @sect3{The <code>main</code> function}

// What remains is the main function of the program. There is nothing here
// that hasn't been shown in several of the previous programs:
int main() {
  try {
    using namespace dealii;
    using namespace Step23;
    int degree = 1;
    MaxwellTD<2> maxwell_td_solver(degree);
    maxwell_td_solver.run();
  } catch (std::exception &exc) {
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
  } catch (...) {
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
