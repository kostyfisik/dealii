/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2006 - 2015 by the deal.II authors
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
 * Author: Hongfeng Ma, Laboratoire Hubert Curien / Universite de Lyon, 2016
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/utilities.h>

namespace Maxwell
{
  using namespace dealii;

  template <int dim>
  class MaxwellTD
  {
  public:
    MaxwellTD (const unsigned int degree);
    void run();

  private:
    void setup_system();
    void assemble_system();
    void solve_b();
    void solve_e();
    void output_results() const;

    const unsigned int degree;

    Triangulation<dim>   triangulation;
    FESystem<dim>        fe;
    DoFHandler<dim>      dof_handler;

    ConstraintMatrix constraints;


    BlockSparsityPattern      sparsity_pattern;

    //Defining matrixs with size of 2X2
    //G + delta_t/2 * P is actually lhs_GP.block(0,0)
    //lhs_GP is a system_matrix assembled from each cells, namely, local_lhs_GP
    BlockSparseMatrix<double>    lhs_GP, rhs_KT, rhs_GP;
    BlockSparseMatrix<double>    lhs_CS, rhs_K, rhs_CS, rhs_Q;

    BlockVector<double> solution;        //solution_b, solution_e
    BlockVector<double> old_solution;    //old_solu_b, old_solu_e

    BlockVector<double> system_rhs;
    BlockVector<double> system_power;

    double time, time_step;
    unsigned int timestep_number;

    double mu_r  = 1;
    double eps_r = 1;
    double sigma_e = 0;
    double sigma_m = 0;
  };



  // TimePulseFactor, used during run()

  template <int dim>
  class TimePulseFactor : public Function<dim>
  {
  public:
    TimePulseFactor() : Function<dim>() {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  // Each object derived from the Function class has a time field that
  // can be set using the Function::set_time and read by
  // Function::get_time.
  template <int dim>
  double TimePulseFactor<dim>::value (const Point<dim> &/*p*/, const unsigned int component) const
  {
    Assert (component == 0, ExcInternalError());

    if (this->get_time() <= 0.5)
      return std::sin (this->get_time() * 4 * numbers::PI);
    else
      return 0;
  }



  // Finally, the incident power as face integration over face boundaries
  template <int dim>
  class PowerBoundaryValues : public Function<dim>
  {
  public:
    PowerBoundaryValues() : Function<dim> (dim) {}
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;
    virtual void vector_value_list (const std::vector<Point<dim>> &points,
                                    std::vector<Vector<double>>   &value_list) const;
  };


  template <int dim>
  void PowerBoundaryValues<dim>::vector_value (const Point<dim> &p,
                                               Vector<double>   &values) const
  {
    Assert (values.size() == dim,
            ExcDimensionMismatch (values.size(), dim));
    Assert (dim > 2, ExcNotImplemented());

    //define system power incidence direction
    //spherical coordinate system
    //theta  is the angle of vector S with Sz
    //phi    is the angle of vector S*cos (theta) with Sx
    //in rad.....
    double incid_theta = 0;
    double incid_phi   = 0;
    double H_magnitude = 1;

    double sin_theta = std::sin (incid_theta);
    double cos_theta = std::cos (incid_theta);
    double sin_phi   = std::sin (incid_phi);
    double cos_phi   = std::cos (incid_phi);

    //    double Z_impendence = 1 * 376.7;
    //    double var_t_pulse = 0;

    //define time dependent value
    /*    if ( (this->get_time() <= 0.5)
          var_t_pulse = std::sin (this->get_time() * 4 * numbers::PI);
          else
          var_t_pulse = 0;
    */

    //define H vector, here assuming mu_r = 1
    if (p[2] >= 0.99)
      {
        values (0) = (sin_theta * sin_theta * sin_phi
                     + cos_theta * cos_theta * cos_phi) * H_magnitude;
        values (1) = (sin_theta * sin_theta * cos_phi
                     + cos_theta * cos_theta * sin_phi) * H_magnitude * (-1);
        values (2) = (sin_theta * cos_theta) * H_magnitude;
        //        return 1;
      }
    else
      {
        values (0) = 0;
        values (1) = 0;
        values (2) = 0;
        //          return 0;
      }
  }



  template <int dim>
  void PowerBoundaryValues<dim>::vector_value_list (const std::vector<Point<dim>> &points,
                                                    std::vector<Vector<double>>   &value_list ) const
  {
    Assert (value_list.size() == points.size(),
            ExcDimensionMismatch (value_list.size(), points.size()));
    const unsigned int n_points = points.size();

    for (unsigned int p = 0; p<n_points; ++p)
      PowerBoundaryValues<dim>::vector_value (points[p], value_list[p]);
  }



  // Constructor
  template <int dim>
  MaxwellTD<dim>::MaxwellTD (const unsigned int degree)
    :
    degree (degree),
    fe (FE_RaviartThomas<dim> (degree), 1, FE_Nedelec<dim> (degree), 1),
    dof_handler (triangulation),
    time_step (1./64)
  {}



  template <int dim>
  void MaxwellTD<dim>::setup_system()
  {
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (4);

    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl;

    dof_handler.distribute_dofs (fe);

    std::cout << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    DoFRenumbering::component_wise (dof_handler);

    std::vector<types::global_dof_index> dofs_per_comonent (dim+dim);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_comonent);
    const unsigned int n_b = dofs_per_comonent[0],
      n_e = dofs_per_comonent[dim];

    std::cout << "n_b: " << n_b << std::endl << "n_e: " << n_e << std::endl;

    BlockDynamicSparsityPattern dsp (2, 2);
    dsp.block(0,0).reinit (n_b, n_b);
    dsp.block(0,1).reinit (n_b, n_e);
    dsp.block(1,0).reinit (n_e, n_b);
    dsp.block(1,1).reinit (n_e, n_e);
    dsp.collect_sizes();

    DoFTools::make_sparsity_pattern (dof_handler, dsp);
    sparsity_pattern.copy_from (dsp);

    lhs_GP.reinit (sparsity_pattern);
    rhs_KT.reinit (sparsity_pattern);
    rhs_GP.reinit (sparsity_pattern);

    lhs_CS.reinit (sparsity_pattern);
    rhs_K.reinit (sparsity_pattern);
    rhs_CS.reinit (sparsity_pattern);
    rhs_Q.reinit (sparsity_pattern);

    solution.reinit (2);
    solution.block(0).reinit (n_b);
    solution.block(1).reinit (n_e);
    solution.collect_sizes();

    old_solution.reinit (2);
    old_solution.block(0).reinit (n_b);
    old_solution.block(1).reinit (n_e);
    old_solution.collect_sizes();

    system_rhs.reinit (2);
    system_rhs.block(0).reinit (n_b);
    system_rhs.block(1).reinit (n_e);
    system_rhs.collect_sizes();

    system_power.reinit (2);
    system_power.block(0).reinit (n_b);
    system_power.block(1).reinit (n_e);
    system_power.collect_sizes();
  }



  template <int dim>
  void MaxwellTD<dim>::assemble_system()
  {
    QGauss<dim>        quadrature_formula (degree+2);
    QGauss<dim-1>    face_quadrature_formula (degree+2);

    FEValues<dim> fe_values (fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values | update_normal_vectors |
                                      update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    std::cout << "dofs_per_cell: " << dofs_per_cell << std::endl;
    std::cout << "n_q_points: " << n_q_points << std::endl;
    std::cout << "n_face_q_points: " << n_face_q_points << std::endl;

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    PowerBoundaryValues<dim>    power_boundary_values;

    std::vector<Vector<double>> boundary_values (n_face_q_points, Vector<double> (dim));

    FullMatrix<double>
      local_lhs_GP    (dofs_per_cell, dofs_per_cell),
      local_rhs_KT  (dofs_per_cell, dofs_per_cell),
      local_rhs_GP (dofs_per_cell, dofs_per_cell);

    FullMatrix<double>
      local_lhs_CS (dofs_per_cell, dofs_per_cell),
      local_rhs_K  (dofs_per_cell, dofs_per_cell),
      local_rhs_CS (dofs_per_cell, dofs_per_cell),
      local_rhs_Q  (dofs_per_cell, dofs_per_cell);

    //define local power
    Vector<double>    local_power    (dofs_per_cell);
    Tensor<1, dim>   power_rhs_H;

    // corresponds to constructor`s
    //     fe (FE_RaviartThomas<dim> (degree), 1, FE_Nedelec<dim> (degree), 1),
    const FEValuesExtractors::Vector B_field (0);
    const FEValuesExtractors::Vector E_field (dim);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        fe_values.reinit (cell);

        local_lhs_GP = 0;
        local_rhs_KT  = 0;
        local_rhs_GP = 0;

        local_lhs_CS = 0;
        local_rhs_K  = 0;
        local_rhs_CS = 0;
        local_rhs_Q  = 0;

        local_power = 0;

        // std::cout << "in cells..assembling.." << std::endl;
        
        // Using notations from paper by Rodrigue and White, 2001
        // W - H(curl) edge elements for electric field
        // F - H(div) face elements for magnetic field
        for (unsigned int q = 0; q<n_q_points; ++q)
          for (unsigned int i = 0; i<dofs_per_cell; ++i)
            {
              const auto &F_i      = fe_values[B_field].value (i, q);
              const auto &curl_W_i = fe_values[E_field].curl  (i, q);
              const auto &W_i      = fe_values[E_field].value (i, q);

              for (unsigned int j = 0; j<dofs_per_cell; ++j)
                {
                  const auto &F_j      = fe_values[B_field].value (j, q);
                  const auto &curl_W_j = fe_values[E_field].curl  (j, q);
                  const auto &W_j      = fe_values[E_field].value (j, q);
                  // G + delta_t/2 * P
                  local_lhs_GP (i, j) += F_i * F_j * fe_values.JxW (q)
                    * (1/mu_r + 1/mu_r * sigma_m * 1/mu_r * time_step/2);
                  // -delta_t * K^{T}
                  local_rhs_KT (i, j) += curl_W_j * F_i * fe_values.JxW (q)
                    * -time_step / mu_r;
                  // G - delta_t/2 * P
                  local_rhs_GP (i, j) += F_i * F_j * fe_values.JxW (q)
                    * (1/mu_r - 1/mu_r * sigma_m * 1/mu_r * time_step/2);
                  // C + delta_t/2 * S
                  local_lhs_CS (i, j) += W_i * W_j * fe_values.JxW (q)
                    * (eps_r + time_step/2 * sigma_e);
                  // delta_t * K
                  local_rhs_K (i, j) += curl_W_i * F_j * fe_values.JxW (q)
                    * time_step * 1/mu_r;
                  // C - delta_t/2 * S
                  local_rhs_CS (i, j) += W_i * W_j * fe_values.JxW (q)
                    * (eps_r - time_step/2 * sigma_e);
                  // -delta_t * Q
                  local_rhs_Q (i, j)  += F_i * W_j * fe_values.JxW (q)
                    * -time_step;
                }
            }

        for (unsigned int face_n = 0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
          if (cell->at_boundary (face_n))
            {
              fe_face_values.reinit (cell, face_n);

              power_boundary_values.vector_value_list
                (fe_face_values.get_quadrature_points(), boundary_values);

              for (unsigned int q = 0; q<n_face_q_points; ++q)
                {
                  power_rhs_H[0] = boundary_values[q] (0);
                  power_rhs_H[1] = boundary_values[q] (1);
                  power_rhs_H[2] = boundary_values[q] (2);
                  /*
                    std::cout << "power_rhs_H: " << power_rhs_H  << std::endl;
                    std::cout << "cross_values..."  << cross_product_3d (power_rhs_H, fe_face_values[E_field].value (4, q)) * fe_face_values.normal_vector (q) * fe_face_values.JxW (q) << std::endl;
                  */

                  for (unsigned int i = 0; i<dofs_per_cell; ++i)
                    local_power (i) += cross_product_3d (power_rhs_H,
                      fe_face_values[E_field].value (i, q))
                       * fe_face_values.normal_vector (q) * fe_face_values.JxW (q);
                  //local_power (i) += 1;
                }
            }
        //        std::cout << "local function..assembling..ok!" << std::endl;

        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i = 0; i<dofs_per_cell; ++i)
          {
            const auto &dof_i = local_dof_indices[i];
            system_power (dof_i) += local_power (i);
            for (unsigned int j = 0; j<dofs_per_cell; ++j)
              {
                const auto &dof_j = local_dof_indices[j];
                lhs_GP.add (dof_i, dof_j, local_lhs_GP (i, j));
                rhs_KT.add (dof_i, dof_j, local_rhs_KT (i, j));
                rhs_GP.add (dof_i, dof_j, local_rhs_GP (i, j));

                lhs_CS.add (dof_i, dof_j, local_lhs_CS (i, j));
                rhs_K.add (dof_i, dof_j, local_rhs_K (i, j));
                rhs_CS.add (dof_i, dof_j, local_rhs_CS (i, j));
                rhs_Q.add (dof_i, dof_j, local_rhs_Q (i, j));
              }
          }
      }            //end of cells loop

    std::cout << "end of assembling..ok!" << std::endl;
  }        //end of assemble_system



  // TODO: Consider applying non-trivial preconditioner to solve_b and solve_e
  // e.g. see step-20 precodition_Jacobi //Kostya
  template <int dim>
  void MaxwellTD<dim>::solve_b()
  {
    SolverControl           solver_control (1000, 1e-12*system_rhs.l2_norm());
    SolverCG<>              cg (solver_control);

    cg.solve (lhs_GP.block(0,0), solution.block(0), system_rhs.block(0),
              PreconditionIdentity());

    std::cout << "   u-equation: " << solver_control.last_step()
              << " CG iterations."
              << std::endl;
  }



  template <int dim>
  void MaxwellTD<dim>::solve_e()
  {
    SolverControl           solver_control (1000, 1e-12*system_rhs.l2_norm());
    SolverCG<>              cg (solver_control);

    cg.solve (lhs_CS.block(1,1), solution.block(1), system_rhs.block(1),
              PreconditionIdentity());

    std::cout << "   v-equation: " << solver_control.last_step()
              << " CG iterations."
              << std::endl;
  }



  template <int dim>
  void MaxwellTD<dim>::output_results() const
  {
    DataOut<dim> data_out;

    std::vector<std::string> solution_names;
    switch (dim)
      {
      case 2:  // TODO: It should be TM or TE case //Kostya
        solution_names.push_back ("Bx");
        solution_names.push_back ("By");
        solution_names.push_back ("Ex");
        solution_names.push_back ("Ey");
        break;

      case 3:
        solution_names.push_back ("Bx");
        solution_names.push_back ("By");
        solution_names.push_back ("Bz");
        solution_names.push_back ("Ex");
        solution_names.push_back ("Ey");
        solution_names.push_back ("Ez");
        break;

      default:
        Assert (false, ExcNotImplemented());
      }

    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, solution_names);

    data_out.build_patches();

    std::ostringstream filename;
    filename << "solution-"
             <<    Utilities::int_to_string (timestep_number, 3)
             <<    ".vtk";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);
  }



  template <int dim>
  void MaxwellTD<dim>::run()
  {
    setup_system();
    std::cout << "setup system...ok! " << std::endl;

    assemble_system();
    std::cout << "assembling...ok! " << std::endl;

    // (For additions, we need a temporary vector that we declare
    // before the loop to avoid repeated memory allocations in each
    // time step.)
    Vector<double> tmp1 (system_rhs.block(0).size());
    Vector<double> tmp2 (system_rhs.block(1).size());

    TimePulseFactor<dim> time_pulse_factor;
     Point<dim> p_time;

    for (time = time_step, timestep_number = 1;
         time <= 2.5;
         time += time_step, ++timestep_number)
      {
        std::cout << "Time step " << timestep_number
                  << " at t = " << time
                  << std::endl;

        std::cout << "____ " << old_solution.block(0).size() <<  std::endl
            << ";;;;;;" << rhs_KT.block(0, 0).m() << std::endl;

        // KT*e_old + GP*b_old
        rhs_KT.block(0,1).vmult ( system_rhs.block(0), old_solution.block(1) );
        rhs_GP.block(0,0).vmult ( tmp1, old_solution.block(0));
        system_rhs.block(0).add (1, tmp1);

        // After so constructing the right hand side vector of the first
        // equation, all we have to do is apply the correct boundary
        // values. As for the right hand side, this is a space-time function
        // evaluated at a particular time, which we interpolate at boundary
        // nodes and then use the result to apply boundary values as we
        // usually do. The result is then handed off to the solve_u()
        // function:

        /*{
          BoundaryValuesB<dim> boundary_values_b_function;
          boundary_values_b_function.set_time (time);

          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values (dof_handler,
                                                    0,
                                                    ZeroFunction<dim> (dim+dim),
                                                    boundary_values);

          std::cout << "boundary B...OK " << std::endl;

          MatrixTools::apply_boundary_values (boundary_values,
                                              lhs_GP.block(0,0),
                                              solution.block(0),
                                              system_rhs.block(0));
        }*/

        solve_b();

        time_pulse_factor.set_time (time);
        
        // K*b_new +CS*e_old
        rhs_K.block(1,0).vmult (system_rhs.block(1), solution.block(0));
        rhs_CS.block(1,1).vmult (tmp2, old_solution.block(1));
        system_rhs.block(1).add (1, tmp2);
        
        system_rhs.block(1).add (time_pulse_factor.value (p_time, 0), system_power.block(1));

        //simply ignoring current J first

        // {
        //   BoundaryValuesE<dim> boundary_values_e_function;
        //   boundary_values_e_function.set_time (time);

        //   std::map<types::global_dof_index, double> boundary_values;
        //   VectorTools::interpolate_boundary_values (dof_handler,
        //                                             0,
        //                                             boundary_values_e_function,
        //                                             boundary_values);

        //   MatrixTools::apply_boundary_values (boundary_values,
        //                                       lhs_CS.block(1,1),
        //                                       solution.block(1),
        //                                       system_rhs.block(1));
        // }

        solve_e();

        output_results();
        old_solution = solution;
      }
  }
}



int main()
{
  try
    {
      using namespace dealii;
      using namespace Maxwell;

      MaxwellTD<3> wave_equation_solver (0);
      wave_equation_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
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
      std::cerr << std::endl << std::endl
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
