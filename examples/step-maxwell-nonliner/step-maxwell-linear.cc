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
    void run ();

  private:
    void setup_system ();
    void assemble_system();
    void solve_b ();
    void solve_e ();
    void output_results () const;

    const unsigned int degree;

    Triangulation<dim>   triangulation;
    FESystem<dim>        fe;
    DoFHandler<dim>      dof_handler;

    ConstraintMatrix constraints;


    BlockSparsityPattern      sparsity_pattern;
    
    //Defining matrixs with size of 2X2
    //G + delta_t/2 * P is actually sys_matrix_b.block(0, 0)
    //sys_matrix_b is a system_matrix assembled from each cells, namely, local_matrix
    BlockSparseMatrix<double>    sys_matrix_b,
      rhs_matrix_k_b, rhs_matrix_gp_b;
    BlockSparseMatrix<double>    sys_matrix_e,
      rhs_matrix_k_e, rhs_matrix_cs_e, rhs_matrix_q_e;
    
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



  // TimePulseFactor, used during run ()

  template <int dim>
  class TimePulseFactor : public Function<dim>
  {
  public:
    TimePulseFactor () : Function<dim> () {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  // Each object derived from the Function class has a time field that
  // can be set using the Function::set_time and read by
  // Function::get_time.
  template <int dim>
  double TimePulseFactor<dim>::value (const Point<dim> &/*p*/, const unsigned int component) const
  {
    Assert (component == 0, ExcInternalError ());
    
    if (this->get_time () <= 0.5)
      return std::sin (this->get_time () * 4 * numbers::PI);
    else
      return 0;
  }
  


  // Finally, the incident power as face integration over face boundaries
  template <int dim>
  class PowerBoundaryValues : public Function<dim>
  {
  public:
    PowerBoundaryValues () : Function<dim> (dim) {}
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;
    virtual void vector_value_list (const std::vector<Point<dim>> &points,
                                    std::vector<Vector<double>>      &value_list) const;
  };


  template <int dim>
  void PowerBoundaryValues<dim>::vector_value (const Point<dim>    &p,
                                               Vector<double>   &values) const
  {
    Assert (values.size () == dim,
            ExcDimensionMismatch (values.size (), dim));
    Assert (dim > 2, ExcNotImplemented ());
    
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
    /*    if ( (this->get_time () <= 0.5)
          var_t_pulse = std::sin (this->get_time () * 4 * numbers::PI);
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
    Assert (value_list.size () == points.size (),
            ExcDimensionMismatch (value_list.size (), points.size ()));
    const unsigned int n_points = points.size ();
    
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
  void MaxwellTD<dim>::setup_system ()
  {
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (4);

    std::cout << "Number of active cells: "
              << triangulation.n_active_cells ()
              << std::endl;

    dof_handler.distribute_dofs (fe);

    std::cout << "Number of degrees of freedom: "
              << dof_handler.n_dofs ()
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
    dsp.collect_sizes ();

    DoFTools::make_sparsity_pattern (dof_handler, dsp);
    sparsity_pattern.copy_from (dsp);

    sys_matrix_b.reinit (sparsity_pattern);
    rhs_matrix_k_b.reinit (sparsity_pattern);
    rhs_matrix_gp_b.reinit (sparsity_pattern);

    sys_matrix_e.reinit (sparsity_pattern);
    rhs_matrix_k_e.reinit (sparsity_pattern);
    rhs_matrix_cs_e.reinit (sparsity_pattern);
    rhs_matrix_q_e.reinit (sparsity_pattern);

    solution.reinit (2);
    solution.block(0).reinit (n_b);
    solution.block(1).reinit (n_e);
    solution.collect_sizes ();
    
    old_solution.reinit (2);
    old_solution.block(0).reinit (n_b);
    old_solution.block(1).reinit (n_e);
    old_solution.collect_sizes ();
    
    system_rhs.reinit (2);
    system_rhs.block(0).reinit (n_b);
    system_rhs.block(1).reinit (n_e);
    system_rhs.collect_sizes ();
    
    system_power.reinit (2);
    system_power.block(0).reinit (n_b);
    system_power.block(1).reinit (n_e);
    system_power.collect_sizes ();
  }


  
  template <int dim>
  void MaxwellTD<dim>::assemble_system ()
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
    const unsigned int n_q_points      = quadrature_formula.size ();
    const unsigned int n_face_q_points = face_quadrature_formula.size ();
    
    std::cout << "dofs_per_cell: " << dofs_per_cell << std::endl;
    std::cout << "n_q_points: " << n_q_points << std::endl;
    std::cout << "n_face_q_points: " << n_face_q_points << std::endl;
    
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    PowerBoundaryValues<dim>    power_boundary_values;
    
    std::vector<Vector<double>> boundary_values (n_face_q_points, Vector<double> (dim));

    FullMatrix<double>
      local_matrix_b (dofs_per_cell, dofs_per_cell),
      local_rhs_k_b  (dofs_per_cell, dofs_per_cell),
      local_rhs_gp_b (dofs_per_cell, dofs_per_cell);
    
    FullMatrix<double>
      local_matrix_e (dofs_per_cell, dofs_per_cell),
      local_rhs_k_e  (dofs_per_cell, dofs_per_cell),
      local_rhs_cs_e (dofs_per_cell, dofs_per_cell),
      local_rhs_q_e  (dofs_per_cell, dofs_per_cell);

    //define local power
    Vector<double>    local_power    (dofs_per_cell);
    Tensor<1, dim>   power_rhs_H;
    
    const FEValuesExtractors::Vector B_field (0);
    const FEValuesExtractors::Vector E_field (dim);
    
    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active (),
      endc = dof_handler.end ();
    for (; cell != endc; ++cell)
      {
        fe_values.reinit (cell);
        
        local_matrix_b = 0;
        local_rhs_k_b  = 0;
        local_rhs_gp_b = 0;
    
        local_matrix_e = 0;
        local_rhs_k_e  = 0;
        local_rhs_cs_e = 0;
        local_rhs_q_e  = 0;
        
        local_power = 0;
        
        //        std::cout << "in cells..assembling.." << std::endl;
        
        for (unsigned int q = 0; q<n_q_points; ++q)
          for (unsigned int i = 0; i<dofs_per_cell; ++i)
            {
              const Tensor<1, dim> phi_i_B      = fe_values[B_field].value (i, q);
              const Tensor<1, dim> curl_phi_i_E = fe_values[E_field].curl  (i, q);
              const Tensor<1, dim> phi_i_E      = fe_values[E_field].value (i, q);
              
              
              for (unsigned int j = 0; j<dofs_per_cell; ++j)
                {
                  const Tensor<1, dim> phi_j_B      = fe_values[B_field].value (j, q);
                  const Tensor<1, dim> curl_phi_j_E = fe_values[E_field].curl  (j, q);
                  const Tensor<1, dim> phi_j_E      = fe_values[E_field].value (j, q);
                  
                  local_matrix_b (i, j) += phi_i_B * phi_j_B * fe_values.JxW (q)
                    * (1/mu_r + 1/mu_r/mu_r * sigma_m * time_step);
                  
                  local_rhs_k_b (i, j) += -1 * time_step * 1 / mu_r
                    * curl_phi_j_E * phi_i_B * fe_values.JxW (q);
                  
                  local_rhs_gp_b (i, j) += phi_i_B * phi_j_B * fe_values.JxW (q)
                    * (1/mu_r - 1/mu_r/mu_r*sigma_m*time_step);
                  
                  local_matrix_e (i, j) += phi_i_E * phi_j_E * fe_values.JxW (q)
                    * (eps_r + time_step/2 * sigma_e);
                  
                  local_rhs_k_e (i, j) += time_step * 1 / mu_r
                    * curl_phi_i_E * phi_j_B * fe_values.JxW (q);
                  
                  local_rhs_cs_e (i, j) += phi_i_E * phi_j_E * fe_values.JxW (q)
                    * (eps_r - time_step/2 * sigma_e);
                  
                  local_rhs_q_e (i, j)  += -1 * time_step
                    * phi_i_B * phi_j_E * fe_values.JxW (q);                  
                }
            }

        for (unsigned int face_n = 0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
          if (cell->at_boundary (face_n))
            {
              fe_face_values.reinit (cell, face_n);
              
              power_boundary_values.vector_value_list
                (fe_face_values.get_quadrature_points (), boundary_values);
              
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
          for (unsigned int j = 0; j<dofs_per_cell; ++j)
            {
              sys_matrix_b.add (local_dof_indices[i], local_dof_indices[j],
                                local_matrix_b (i, j));
              rhs_matrix_k_b.add (local_dof_indices[i], local_dof_indices[j],
                                  local_rhs_k_b (i, j));
              rhs_matrix_gp_b.add (local_dof_indices[i], local_dof_indices[j],
                                   local_rhs_gp_b (i, j));
              
              sys_matrix_e.add (local_dof_indices[i], local_dof_indices[j],
                                local_matrix_e (i, j));
              rhs_matrix_k_e.add (local_dof_indices[i], local_dof_indices[j],
                                  local_rhs_k_e (i, j));
              rhs_matrix_cs_e.add (local_dof_indices[i], local_dof_indices[j],
                                   local_rhs_cs_e (i, j));
              rhs_matrix_q_e.add (local_dof_indices[i], local_dof_indices[j],
                                  local_rhs_q_e (i, j));
            }
        
        for (unsigned int i = 0; i<dofs_per_cell; ++i)
          system_power (local_dof_indices[i]) += local_power (i);
      }            //end of cells loop
    
    std::cout << "end of assembling..ok!" << std::endl;
  }        //end of assemble_system

        

  // @sect4{WaveEquation::solve_u and WaveEquation::solve_v}

  // The next two functions deal with solving the linear systems associated
  // with the equations for $U^n$ and $V^n$. Both are not particularly
  // interesting as they pretty much follow the scheme used in all the
  // previous tutorial programs.
  //
  // One can make little experiments with preconditioners for the two matrices
  // we have to invert. As it turns out, however, for the matrices at hand
  // here, using Jacobi or SSOR preconditioners reduces the number of
  // iterations necessary to solve the linear system slightly, but due to the
  // cost of applying the preconditioner it is no win in terms of run-time. It
  // is not much of a loss either, but let's keep it simple and just do
  // without:
  template <int dim>
  void MaxwellTD<dim>::solve_b ()
  {
    SolverControl           solver_control (1000, 1e-12*system_rhs.l2_norm ());
    SolverCG<>              cg (solver_control);
    
    cg.solve (sys_matrix_b.block(0, 0), solution.block(0), system_rhs.block(0),
              PreconditionIdentity ());

    std::cout << "   u-equation: " << solver_control.last_step ()
              << " CG iterations."
              << std::endl;
  }

  

  template <int dim>
  void MaxwellTD<dim>::solve_e ()
  {
    SolverControl           solver_control (1000, 1e-12*system_rhs.l2_norm ());
    SolverCG<>              cg (solver_control);
    
    cg.solve (sys_matrix_e.block(1, 1), solution.block(1), system_rhs.block(1),
              PreconditionIdentity ());
    
    std::cout << "   v-equation: " << solver_control.last_step ()
              << " CG iterations."
              << std::endl;
  }
  


  // @sect4{WaveEquation::output_results}

  // Likewise, the following function is pretty much what we've done
  // before. The only thing worth mentioning is how here we generate a string
  // representation of the time step number padded with leading zeros to 3
  // character length using the Utilities::int_to_string function's second
  // argument.
  template <int dim>
  void MaxwellTD<dim>::output_results () const
  {
    DataOut<dim> data_out;

    std::vector<std::string> solution_names;
    switch (dim)
      {
      case 2:
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
        Assert (false, ExcNotImplemented ());
      }


    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, solution_names);

    data_out.build_patches ();

    std::ostringstream filename;
    filename << "solution-"
             <<    Utilities::int_to_string (timestep_number, 3)
             <<    ".vtk";
    
    std::ofstream output (filename.str ().c_str ());
    data_out.write_vtk (output);
  }

  

  // @sect4{WaveEquation::run}

  // The following is really the only interesting function of the program. It
  // contains the loop over all time steps, but before we get to that we have
  // to set up the grid, DoFHandler, and matrices. In addition, we have to
  // somehow get started with initial values. To this end, we use the
  // VectorTools::project function that takes an object that describes a
  // continuous function and computes the $L^2$ projection of this function
  // onto the finite element space described by the DoFHandler object. Can't
  // be any simpler than that:
  template <int dim>
  void MaxwellTD<dim>::run ()
  {
    setup_system ();
    
    std::cout << "setup system...ok! " << std::endl;
    
    assemble_system ();

    std::cout << "assembling...ok! " << std::endl;

    /*
      VectorTools::project (dof_handler, constraints, QGauss<dim> (3),
      InitialValuesU<dim> (),
      old_solution_u);
      VectorTools::project (dof_handler, constraints, QGauss<dim> (3),
      InitialValuesV<dim> (),
      old_solution_v);
    */
    
    // The next thing is to loop over all the time steps until we reach the
    // end time ($T = 5$ in this case). In each time step, we first have to
    // solve for $U^n$, using the equation $ (M^n + k^2\theta^2 A^n)U^n  = $
    // $ (M^{n, n-1} - k^2\theta (1-\theta) A^{n, n-1})U^{n-1} + kM^{n, n-1}V^{n-1}
    // +$ $k\theta \left[k \theta F^n + k (1-\theta) F^{n-1} \right]$. Note
    // that we use the same mesh for all time steps, so that $M^n = M^{n, n-1} = M$
    // and $A^n = A^{n, n-1} = A$. What we therefore have to do first is to add up
    // $MU^{n-1} - k^2\theta (1-\theta) AU^{n-1} + kMV^{n-1}$ and the forcing
    // terms, and put the result into the <code>system_rhs</code> vector. (For
    // these additions, we need a temporary vector that we declare before the
    // loop to avoid repeated memory allocations in each time step.)
    //
    Vector<double> tmp1 (system_rhs.block(0).size ());
    Vector<double> tmp2 (system_rhs.block(1).size ());
    //    Vector<double> forcing_terms (solution_u.size ());
    TimePulseFactor<dim> time_pulse_factor;
    Point<dim> p_time;
    
    for (timestep_number = 1, time = time_step;
         time <= 0.5;
         time += time_step, ++timestep_number)
      {
        std::cout << "Time step " << timestep_number
                  << " at t = " << time
                  << std::endl;
        
        std::cout << "____ " << old_solution.block(0).size () <<  std::endl
            << ";;;;;;" << rhs_matrix_k_b.block(0, 0).m () << std::endl;
        
        rhs_matrix_k_b.block(0, 1).vmult ( system_rhs.block(0), old_solution.block(1) );
        rhs_matrix_gp_b.block(0, 0).vmult ( tmp1, old_solution.block(0));
        system_rhs.block(0).add (1, tmp1);
        
        /*
          mt_G.vmult (system_rhs, old_solution_b);
          
          mt_P.vmult (tmp, old_solution_b);
          system_rhs.add (-time_step/2, tmp);
          
          mt_K.vmult (tmp, old_solution_e);
          system_rhs.add (-time_step, tmp);
        */
        
        
        // After so constructing the right hand side vector of the first
        // equation, all we have to do is apply the correct boundary
        // values. As for the right hand side, this is a space-time function
        // evaluated at a particular time, which we interpolate at boundary
        // nodes and then use the result to apply boundary values as we
        // usually do. The result is then handed off to the solve_u ()
        // function:
       
        /*
          {
          BoundaryValuesB<dim> boundary_values_b_function;
          boundary_values_b_function.set_time (time);
          
          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values (dof_handler,
                                                    0,
                                                    ZeroFunction<dim> (dim+dim),
                                                    boundary_values);

          // The matrix for solve_u () is the same in every time steps, so one
          // could think that it is enough to do this only once at the
          // beginning of the simulation. However, since we need to apply
          // boundary values to the linear system (which eliminate some matrix
          // rows and columns and give contributions to the right hand side),
          // we have to refill the matrix in every time steps before we
          // actually apply boundary data. The actual content is very simple:
          // it is the sum of the mass matrix and a weighted Laplace matrix:
//          matrix_b.copy_from (mt_G);
//          matrix_b.add (time_step/2.0, mt_P);
//
            
std::cout << "boundary B...OK " << std::endl;



          MatrixTools::apply_boundary_values (boundary_values,
                                              sys_matrix_b.block(0, 0),
                                              solution.block(0),
                                              system_rhs.block(0));
        }
        
*/        
        solve_b ();


        // The second step, i.e. solving for $V^n$, works similarly, except
        // that this time the matrix on the left is the mass matrix (which we
        // copy again in order to be able to apply boundary conditions, and
        // the right hand side is $MV^{n-1} - k\left[ \theta A U^n +
        // (1-\theta) AU^{n-1}\right]$ plus forcing terms. %Boundary values
        // are applied in the same way as before, except that now we have to
        // use the BoundaryValuesV class:

        /*
          mt_C.vmult (system_rhs, old_solution_e);
          
          mt_S.vmult (tmp, old_solution_e);
          system_rhs.add (-time_step/2, tmp);
          
          mt_Kt.vmult (tmp, solution_b);
        system_rhs.add (time_step, tmp);
        */
        
        time_pulse_factor.set_time (time);
        
        rhs_matrix_k_e.block(1, 0).vmult (system_rhs.block(1), solution.block(0));
        rhs_matrix_cs_e.block(1, 1).vmult (tmp2, old_solution.block(1));
        
        system_rhs.block(1).add (1, tmp2);
        system_rhs.block(1).add (time_pulse_factor.value (p_time, 0), system_power.block(1));
        
        //simply ignoring current J first
    
        
        /*

          {
          BoundaryValuesE<dim> boundary_values_e_function;
          boundary_values_e_function.set_time (time);
          
          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values (dof_handler,
          0,
          boundary_values_e_function,
          boundary_values);
          
          // matrix_e.copy_from (mt_C);
          // matrix_e.add (time_step/2.0, mt_S);

          MatrixTools::apply_boundary_values (boundary_values,
          sys_matrix_e.block(1, 1),
          solution.block(1),
          system_rhs.block(1));
          }
          
        */
        
        solve_e ();
        
        // Finally, after both solution components have been computed, we
        // output the result, compute the energy in the solution, and go on to
        // the next time step after shifting the present solution into the
        // vectors that hold the solution at the previous time step. Note the
        // function SparseMatrix::matrix_norm_square that can compute
        // $\left<V^n, MV^n\right>$ and $\left<U^n, AU^n\right>$ in one step,
        // saving us the expense of a temporary vector and several lines of
        // code:
        output_results ();
        /*
          std::cout << "   Total energy: "
          << (mass_matrix.matrix_norm_square (solution_v) +
          laplace_matrix.matrix_norm_square (solution_u)) / 2
                  << std::endl;
        */
        old_solution = solution;
      }
  }
}



// @sect3{The <code>main</code> function}

// What remains is the main function of the program. There is nothing here
// that hasn't been shown in several of the previous programs:
int main ()
{
  try
    {
      using namespace dealii;
      using namespace Maxwell;
      
      MaxwellTD<3> wave_equation_solver (0);
      wave_equation_solver.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what () << std::endl
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
