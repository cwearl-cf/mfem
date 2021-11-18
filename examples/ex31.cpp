//                                Solution of optimal design problem
//
// Compile with: make ex31
//
// Sample runs:
//    ex31 -r 3
//    ex31 -m ../data/star.mesh -r 3 -mf 0.5 -tol 1e-3 -o 3
//
// Description:  This examples solves the following PDE-constrained
//               optimization problem for the diffusion coefficient K:
//
//         min J(K) = (f,u)
//
//         subject to   - div( K\nabla u ) = f    in \Omega
//                                       u = 0    on \partial\Omega
//         and            \int_\Omega K dx = V
//         and            K_min <= K(x) <= K_max
//

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

/** The Lagrangian for this problem is
 *
 *    L(u,K,p,\lambda) = (f,u) - (K \nabla u, \nabla p) + (f,p) - ( V - (K,1) ) \lambda
 *
 *      u, p \in H^1_0
 *      K \in L^1 \cap L^\infty
 *      \lambda \in \mathbb{R}
 *
 *  Note that
 *
 *    \partial_p L = 0        (1)
 *
 *  delivers the state equation
 *
 *    (\nabla u, \nabla v) = (f,v)  for all v in H^1_0
 *
 *  and
 *
 *    \partial_u L = 0        (2)
 *
 *  delivers the adjoint equation (same as the state eqn)
 *
 *    (\nabla p, \nabla v) = (f,v)  for all v in H^1_0
 *
 *  and at the solutions u=p of (1) and (2), respectively,
 *
 *  D_K J = D_K L = \partial_u L \partial_K u + \partial_p L \partial_K p
 *                 + \partial_\lambda L \partial_K \lambda + \partial_K L
 *                = \partial_K L
 *                = (\lambda - |\nabla u|^2, \cdot)
 *
 * Likewise,
 *
 *    D_K J = (K,1) - V
 *
 * We update the control f_k with the projected gradient method
 *
 *    K_{k+1} = P( K_k - \gamma (\lambda_k - |\nabla u_k|^2) )
 *    \lambda_{k+1} = \lambda_k + \gamma ((K_k,1) - V)
 *
 * where P is the projection operator enforcing K_min <= K(x) <= K_max
 * and \gamma is a specified step length.
 *
 */

class DykstraProjection
{
protected:
   int max_its = 10;
   double domain_volume;
   double volume_fraction = 0.5;
   double K_max = 1.0;
   double K_min = 1e-3;
   GridFunction *p;
   GridFunction *q;
   LinearForm *vol_form;

   void ApplyBoundConstraint(GridFunction &K)
   {
      for (int i = 0; i < K.Size(); i++)
      {
         if (K[i] > K_max)
         {
            K[i] = K_max;
         }
         else if (K[i] < K_min)
         {
            K[i] = K_min;
         }
         else
         {
            // do nothing
         }
      }
   }

public:
   DykstraProjection(LinearForm &vol_form_, FiniteElementSpace &control_fes)
   {
      vol_form = &vol_form_;
      GridFunction onegf(&control_fes);
      onegf = 1.0;
      domain_volume = (*vol_form)(onegf);
      p = new GridFunction(&control_fes);
      q = new GridFunction(&control_fes);
   }

   void SetVolumeFraction(double volume_fraction_)
   {
      volume_fraction = volume_fraction_;
   }

   void SetKBounds(double K_min_, double K_max_)
   {
      K_min = K_min_;
      K_max = K_max_;
   }

   double Eval(GridFunction &K)
   {
      // Project onto intersection of two constraint sets.
      // Uses Dykstra's projection algorithm.
      *p = 0.0;
      *q = 0.0;
      double volume = (*vol_form)(K);
      double misfit = ( volume / domain_volume ) - volume_fraction;
      for (int k = 0; k < max_its; k++)
      {
         // STEP 1: Project K+p onto { K : \int K <= volume_fraction * vol }.
         *p += K;
         if (misfit > 1e-1)
         {
            K = *p;
            K -= misfit;
         }
         else if (k > 0)
         {
            break;
         }

         // STEP 2: Update p.
         *p -= K;

         // STEP 3: Project K+q onto { K : K(x) \in [K_min,K_max] }.
         *q += K;
         K = *q;
         ApplyBoundConstraint(K);

         // STEP 4: Update q.
         *q -= K;

         volume = (*vol_form)(K);
         misfit = ( volume / domain_volume ) - volume_fraction;
      }
      return misfit;
   }
};

double load(const Vector & x)
{
   double x1 = x(0);
   double x2 = x(1);
   double r = sqrt(x1*x1 + x2*x2);
   if (r <= 0.5)
   {
      return 1.0;
   }
   else
   {
      return 0.0;
   }
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int ref_levels = 2;
   int order = 2;
   bool visualization = true;
   double step_length_K = 1.0;
   double step_length_lambda = 1.0;
   double volume_fraction = 0.5;
   int max_it = 1e3;
   double tol = 1e-2;
   double K_max = 1.0;
   double K_min = 1e-3;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&step_length_K, "-sl1", "--step-length-K",
                  "Step length for gradient descent (K).");
   args.AddOption(&step_length_lambda, "-sl2", "--step-length-lambda",
                  "Step length for gradient ascent (lambda).");
   args.AddOption(&tol, "-tol", "--tolerance",
                  "Convergence tolerance.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&volume_fraction, "-vf", "--volume-fraction",
                  "Volume fraction for diffusion coefficient.");
   args.AddOption(&K_max, "-Kmax", "--K-max",
                  "Maximum of diffusion diffusion coefficient.");
   args.AddOption(&K_min, "-Kmin", "--K-min",
                  "Minimum of diffusion diffusion coefficient.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 3. Define the load f.
   FunctionCoefficient f(load);

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 5. Define the vector finite element spaces representing the state variable u,
   //    adjoint variable p, and the control variable f.
   H1_FECollection state_fec(order, dim);
   L2_FECollection control_fec(order-1, dim, BasisType::Positive);
   FiniteElementSpace state_fes(&mesh, &state_fec);
   FiniteElementSpace control_fes(&mesh, &control_fec);

   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   cout << "Number of state unknowns: " << state_size << endl;
   cout << "Number of control unknowns: " << control_size << endl;

   // 6. All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_tdof_list;
   state_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Set the initial guess for K and the boundary conditions for u.
   GridFunction u(&state_fes);
   GridFunction K(&control_fes);
   GridFunction K_old(&control_fes);
   u = 0.0;
   K = 1.0;
   K_old = K;

   // 8. Set up the linear form b(.) for the state equation.
   LinearForm b(&state_fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(f));
   b.Assemble();
   OperatorPtr A;
   Vector B, X;

   // 9. Define the gradient function
   GridFunction grad(&control_fes);
   grad = 0.0;

   // 10. Define some tools for later
   ConstantCoefficient zero(0.0);
   ConstantCoefficient one(1.0);
   GridFunction onegf(&control_fes);
   onegf = 1.0;
   LinearForm vol_form(&control_fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
   double domain_volume = vol_form(onegf);

   // 11. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_u,sout_p,sout_K;
   if (visualization)
   {
      sout_u.open(vishost, visport);
      sout_K.open(vishost, visport);
      sout_u.precision(8);
      sout_K.precision(8);
   }

   // 12. Project initial K onto constraint set.
   DykstraProjection Projector(vol_form, control_fes);
   Projector.SetVolumeFraction(volume_fraction);
   Projector.SetKBounds(K_min,K_max);
   Projector.Eval(K);

   // 13. Optimize.
   double lambda = 0.0;
   for (int k = 1; k <= max_it; k++)
   {
      // A. Compute volume
      double volume = vol_form(K);

      // B. Form state equation
      BilinearForm a(&state_fes);
      GridFunctionCoefficient diffusion_coeff(&K);
      a.AddDomainIntegrator(new DiffusionIntegrator(diffusion_coeff));
      a.Assemble();
      a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

      // C. Solve state equation
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 0, 800, 1e-12, 0.0);

      // D. Recover state variable
      a.RecoverFEMSolution(X, b, u);
      double compliance = b(u);

      // E. Use relative tolerance
      if (k == 1)
      {
         tol *= compliance + 1e-6;
      }

      // F. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         sout_u << "solution\n" << mesh << u
                << "window_title 'State u'" << flush;
      }

      // G. Constuct gradient function (i.e., lambda - |\nabla u|^2)
      GradientGridFunctionCoefficient grad_u(&u);
      InnerProductCoefficient norm2_grad_u(grad_u,grad_u);
      grad.ProjectCoefficient(norm2_grad_u);
      grad *= -1.0;
      grad += lambda;

      // H. Update K.
      for (int i = 0; i < K.Size(); i++)
      {
         // K[i] -= step_length_K*grad[i]/log(k+1);
         // K[i] -= step_length_K*grad[i]/sqrt(k);
         // K[i] -= step_length_K*grad[i];
         // K[i] *= exp(-step_length_K*grad[i]);
         K[i] = log(K[i]/(1.0 - K[i]));
         K[i] -= step_length_K*grad[i];
         K[i] = 1.0/(1.0 + exp(-K[i]));
      }

      // I. Project K onto constraint set.
      double grad_lambda = Projector.Eval(K);
      cout << "grad_lambda = " << grad_lambda << endl;

      // J. Update lambda.
      lambda += step_length_lambda*grad_lambda;
      lambda = max(0.0, lambda);

      // K. Compute norm of update.
      K_old -= K;
      // double norm = K_old.ComputeL2Error(zero)/step_length_K*log(k+1);
      // double norm = K_old.ComputeL2Error(zero)/step_length_K*sqrt(k);
      double norm = K_old.ComputeL2Error(zero)/step_length_K;
      norm = sqrt(norm*norm + grad_lambda*grad_lambda);
      K_old = K;

      // L. Print and exit if norm of grad is small enough.
      mfem::out << "norm of reduced gradient = " << norm << endl;
      mfem::out << "compliance = " << compliance << endl;
      mfem::out << "volume_fraction = " << volume / domain_volume << endl;
      mfem::out << "lambda = " << lambda << endl;
      if (norm < tol)
      {
         break;
      }

      if (visualization)
      {
         sout_K << "solution\n" << mesh << K
                << "window_title 'Control K'" << flush;
      }

   }

   return 0;
}