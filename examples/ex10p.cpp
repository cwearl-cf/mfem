//                       MFEM Example 10 - Parallel Version
//
// Compile with: make ex10p
//
// Sample runs:
//    mpirun -np 4 ex10p -m ../data/beam-quad.mesh -s 3 -rs 2 -dt 3
//    mpirun -np 4 ex10p -m ../data/beam-tri.mesh -s 3 -rs 2 -dt 3
//    mpirun -np 4 ex10p -m ../data/beam-hex.mesh -s 2 -rs 1 -dt 3
//    mpirun -np 4 ex10p -m ../data/beam-tet.mesh -s 2 -rs 1 -dt 3
//    mpirun -np 4 ex10p -m ../data/beam-wedge.mesh -s 2 -rs 1 -dt 3
//    mpirun -np 4 ex10p -m ../data/beam-quad.mesh -s 14 -rs 2 -dt 0.03 -vs 20
//    mpirun -np 4 ex10p -m ../data/beam-hex.mesh -s 14 -rs 1 -dt 0.05 -vs 20
//    mpirun -np 4 ex10p -m ../data/beam-quad-amr.mesh -s 3 -rs 2 -dt 3
//
// Description:  This examples solves a time dependent nonlinear elasticity
//               problem of the form dv/dt = H(x) + S v, dx/dt = v, where H is a
//               hyperelastic model and S is a viscosity operator of Laplacian
//               type. The geometry of the domain is assumed to be as follows:
//
//                                 +---------------------+
//                    boundary --->|                     |
//                    attribute 1  |                     |
//                    (fixed)      +---------------------+
//
//               The example demonstrates the use of nonlinear operators (the
//               class HyperelasticOperator defining H(x)), as well as their
//               implicit time integration using a Newton method for solving an
//               associated reduced backward-Euler type nonlinear equation
//               (class ReducedSystemOperator). Each Newton step requires the
//               inversion of a Jacobian matrix, which is done through a
//               (preconditioned) inner solver. Note that implementing the
//               method HyperelasticOperator::ImplicitSolve is the only
//               requirement for high-order implicit (SDIRK) time integration.
//
//               We recommend viewing examples 2 and 9 before viewing this
//               example.

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

void test_solution(ParGridFunction const & x, double const * const expected_values ) {
	int myid = Mpi::WorldRank();
	auto end = x.end();
	int i = 0;
	std::cout << "output: " << myid << " [ ";
	for( auto iter = x.begin(); iter != end; iter++, i++ ) {
        std::cout << *iter << ", ";
		if( std::abs( *iter - expected_values[i] ) > 1e-5 )
            std::cout << "In Rank " << myid << ": Difference in solution from expectation. Position: " << i << " acutal value: " << *iter << " expected value: " << expected_values[i] << std::endl;
		assert( std::abs( *iter - expected_values[i] ) < 1e-5 );
	}
	std::cout << "]";
	std::cout << std::endl;
};

void set_grev_pts( const mfem::KnotVector & kv, const int numIntervals, const int p, const double scaling, mfem::Array< double > & grev_pts )
{
	for( int i = 1; i < kv.Size() - p; i++ ) {
		double sum = 0.0;
		for ( int k = 0; k < p; k ++ ) {
			sum += kv[ i + k ];
		}
		grev_pts.Append( ( sum / ( p * numIntervals ) ) * scaling );
	}
};

mfem::Mesh build_mesh( const int poly_degree, const int x, const int y, int cont = -1, const double x_scaling = 1.0, const double y_scaling = 1.0 )
{
	using namespace mfem;

	if( cont == -1 )
		cont = poly_degree - 1;

	Array< double > x_intervals;
	Array< double > y_intervals;
	Array< int > x_continuity;
	Array< int > y_continuity;

	x_intervals.Append( 1 );
	x_continuity.Append( -1 );

	for( int i = 0; i < x - 1; i++ ) {
		x_intervals.Append( 1 );
		x_continuity.Append( cont );
	}
	x_continuity.Append( -1 );

	y_intervals.Append( 1 );
	y_continuity.Append( -1 );

	for( int i = 0; i < y - 1; i++ ) {
		y_intervals.Append( 1 );
		y_continuity.Append( cont );
	}
	y_continuity.Append( -1 );

	const KnotVector xkv( poly_degree, x_intervals, x_continuity );
	const KnotVector ykv( poly_degree, y_intervals, y_continuity );
	Array< NURBSPatch* > patches;

	Array< double > x_grev_pts;
	set_grev_pts( xkv, x, poly_degree, x_scaling, x_grev_pts );
	Array< double > y_grev_pts;
	set_grev_pts( ykv, y, poly_degree, y_scaling, y_grev_pts );

	Array< double > pts( 3 * xkv.GetNCP() * ykv.GetNCP() );
	int count = 0;
	for( int j = 0; j < ykv.GetNCP(); ++j )
		for( int i = 0; i < xkv.GetNCP(); ++i ) {
			pts[ count + 0 ] = x_grev_pts[ i ];
			pts[ count + 1 ] = y_grev_pts[ j ];
			pts[ count + 2 ] = 1;
			count += 3;
		}
	patches.Append( new NURBSPatch( &xkv, &ykv, 3, pts.GetData() ) );
	patches[ 0 ]->Print(std::cout);
	Mesh patch_topology = Mesh::MakeCartesian2D( 1, 1, Element::Type::QUADRILATERAL);
	NURBSExtension ne( &patch_topology, patches );
	return Mesh( ne );
};

void set_p_and_r( const int size, SparseMatrix & p, SparseMatrix & r )
{
	for( size_t i = 0; i < size; ++i )
	{
		p.Add( i, i, 1 );
		r.Add( i, i, 1 );
	}
	p.Finalize();
	r.Finalize();
};


class ReducedSystemOperator;

/** After spatial discretization, the hyperelastic model can be written as a
 *  system of ODEs:
 *     dv/dt = -M^{-1}*(H(x) + S*v)
 *     dx/dt = v,
 *  where x is the vector representing the deformation, v is the velocity field,
 *  M is the mass matrix, S is the viscosity matrix, and H(x) is the nonlinear
 *  hyperelastic operator.
 *
 *  Class HyperelasticOperator represents the right-hand side of the above
 *  system of ODEs. */
class HyperelasticOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   Array<int> ess_tdof_list;

   ParBilinearForm M, S;
   ParNonlinearForm H;
   double viscosity;
   HyperelasticModel *model;

   HypreParMatrix *Mmat; // Mass matrix from ParallelAssemble()
   CGSolver M_solver;    // Krylov solver for inverting the mass matrix M
   HypreSmoother M_prec; // Preconditioner for the mass matrix M

   /** Nonlinear operator defining the reduced backward Euler equation for the
       velocity. Used in the implementation of method ImplicitSolve. */
   ReducedSystemOperator *reduced_oper;

   /// Newton solver for the reduced backward Euler equation
   NewtonSolver newton_solver;

   /// Solver for the Jacobian solve in the Newton method
   Solver *J_solver;
   /// Preconditioner for the Jacobian solve in the Newton method
   Solver *J_prec;

   mutable Vector z; // auxiliary vector

public:
   HyperelasticOperator(ParFiniteElementSpace &f, Array<int> &ess_bdr,
                        double visc, double mu, double K);

   /// Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;
   /** Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   double ElasticEnergy(const ParGridFunction &x) const;
   double KineticEnergy(const ParGridFunction &v) const;
   void GetElasticEnergyDensity(const ParGridFunction &x,
                                ParGridFunction &w) const;

   virtual ~HyperelasticOperator();
};

/** Nonlinear operator of the form:
    k --> (M + dt*S)*k + H(x + dt*v + dt^2*k) + S*v,
    where M and S are given BilinearForms, H is a given NonlinearForm, v and x
    are given vectors, and dt is a scalar. */
class ReducedSystemOperator : public Operator
{
private:
   ParBilinearForm *M, *S;
   ParNonlinearForm *H;
   mutable HypreParMatrix *Jacobian;
   double dt;
   const Vector *v, *x;
   mutable Vector w, z;
   const Array<int> &ess_tdof_list;

public:
   ReducedSystemOperator(ParBilinearForm *M_, ParBilinearForm *S_,
                         ParNonlinearForm *H_, const Array<int> &ess_tdof_list);

   /// Set current dt, v, x values - needed to compute action and Jacobian.
   void SetParameters(double dt_, const Vector *v_, const Vector *x_);

   /// Compute y = H(x + dt (v + dt k)) + M k + S (v + dt k).
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
   virtual Operator &GetGradient(const Vector &k) const;

   virtual ~ReducedSystemOperator();
};


/** Function representing the elastic energy density for the given hyperelastic
    model+deformation. Used in HyperelasticOperator::GetElasticEnergyDensity. */
class ElasticEnergyCoefficient : public Coefficient
{
private:
   HyperelasticModel     &model;
   const ParGridFunction &x;
   DenseMatrix            J;

public:
   ElasticEnergyCoefficient(HyperelasticModel &m, const ParGridFunction &x_)
      : model(m), x(x_) { }
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   virtual ~ElasticEnergyCoefficient() { }
};

void InitialDeformation(const Vector &x, Vector &y);

void InitialVelocity(const Vector &x, Vector &v);

void visualize(ostream &os, ParMesh *mesh,
               ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name = NULL,
               bool init_vis = false);


int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   int order = 2;
   int ode_solver_type = 3;
   double t_final = 60.0;
   double dt = 3.0;
   double visc = 1e-2;
   double mu = 0.25;
   double K = 5.0;
   bool visualization = true;
   int vis_steps = 1;
   
   // 3. Read the serial mesh from the given mesh file on all processors. We can
   //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   Mesh mesh = build_mesh( 2, 8, 1, 1, 8.0, 1.0 );
   int dim = mesh.Dimension();
   
   // 4. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   ODESolver *ode_solver;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1:  ode_solver = new BackwardEulerSolver; break;
      case 2:  ode_solver = new SDIRK23Solver(2); break;
      case 3:  ode_solver = new SDIRK33Solver; break;
      // Explicit methods
      case 11: ode_solver = new ForwardEulerSolver; break;
      case 12: ode_solver = new RK2Solver(0.5); break; // midpoint method
      case 13: ode_solver = new RK3SSPSolver; break;
      case 14: ode_solver = new RK4Solver; break;
      case 15: ode_solver = new GeneralizedAlphaSolver(0.5); break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
         ode_solver = new SDIRK33Solver;
   }
   
   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   
   // 7. Define the parallel vector finite element spaces representing the mesh
   //    deformation x_gf, the velocity v_gf, and the initial configuration,
   //    x_ref. Define also the elastic energy density, w_gf, which is in a
   //    discontinuous higher-order space. Since x and v are integrated in time
   //    as a system, we group them together in block vector vx, on the unique
   //    parallel degrees of freedom, with offsets given by array true_offset.
   ParFiniteElementSpace * fespace = (ParFiniteElementSpace*)pmesh->GetNodes()->FESpace();
   
   SparseMatrix p( fespace->GetVSize(), fespace->GetVSize() );
   SparseMatrix r( fespace->GetVSize(), fespace->GetVSize() );
   set_p_and_r( fespace->GetVSize(), p, r );
   
   HYPRE_BigInt glob_size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of velocity/deformation unknowns: " << glob_size << endl;
   }
   int true_size = fespace->GetTrueVSize();
   int full_size = fespace->GetVSize();
   std::cout << "rs: " << myid << " " << true_size << " " << full_size << std::endl;
   Array<int> true_offset(3);
   true_offset[0] = 0;
   true_offset[1] = true_size;
   true_offset[2] = 2*true_size;
   
   BlockVector vx(true_offset);
   ParGridFunction v_gf, x_gf;
   v_gf.MakeTRef(fespace, vx, true_offset[0]);
   x_gf.MakeTRef(fespace, vx, true_offset[1]);
   
   ParGridFunction x_ref(fespace);
   pmesh->GetNodes(x_ref);
   
   L2_FECollection w_fec(order + 1, dim);
   ParFiniteElementSpace w_fespace(pmesh, &w_fec);
   ParGridFunction w_gf(&w_fespace);
   
   // 8. Set the initial conditions for v_gf, x_gf and vx, and define the
   //    boundary conditions on a beam-like mesh (see description above).
   //    TODO: fix indexing, currently local and global indexing are expected to be the same but in this test they are different
   Vector control_points;
   x_ref.GetTrueDofs( control_points );
   const int num_points = control_points.Size() / 2;
   std::cout << "My rank: " << myid << " num_points: " << num_points << std::endl;
   for(int i = 0; i < num_points; i++) {
   	int lx = i*2, ly = i*2+1;
   	double temp1[3], temp2[3];
   	Vector point( temp1, 2 );
   	Vector deform( temp2, 2 );
   	point[0] = control_points[lx];
   	point[1] = control_points[ly];
   	v_gf[lx] = 0;
   	v_gf[ly] = point[0] / 80.0;
   	InitialDeformation( point, deform );
   	x_gf[lx] = deform[0];
   	x_gf[ly] = deform[1];
   	std::cout << "R: " << myid << " " << i << " [" << point[0] << "," << point[1] << "] (" << deform[0] << "," << deform[1] << ") <" << 0.0 << "," << point[0] / 80.0 << ">" << std::endl;
   }
   
   v_gf.SetFromTrueVector(); x_gf.SetFromTrueVector();
   
   Array<int> ess_bdr(fespace->GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[3] = 1; // boundary attribute 4 (index 3) is fixed
   
   // 9. Initialize the hyperelastic operator, the GLVis visualization and print
   //    the initial energies.
   HyperelasticOperator oper(*fespace, ess_bdr, visc, mu, K);
   
   socketstream vis_v, vis_w;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_v.open(vishost, visport);
      vis_v.precision(8);
      visualize(vis_v, pmesh, &x_gf, &v_gf, "Velocity", true);
      // Make sure all ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      vis_w.open(vishost, visport);
      if (vis_w)
      {
         oper.GetElasticEnergyDensity(x_gf, w_gf);
         vis_w.precision(8);
         visualize(vis_w, pmesh, &x_gf, &w_gf, "Elastic energy density", true);
      }
      if (myid == 0)
      {
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }
   
   double ee0 = oper.ElasticEnergy(x_gf);
   double ke0 = oper.KineticEnergy(v_gf);
   //NonlinearForm H(fespace);
   //std::cout << "rw: " << myid << " " << H.Width() << "==" << x_gf.Size() << " / " << v_gf.Size() << std::endl;
   //const int le = H.GetEnergy(x_gf);
   //std::cout << "rr: " << myid << " " << le << std::endl;
   if (myid == 0)
   {
      cout << "initial elastic energy (EE) = " << ee0 << endl;
      cout << "initial kinetic energy (KE) = " << ke0 << endl;
      cout << "initial   total energy (TE) = " << (ee0 + ke0) << endl;
   }
   
   double t = 0.0;
   oper.SetTime(t);
   ode_solver->Init(oper);
   
   // 10. Perform time-integration
   //     (looping over the time iterations, ti, with a time-step dt).
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      double dt_real = min(dt, t_final - t);
   
      ode_solver->Step(vx, t, dt_real);
   
      last_step = (t >= t_final - 1e-8*dt);
   
      if (last_step || (ti % vis_steps) == 0)
      {
         v_gf.SetFromTrueVector(); x_gf.SetFromTrueVector();
   
         double ee = oper.ElasticEnergy(x_gf);
         double ke = oper.KineticEnergy(v_gf);
   
         if (myid == 0)
         {
            cout << "step " << ti << ", t = " << t << ", EE = " << ee
                 << ", KE = " << ke << ", ΔTE = " << (ee+ke)-(ee0+ke0) << endl;
         }
   
         if (visualization)
         {
            visualize(vis_v, pmesh, &x_gf, &v_gf);
            if (vis_w)
            {
               oper.GetElasticEnergyDensity(x_gf, w_gf);
               visualize(vis_w, pmesh, &x_gf, &w_gf);
            }
         }
      }
   }
   
   // 11. Save the displaced mesh, the velocity and elastic energy.
   {
      v_gf.SetFromTrueVector(); x_gf.SetFromTrueVector();
      // Note: uncomment out the lines below to save the results of this test
      //GridFunction *nodes = &x_gf;
      //int owns_nodes = 0;
      //pmesh->SwapNodes(nodes, owns_nodes);
   
      //ostringstream mesh_name, velo_name, ee_name;
      //mesh_name << "deformed." << setfill('0') << setw(6) << myid;
      //velo_name << "velocity." << setfill('0') << setw(6) << myid;
      //ee_name << "elastic_energy." << setfill('0') << setw(6) << myid;
   
      //ofstream mesh_ofs(mesh_name.str().c_str());
      //mesh_ofs.precision(8);
      //pmesh->Print(mesh_ofs);
      //pmesh->SwapNodes(nodes, owns_nodes);
      //ofstream velo_ofs(velo_name.str().c_str());
      //velo_ofs.precision(8);
      //v_gf.Save(velo_ofs);
      //ofstream ee_ofs(ee_name.str().c_str());
      //ee_ofs.precision(8);
      //oper.GetElasticEnergyDensity(x_gf, w_gf);
      //w_gf.Save(ee_ofs);
   }
   
   // 12. Free the used memory.
   delete ode_solver;
   delete pmesh;return 0;
}

void visualize(ostream &os, ParMesh *mesh,
               ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name, bool init_vis)
{
   if (!os)
   {
      return;
   }

   GridFunction *nodes = deformed_nodes;
   int owns_nodes = 0;

   mesh->SwapNodes(nodes, owns_nodes);

   os << "parallel " << mesh->GetNRanks()
      << " " << mesh->GetMyRank() << "\n";
   os << "solution\n" << *mesh << *field;

   mesh->SwapNodes(nodes, owns_nodes);

   if (init_vis)
   {
      os << "window_size 800 800\n";
      os << "window_title '" << field_name << "'\n";
      if (mesh->SpaceDimension() == 2)
      {
         os << "view 0 0\n"; // view from top
         os << "keys jl\n";  // turn off perspective and light
      }
      os << "keys cm\n";         // show colorbar and mesh
      // update value-range; keep mesh-extents fixed
      os << "autoscale value\n";
      os << "pause\n";
   }
   os << flush;
}


ReducedSystemOperator::ReducedSystemOperator(
   ParBilinearForm *M_, ParBilinearForm *S_, ParNonlinearForm *H_,
   const Array<int> &ess_tdof_list_)
   : Operator(M_->ParFESpace()->TrueVSize()), M(M_), S(S_), H(H_),
     Jacobian(NULL), dt(0.0), v(NULL), x(NULL), w(height), z(height),
     ess_tdof_list(ess_tdof_list_)
{ }

void ReducedSystemOperator::SetParameters(double dt_, const Vector *v_,
                                          const Vector *x_)
{
   dt = dt_;  v = v_;  x = x_;
}

void ReducedSystemOperator::Mult(const Vector &k, Vector &y) const
{
   // compute: y = H(x + dt*(v + dt*k)) + M*k + S*(v + dt*k)
   add(*v, dt, k, w);
   add(*x, dt, w, z);
   H->Mult(z, y);
   M->TrueAddMult(k, y);
   S->TrueAddMult(w, y);
   y.SetSubVector(ess_tdof_list, 0.0);
}

Operator &ReducedSystemOperator::GetGradient(const Vector &k) const
{
   delete Jacobian;
   SparseMatrix *localJ = Add(1.0, M->SpMat(), dt, S->SpMat());
   add(*v, dt, k, w);
   add(*x, dt, w, z);
   localJ->Add(dt*dt, H->GetLocalGradient(z));
   Jacobian = M->ParallelAssemble(localJ);
   delete localJ;
   HypreParMatrix *Je = Jacobian->EliminateRowsCols(ess_tdof_list);
   delete Je;
   return *Jacobian;
}

ReducedSystemOperator::~ReducedSystemOperator()
{
   delete Jacobian;
}


HyperelasticOperator::HyperelasticOperator(ParFiniteElementSpace &f,
                                           Array<int> &ess_bdr, double visc,
                                           double mu, double K)
   : TimeDependentOperator(2*f.TrueVSize(), 0.0), fespace(f),
     M(&fespace), S(&fespace), H(&fespace),
     viscosity(visc), M_solver(f.GetComm()), newton_solver(f.GetComm()),
     z(height/2)
{
   const double rel_tol = 1e-8;
   const int skip_zero_entries = 0;

   const double ref_density = 1.0; // density in the reference configuration
   ConstantCoefficient rho0(ref_density);
   M.AddDomainIntegrator(new VectorMassIntegrator(rho0));
   M.Assemble(skip_zero_entries);
   M.Finalize(skip_zero_entries);
   Mmat = M.ParallelAssemble();
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   HypreParMatrix *Me = Mmat->EliminateRowsCols(ess_tdof_list);
   delete Me;

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(30);
   M_solver.SetPrintLevel(0);
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(*Mmat);

   model = new NeoHookeanModel(mu, K);
   H.AddDomainIntegrator(new HyperelasticNLFIntegrator(model));
   H.SetEssentialTrueDofs(ess_tdof_list);

   ConstantCoefficient visc_coeff(viscosity);
   S.AddDomainIntegrator(new VectorDiffusionIntegrator(visc_coeff));
   S.Assemble(skip_zero_entries);
   S.Finalize(skip_zero_entries);

   reduced_oper = new ReducedSystemOperator(&M, &S, &H, ess_tdof_list);

   HypreSmoother *J_hypreSmoother = new HypreSmoother;
   J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
   J_hypreSmoother->SetPositiveDiagonal(true);
   J_prec = J_hypreSmoother;

   MINRESSolver *J_minres = new MINRESSolver(f.GetComm());
   J_minres->SetRelTol(rel_tol);
   J_minres->SetAbsTol(0.0);
   J_minres->SetMaxIter(300);
   J_minres->SetPrintLevel(-1);
   J_minres->SetPreconditioner(*J_prec);
   J_solver = J_minres;

   newton_solver.iterative_mode = false;
   newton_solver.SetSolver(*J_solver);
   newton_solver.SetOperator(*reduced_oper);
   newton_solver.SetPrintLevel(1); // print Newton iterations
   newton_solver.SetRelTol(rel_tol);
   newton_solver.SetAbsTol(0.0);
   newton_solver.SetAdaptiveLinRtol(2, 0.5, 0.9);
   newton_solver.SetMaxIter(10);
}

void HyperelasticOperator::Mult(const Vector &vx, Vector &dvx_dt) const
{
   // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
   int sc = height/2;
   Vector v(vx.GetData() +  0, sc);
   Vector x(vx.GetData() + sc, sc);
   Vector dv_dt(dvx_dt.GetData() +  0, sc);
   Vector dx_dt(dvx_dt.GetData() + sc, sc);

   H.Mult(x, z);
   if (viscosity != 0.0)
   {
      S.TrueAddMult(v, z);
      z.SetSubVector(ess_tdof_list, 0.0);
   }
   z.Neg(); // z = -z
   M_solver.Mult(z, dv_dt);

   dx_dt = v;
}

void HyperelasticOperator::ImplicitSolve(const double dt,
                                         const Vector &vx, Vector &dvx_dt)
{
   int sc = height/2;
   Vector v(vx.GetData() +  0, sc);
   Vector x(vx.GetData() + sc, sc);
   Vector dv_dt(dvx_dt.GetData() +  0, sc);
   Vector dx_dt(dvx_dt.GetData() + sc, sc);

   // By eliminating kx from the coupled system:
   //    kv = -M^{-1}*[H(x + dt*kx) + S*(v + dt*kv)]
   //    kx = v + dt*kv
   // we reduce it to a nonlinear equation for kv, represented by the
   // reduced_oper. This equation is solved with the newton_solver
   // object (using J_solver and J_prec internally).
   reduced_oper->SetParameters(dt, &v, &x);
   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
   newton_solver.Mult(zero, dv_dt);
   MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge.");
   add(v, dt, dv_dt, dx_dt);
}

double HyperelasticOperator::ElasticEnergy(const ParGridFunction &x) const
{
   return H.GetEnergy(x);
}

double HyperelasticOperator::KineticEnergy(const ParGridFunction &v) const
{
   double loc_energy = 0.5*M.InnerProduct(v, v);
   double energy;
   MPI_Allreduce(&loc_energy, &energy, 1, MPI_DOUBLE, MPI_SUM,
                 fespace.GetComm());
   return energy;
}

void HyperelasticOperator::GetElasticEnergyDensity(
   const ParGridFunction &x, ParGridFunction &w) const
{
   ElasticEnergyCoefficient w_coeff(*model, x);
   w.ProjectCoefficient(w_coeff);
}

HyperelasticOperator::~HyperelasticOperator()
{
   delete J_solver;
   delete J_prec;
   delete reduced_oper;
   delete model;
   delete Mmat;
}


double ElasticEnergyCoefficient::Eval(ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   model.SetTransformation(T);
   x.GetVectorGradient(T, J);
   // return model.EvalW(J);  // in reference configuration
   return model.EvalW(J)/J.Det(); // in deformed configuration
}


void InitialDeformation(const Vector &x, Vector &y)
{
   // set the initial configuration to be the same as the reference, stress
   // free, configuration
   y = x;
}

void InitialVelocity(const Vector &x, Vector &v)
{
   const int dim = x.Size();
   const double s = 0.1/64.;

   v = 0.0;
   v(dim-1) = s*x(0)*x(0)*(8.0-x(0));
   v(0) = -s*x(0)*x(0);
}
