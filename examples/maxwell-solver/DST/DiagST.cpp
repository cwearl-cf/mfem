//Diagonal Source Transfer Preconditioner

#include "DiagST.hpp"

DiagST::DiagST(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
         double omega_, Coefficient * ws_,  int nrlayers_)
   : Solver(2*bf_->FESpace()->GetTrueVSize(), 2*bf_->FESpace()->GetTrueVSize()), 
     bf(bf_), Pmllength(Pmllength_), omega(omega_), ws(ws_), nrlayers(nrlayers_)
{
   Mesh * mesh = bf->FESpace()->GetMesh();
   dim = mesh->Dimension();

   // ----------------- Step 1 --------------------
   // Introduce 2 layered partitios of the domain 
   // 
   int partition_kind;

   // 1. Ovelapping partition with overlap = 2h 
   partition_kind = 2; // Non Overlapping partition 
   int nx=4;
   int ny=4; 
   int nz=1;
   povlp = new MeshPartition(mesh, partition_kind,nx,ny,nz);
   nxyz[0] = povlp->nxyz[0];
   nxyz[1] = povlp->nxyz[1];
   nxyz[2] = povlp->nxyz[2];
   nrpatch = povlp->nrpatch;
   // cout<< "nrpatch = " << nrpatch << endl;
   // cout << "nx = " << nx << endl;
   // cout << "ny = " << ny << endl;
   // cout << "nz = " << nz << endl;
   subdomains = povlp->subdomains;
   // for (int k = 0; k<nxyz[2]; k++)
   // {
   //    for (int j = 0; j<nxyz[1]; j++)
   //    {
   //       for (int i = 0; i<nxyz[0]; i++)
   //       {
   //          Array<int> ijk(3);
   //          ijk[0]=i;
   //          ijk[1]=j;
   //          ijk[2]=k;
   //          // cout << "("<<i<<","<<j<<","<<k<<") = " << povlp->subdomains(i,j,k) << endl;
   //          cout << "("<<i<<","<<j<<","<<k<<") = " << GetPatchId(ijk) << endl;
   //       }
   //    }   
   // }

   // for (int ip = 0; ip<nrpatch; ip++)
   // {
   //    int i, j, k;
   //    Getijk(ip, i,j,k);
   //    cout << "ip = " << ip << ": ("<<i<<","<<j<<","<<k<<")"<< endl;
   // }

   

   //
   // ----------------- Step 1a -------------------
   // Save the partition for visualization
   // SaveMeshPartition(povlp->patch_mesh, "output/mesh_ovlp.", "output/sol_ovlp.");

   // // // ------------------Step 2 --------------------
   // // // Construct the dof maps from subdomains to global (for the extended and not)
   ovlp_prob  = new DofMap(bf,povlp,nrlayers); 

   // ------------------Step 3 --------------------
   // Assemble the PML Problem matrices and factor them
   PmlMat.SetSize(nrpatch);
   PmlMatInv.SetSize(nrpatch);
   for (int ip=0; ip<nrpatch; ip++)
   {
      PmlMat[ip] = GetPmlSystemMatrix(ip);
      PmlMatInv[ip] = new KLUSolver;
      PmlMatInv[ip]->SetOperator(*PmlMat[ip]);
   }

   // Set up src arrays size
   f_orig.SetSize(nrpatch);
   f_transf.SetSize(nrpatch);


   // Construct a simple map used for directions of transfer
   ConstructDirectionsMap();
   for (int ip=0; ip<nrpatch; ip++)
   {
      int n = 2*ovlp_prob->fespaces[ip]->GetTrueVSize(); // (x 2 for complex ) 
      f_orig[ip] = new Vector(n); *f_orig[ip] = 0.0;
      f_transf[ip].SetSize(ntransf_directions);
      for (int i=0;i<ntransf_directions; i++)
      {
         f_transf[ip][i] = new Vector(n); *f_transf[ip][i] = 0.0;
      }
   }
}

SparseMatrix * DiagST::GetPmlSystemMatrix(int ip)
{
   double h = GetUniformMeshElementSize(ovlp_prob->PmlMeshes[ip]);
   Array2D<double> length(dim,2);
   length = h*(nrlayers);

   CartesianPML pml(ovlp_prob->PmlMeshes[ip], length);
   pml.SetOmega(omega);

   Array <int> ess_tdof_list;
   if (ovlp_prob->PmlMeshes[ip]->bdr_attributes.Size())
   {
      Array<int> ess_bdr(ovlp_prob->PmlMeshes[ip]->bdr_attributes.Max());
      ess_bdr = 1;
      ovlp_prob->PmlFespaces[ip]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ConstantCoefficient one(1.0);
   ConstantCoefficient sigma(-pow(omega, 2));

   PmlMatrixCoefficient c1_re(dim,pml_detJ_JT_J_inv_Re,&pml);
   PmlMatrixCoefficient c1_im(dim,pml_detJ_JT_J_inv_Im,&pml);

   PmlCoefficient detJ_re(pml_detJ_Re,&pml);
   PmlCoefficient detJ_im(pml_detJ_Im,&pml);

   ProductCoefficient c2_re0(sigma, detJ_re);
   ProductCoefficient c2_im0(sigma, detJ_im);

   ProductCoefficient c2_re(c2_re0, *ws);
   ProductCoefficient c2_im(c2_im0, *ws);

   SesquilinearForm a(ovlp_prob->PmlFespaces[ip],ComplexOperator::HERMITIAN);

   a.AddDomainIntegrator(new DiffusionIntegrator(c1_re),
                         new DiffusionIntegrator(c1_im));
   a.AddDomainIntegrator(new MassIntegrator(c2_re),
                         new MassIntegrator(c2_im));
   a.Assemble();

   OperatorPtr Alocal;
   a.FormSystemMatrix(ess_tdof_list,Alocal);
   ComplexSparseMatrix * AZ_ext = Alocal.As<ComplexSparseMatrix>();
   SparseMatrix * Mat = AZ_ext->GetSystemMatrix();
   Mat->Threshold(0.0);
   return Mat;
}


void DiagST::Mult(const Vector &r, Vector &z) const
{
   // Step 0
   // Restrict original sources to the patches
   for (int ip=0; ip<nrpatch; ip++)
   {
      *f_orig[ip] = 0.0;
      for (int i=0;i<ntransf_directions; i++)
      {
         *f_transf[ip][i] = 0.0;
      }
   }
   for (int ip=0; ip<nrpatch; ip++)
   {
      Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
      r.GetSubVector(*Dof2GlobalDof,*f_orig[ip]);
   }

   char vishost[] = "localhost";
   int  visport   = 19916;
   z = 0.0; 
   Vector rnew(r);
   Vector znew(z);
   znew = 0.0;
   
   // in 2D there are a total of 4 sweeps
   // with nx + ny - 1 serial steps each
   // --------------------------------------------
   //       Sweep in the direction (1,1)
   // --------------------------------------------
   int nx = nxyz[0];
   int ny = nxyz[1];

   int nsteps = nx + ny - 1;
   // loop through the steps
   for (int s = 0; s<nsteps; s++)
   {
      // the patches involved are the ones such that
      // i+j = s
      cout << "Step no: " << s << endl;
      for (int i=0;i<nx; i++)
      {
         int j = s-i;
         if (j<0 || j>=ny) continue;
         cout << "Patch no: (" << i <<"," << j << ")" << endl; 

         // find patch id
         Array<int> ij(2); ij[0] = i; ij[1]=j;
         int ip = GetPatchId(ij);
         cout << "ip = " << ip << endl;

         // Solve the PML problem in patch ip with all sources
         // Original and all transfered (maybe some of them)
         Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
         Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
         int ndofs = Dof2GlobalDof->Size();
         Vector sol_local(ndofs);
         Vector res_local(ndofs);
         res_local = *f_orig[ip];
         for (int nd=0; nd<ntransf_directions; nd++)
         {
            res_local += *f_transf[ip][nd];
         }
         // Extend by zero to the PML mesh
         int nrdof_ext = PmlMat[ip]->Height();
      
         Vector res_ext(nrdof_ext); res_ext = 0.0;
         Vector sol_ext(nrdof_ext); sol_ext = 0.0;

         res_ext.SetSubVector(*Dof2PmlDof,res_local);
         PmlMatInv[ip]->Mult(res_ext, sol_ext);

         // Multiply with the cutoff functions, find the new sources and 
         // and propagate to all neighboring subdomains 
         // (possible 8 in 2D, 26 in 3D)
         TransferSources(ip, sol_ext);
         Vector cfsol_ext(sol_ext.Size());
         // cut off the ip solution to all possible directions


         Array<int>directions(2); directions = 0; 
         if (i+1<nx) directions[0] = 1;
         if (j+1<ny) directions[1] = 1;
         GetCutOffSolution(sol_ext,cfsol_ext,ip,directions,true);
         // directions = 0;
         // if (i-1>=0) directions[0] = -1;
         // if (j-1>=0) directions[1] = -1;
         // sol_ext = cfsol_ext;
         // GetCutOffSolution(sol_ext,cfsol_ext,ip,directions,true);

         cfsol_ext.GetSubVector(*Dof2PmlDof, sol_local);
         znew = 0.0;
         znew.SetSubVector(*Dof2GlobalDof, sol_local);
         z+=znew;
        

      }
      socketstream zsock(vishost, visport);
      PlotSolution(z,zsock,0);
      cin.get();
   }
   




   // // int ip = 4;

   // cout << "Testing patch ip = " << ip << endl;

   // // Set source
   // Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
   // Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
   // int ndofs = Dof2GlobalDof->Size();
   // Vector sol_local;
   // sol_local.SetSize(ndofs);
   // rnew.GetSubVector(*Dof2GlobalDof, *f_orig[ip]);

   // // Extend by zero to the PML mesh
   // int nrdof_ext = PmlMat[ip]->Height();
      
   // Vector res_ext(nrdof_ext); res_ext = 0.0;
   // Vector sol_ext(nrdof_ext); sol_ext = 0.0;

   // res_ext.SetSubVector(*Dof2PmlDof,f_orig[ip]->GetData());
   // PmlMatInv[ip]->Mult(res_ext, sol_ext);


   // sol_ext.GetSubVector(*Dof2PmlDof, sol_local);
   // Array<int>directions(2);
   // directions[0]=-1;
   // directions[1]=0;
   // GetCutOffSolution(sol_ext,ip,directions, true);

   // Vector res_local;
   // res_local.SetSize(ndofs);
   // PmlMat[ip]->Mult(sol_ext,res_ext); res_ext*=-1.0;
   // res_ext.GetSubVector(*Dof2PmlDof, res_local);

   // sol_local = 0.0;
   // sol_ext.GetSubVector(*Dof2PmlDof, sol_local);
   // socketstream znewsol_sock(vishost, visport);
   // znew.SetSubVector(*Dof2GlobalDof,sol_local);
   // PlotSolution(znew,znewsol_sock,0);

   // Vector Psi1;
   // int ip1 = SourceTransfer(res_local,directions,ip,Psi1);
   // // cout << "ip = " << ip << endl;
   // // Array<int> * Dof2GlobalDof1 = &ovlp_prob->Dof2GlobalDof[ip1];

   // // socketstream znewsol_sock1(vishost, visport);
   // // znew = 0.0;
   // // znew.SetSubVector(*Dof2GlobalDof1,Psi1);
   // // PlotSolution(znew,znewsol_sock1,0);




   // // ---------------------------------------------------
   // Array<int> *Dof2GlobalDof1 = &ovlp_prob->Dof2GlobalDof[ip1];
   // Array<int> *Dof2PmlDof1 = &ovlp_prob->Dof2PmlDof[ip1];
   // ndofs = Dof2GlobalDof1->Size();
   // sol_local.SetSize(ndofs); sol_local=0.0;

   // // Extend by zero to the PML mesh
   // nrdof_ext = PmlMat[ip1]->Height();
      
   // res_ext.SetSize(nrdof_ext); res_ext = 0.0;
   // sol_ext.SetSize(nrdof_ext); sol_ext = 0.0;

   // res_ext.SetSubVector(*Dof2PmlDof1,Psi1.GetData());
   // PmlMatInv[ip1]->Mult(res_ext, sol_ext);

   // sol_ext.GetSubVector(*Dof2PmlDof1, sol_local);

   // socketstream znew_sock(vishost, visport);
   // znew2.SetSubVector(*Dof2GlobalDof1,sol_local);
   // // znew+=znew2;
   // PlotSolution(znew2,znew_sock,0);
   // // PlotSolution(znew2,znew_sock,0);

   // z=znew;
   // z+=znew2;

   // socketstream z_sock(vishost, visport);
   // // znew+=znew2;
   // PlotSolution(z,z_sock,0);
  






   // socketstream pmlsol_sock(vishost, visport);
   // FiniteElementSpace * fespace = ovlp_prob->PmlFespaces[ip];
   // Mesh * mesh = fespace->GetMesh();
   // GridFunction gf(fespace);
   // double * data = sol_ext.GetData();
   // gf.SetData(data);
   
   // string keys = "keys mrRljc\n";
   // pmlsol_sock << "solution\n" << *mesh << gf << keys << flush;

   // socketstream sol_sock(vishost, visport);
   // FiniteElementSpace * fespacel = ovlp_prob->fespaces[ip];
   // Mesh * meshl = fespacel->GetMesh();
   // GridFunction gfl(fespacel);
   // double * datal = sol_local.GetData();
   // gfl.SetData(datal);
   
   // sol_sock << "solution\n" << *meshl << gfl << keys << flush;


   // // forward source transfer algorithm




   // // forward source transfer algorithm
   // socketstream znewsol_sock1(vishost, visport);
   // znew=0.0;
   // znew.SetSubVector(*Dof2GlobalDof,sol_local);
   // PlotSolution(znew,znewsol_sock1,0);

   //       GetCutOffSolution(sol_ext, ip, direction, true);
   //       // find the residual
   //       // Vector respml(sol_ext.Size());
   //       // PmlMat[ip]->Mult(sol_ext,respml);

   //       // restrict to non-pml problem
   //       // Vector res(ndofs); 
   //       // respml.GetSubVector(*Dof2PmlDof, res);
   //       // source to be transfered
   //       // res.GetSubVector(lmap->map2[ip],ftransf[ip+1]);

   //    }
   //    //-----------------------------------------------
   //    //-----------------------------------------------



   //    sol_ext.GetSubVector(*Dof2PmlDof,sol_local);
   //    znew = 0.0;
   //    znew.SetSubVector(*Dof2GlobalDof,sol_local);

   //    Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip+1];
   //    fsol[ip+1].SetSize(nDof2GlobalDof->Size());
   //    znew.GetSubVector(*nDof2GlobalDof,fsol[ip+1]);



   //    // Find residual;
   //    Vector respml(sol_ext.Size());
   //    Vector reslocal(sol_local.Size());
   //    PmlMat[ip]->Mult(sol_ext,respml);
   //    respml *= -1.0;

   //    respml.GetSubVector(*Dof2PmlDof,reslocal);
   //    // source to be transfered
   //    // reslocal.GetSubVector(lmap->map2[ip],ftransf[ip+1]);
   //    // ftransf[ip+1]+= fn[ip+1];
   //    rnew.AddElementVector(*Dof2GlobalDof,reslocal);

   //    // Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip+1];
   //    // fsol[ip+1].SetSize(nDof2GlobalDof->Size());
   //    // znew.GetSubVector(*nDof2GlobalDof,fsol[ip+1]);

   //    // if (ip != nrpatch-1) 
   //    // {
   //    //    int direction = 1;
   //    //    GetCutOffSolution(znew, ip, direction);
   //    //    // find the residual
   //    // }
      
   //    // socketstream sockznew(vishost, visport);
   //    // PlotSolution(znew,sockznew,0); cin.get();
   //    z1+=znew;

      // A->Mult(znew, raux);
      // rnew -= raux;
   // }

   // backward source transfer algorithm
   // for (int ip=0; ip<=nrpatch; ip++)
   // {
   //    Array<int> *Dof2GDof = &novlp_prob->Dof2GlobalDof[ip];
   //    fn[ip].SetSize(Dof2GDof->Size()); fn[ip] = 0.0;
   //    ftransf[ip].SetSize(Dof2GDof->Size()); ftransf[ip] = 0.0;
   //    r.GetSubVector(*Dof2GDof,fn[ip]);
   // }
   // rnew = r;
   // for (int ip = nrpatch-1; ip >=0; ip--)
   // {
   //    Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
   //    Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
   //    int ndofs = Dof2GlobalDof->Size();
   //    res_local.SetSize(ndofs); res_local = 0.0;
   //    sol_local.SetSize(ndofs);
   //    rnew.GetSubVector(*Dof2GlobalDof, res_local);

   //    // if (ip == nrpatch-1) ftransf[ip+1] = fn[ip+1];
   //    // res_local = 0.0;
   //    // res_local.SetSubVector(lmap->map1[ip],ftransf[ip]);
   //    // res_local.SetSubVector(lmap->map2[ip],fn[ip+1]);

   //    //-----------------------------------------------
   //    // Extend by zero to the extended mesh
   //    int nrdof_ext = PmlMat[ip]->Height();
         
   //    Vector res_ext(nrdof_ext); res_ext = 0.0;
   //    Vector sol_ext(nrdof_ext); sol_ext = 0.0;

   //    res_ext.SetSubVector(*Dof2PmlDof,res_local.GetData());
   //    PmlMatInv[ip]->Mult(res_ext, sol_ext);

   //    //-----------------------------------------------
   //    //          FOR PURE ST
   //    //-----------------------------------------------
   //    if (ip != 0) 
   //    {
   //       int direction = -1;
   //       GetCutOffSolution(sol_ext, ip, direction, true);
   //       // find the residual
   //       // Vector respml(sol_ext.Size());
   //       // PmlMat[ip]->Mult(sol_ext,respml);

   //       // restrict to non-pml problem
   //       // Vector res(ndofs); 
   //       // respml.GetSubVector(*Dof2PmlDof, res);
   //       // source to be transfered
   //       // res.GetSubVector(lmap->map2[ip],ftransf[ip+1]);

   //    }
   //    //-----------------------------------------------
   //    //-----------------------------------------------

   //    sol_ext.GetSubVector(*Dof2PmlDof,sol_local);

   //    znew = 0.0;
   //    znew.SetSubVector(*Dof2GlobalDof,sol_local);

   //    Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip];
   //    bsol[ip].SetSize(nDof2GlobalDof->Size());
   //    znew.GetSubVector(*nDof2GlobalDof,bsol[ip]);

   //    // Find residual;
   //    Vector respml(sol_ext.Size());
   //    Vector reslocal(sol_local.Size());
   //    PmlMat[ip]->Mult(sol_ext,respml);
   //    respml *= -1.0;

   //    respml.GetSubVector(*Dof2PmlDof,reslocal);
   //    // source to be transfered
   //    // reslocal.GetSubVector(lmap->map1[ip],ftransf[ip]);
   //    // ftransf[ip]+= fn[ip];
   //    rnew.AddElementVector(*Dof2GlobalDof,reslocal);

   // //    Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip];
   // //    bsol[ip].SetSize(nDof2GlobalDof->Size());
   // //    znew.GetSubVector(*nDof2GlobalDof,bsol[ip]);
   // //    if (ip != 0) 
   // //    {
   // //       int direction = -1;
   // //       GetCutOffSolution(znew, ip, direction);
   // //    }
   // //    if (ip != nrpatch-1) z2+=znew;
   // //    A->Mult(znew, raux);
   // //    rnew -= raux;
   //    z2+=znew;
   // }

   // z = z2;
   // Array<Vector> gsol(nrpatch+1);
   // // socketstream subsol5_sock(vishost, visport);

   // for (int ip = 0; ip<=nrpatch; ip++)
   // {
   //    if (ip == 0) 
   //    {
   //       gsol[ip].SetSize(bsol[ip].Size());
   //       gsol[ip] = 0.0;
   //       gsol[ip] = bsol[ip];
   //    }
   //    else if (ip == nrpatch)
   //    {
   //       gsol[ip].SetSize(fsol[ip].Size());
   //       gsol[ip] = 0.0;
   //       // gsol[ip] = fsol[ip];
   //    }
   //    else
   //    {
   //       gsol[ip].SetSize(fsol[ip].Size());
   //       gsol[ip] = 0.0;
   //       gsol[ip] += bsol[ip];
   //       // gsol[ip] += fsol[ip];
   //    }
      
   //    znew = 0.0;
   //    Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip];
   //    znew.SetSubVector(*nDof2GlobalDof,gsol[ip]);
   //    // PlotSolution(znew, subsol5_sock,0); cin.get();
   //    z.SetSubVector(*nDof2GlobalDof,gsol[ip]);
   //    // z.AddElementVector(*nDof2GlobalDof,gsol[ip]);
   // }
   // z = z1;

}

void DiagST::PlotSolution(Vector & sol, socketstream & sol_sock, int ip) const
{
   FiniteElementSpace * fespace = bf->FESpace();
   Mesh * mesh = fespace->GetMesh();
   GridFunction gf(fespace);
   double * data = sol.GetData();
   // gf.SetData(&data[fespace->GetTrueVSize()]);
   gf.SetData(data);
   
   string keys;
   if (ip == 0) keys = "keys mrRljc\n";
   sol_sock << "solution\n" << *mesh << gf << keys << "valuerange -0.1 0.1 \n"  << flush;
}

void DiagST::GetCutOffSolution(const Vector & sol, Vector & cfsol, 
                               int ip0, Array<int> directions, bool local) const
{
   // int l,k;
   int d = directions.Size();
   int directx = directions[0]; // 1,0,-1
   int directy = directions[1]; // 1,0,-1
   int directz;
   if (d ==3) directz = directions[2];

   // cout << "ip0 = " << ip0 << endl; 

   int i0, j0, k0;
   Getijk(ip0,i0, j0, k0);
   // cout << "(i0,j0) = " << "(" <<i0 <<","<<j0<<")" << endl;

   // 2D for now...
   // Find the id of the neighboring patch
   int i1 = i0 + directx;
   int j1 = j0 + directy;
   MFEM_VERIFY(i1 < nxyz[0] && i1>=0, "GetCutOffSolution: i1 out of bounds");
   MFEM_VERIFY(j1 < nxyz[1] && j1>=0, "GetCutOffSolution: j1 out of bounds");
   
   Array<int> ijk(d);
   ijk[0] = i1;
   ijk[1] = j1;
   int ip1 = GetPatchId(ijk);

   // cout << "ip1 = " << ip1 << endl;
   // cout << "(i1,j1) = " << "(" << i1 <<","<<j1<<")" << endl;

   Mesh * mesh0 = ovlp_prob->fespaces[ip0]->GetMesh();
   Mesh * mesh1 = ovlp_prob->fespaces[ip1]->GetMesh();
   
   Vector pmin0, pmax0;
   Vector pmin1, pmax1;
   mesh0->GetBoundingBox(pmin0, pmax0);
   mesh1->GetBoundingBox(pmin1, pmax1);

   Array2D<double> h(dim,2); h = 0.0;
   
   if (directions[0]==1)
   {
      h[0][1] = pmax0[0] - pmin1[0];
   }
   if (directions[0]==-1)
   {
      h[0][0] = pmax1[0] - pmin0[0];
   }
   if (directions[1]==1)
   {
      h[1][1] = pmax0[1] - pmin1[1];
   }
   if (directions[1]==-1)
   {
      h[1][0] = pmax1[1] - pmin0[1];
   }

   CutOffFnCoefficient cf(CutOffFncn, pmin0, pmax0, h);

   double * data = sol.GetData();

   FiniteElementSpace * fespace;
   if (!local)
   {
      fespace = bf->FESpace();
   }
   else
   {
      fespace = ovlp_prob->PmlFespaces[ip0];
   }
   
   int n = fespace->GetTrueVSize();
   // GridFunction cutF(fespace);
   // cutF.ProjectCoefficient(cf);
   // char vishost[] = "localhost";
   // int  visport   = 19916;

   // socketstream sub_sock1(vishost, visport);
   // sub_sock1 << "solution\n" << *fespace->GetMesh() << cutF << flush;
   // cin.get();


   GridFunction solgf_re(fespace, data);
   GridFunction solgf_im(fespace, &data[n]);

   // socketstream sub_sock(vishost, visport);
   // sub_sock << "solution\n" << *fespace->GetMesh() << solgf_re << flush;
   // cin.get();


   GridFunctionCoefficient coeff1_re(&solgf_re);
   GridFunctionCoefficient coeff1_im(&solgf_im);

   ProductCoefficient prod_re(coeff1_re, cf);
   ProductCoefficient prod_im(coeff1_im, cf);

   ComplexGridFunction gf(fespace);
   gf.ProjectCoefficient(prod_re,prod_im);

   cfsol.SetSize(sol.Size());
   cfsol = gf;
   // socketstream sub_sock2(vishost, visport);
   // sub_sock2 << "solution\n" << *fespace->GetMesh() << gf.real() << flush;
   // cin.get();
}

DiagST::~DiagST()
{
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      delete PmlMatInv[ip];
      delete PmlMat[ip];
   }
   PmlMat.DeleteAll();
   PmlMatInv.DeleteAll();
   for (int ip=0; ip<nrpatch; ip++)
   {
      delete f_orig[ip];
      for (int i=0;i<ntransf_directions; i++)
      {
         delete f_transf[ip][i];
      }
   }
}

void DiagST::Getijk(int ip, int & i, int & j, int & k) const
{
   k = ip/(nxyz[0]*nxyz[1]);
   j = (ip-k*nxyz[0]*nxyz[1])/nxyz[0];
   i = (ip-k*nxyz[0]*nxyz[1])%nxyz[0];
}

int DiagST::GetPatchId(const Array<int> & ijk) const
{
   int d=ijk.Size();
   if (d==2)
   {
      return subdomains(ijk[0],ijk[1],0);
   }
   else
   {
      return subdomains(ijk[0],ijk[1],ijk[2]);
   }
}

int DiagST::SourceTransfer(const Vector & Psi0, Array<int> direction, int ip0, Vector & Psi1) const
{
   // For now 2D problems only
   // Directions
   // direction (1,1)
   int i0,j0,k0;
   Getijk(ip0,i0,j0,k0);

   int i1 = i0+direction[0];   
   int j1 = j0+direction[1];   
   Array<int> ij(2); ij[0]=i1; ij[1]=j1;
   int ip1 = GetPatchId(ij);

   MFEM_VERIFY(i1 < nxyz[0] && i1>=0, "SourceTransfer: i1 out of bounds");
   MFEM_VERIFY(j1 < nxyz[1] && j1>=0, "SourceTransfer: j1 out of bounds");

   Array<int> * Dof2GlobalDof0 = &ovlp_prob->Dof2GlobalDof[ip0];
   Array<int> * Dof2GlobalDof1 = &ovlp_prob->Dof2GlobalDof[ip1];
   Psi1.SetSize(Dof2GlobalDof1->Size()); Psi1=0.0;
   Vector r(2*bf->FESpace()->GetTrueVSize());
   r = 0.0;
   r.SetSubVector(*Dof2GlobalDof0,Psi0);
   r.GetSubVector(*Dof2GlobalDof1,Psi1);
   return ip1;
}

void DiagST::ConstructDirectionsMap()
{
   // total of 8 possible directions of transfer (2D)
   // form left            ( 1 ,  0)
   // form left-above      ( 1 , -1)
   // form left-below      ( 1 ,  1)
   // form right           (-1 ,  0)
   // form right-below     (-1 ,  1)
   // form right-above     (-1 , -1)
   // form above           ( 0 , -1)
   // form below           ( 0 ,  1)
   ntransf_directions = pow(3,dim);

   dirx.SetSize(ntransf_directions);
   diry.SetSize(ntransf_directions);
   int n=3;
   Array<int> ijk(dim);
   if (dim==2)
   {
      for (int i=-1; i<=1; i++) // directions x
      {
         for (int j=-1; j<=1; j++) // directions y
         {
            ijk[0]=i;
            ijk[1]=j;
            int k=GetDirectionId(ijk);
            dirx[k]=i;
            diry[k]=j;
         }
      }
   }
   else if (dim==3)
   {
      dirz.SetSize(ntransf_directions);
      for (int i=-1; i<=1; i++) // directions x
      {
         for (int j=-1; j<=1; j++) // directions y
         {
            for (int k=-1; k<=1; k++) // directions zß
            {
               ijk[0]=i;
               ijk[1]=j;
               ijk[2]=k;
               int l=GetDirectionId(ijk);
               dirx[l]=i;
               diry[l]=j;
               dirz[l]=k;
            }
         }
      }
   }

   // cout << "dirx = " << endl;
   // dirx.Print(cout,ntransf_directions);
   // cout << "diry = " << endl;
   // diry.Print(cout,ntransf_directions);

   if (dim==2)
   {
      for (int id=0; id<9; id++)
      {
         GetDirectionijk(id,ijk);
         // cout << "for id = " << id << ": (" <<ijk[0] << ", " << ijk[1] << ")" << endl;
      }
   }
   else
   {
      cout << "dirz = " << endl;
      dirz.Print(cout,ntransf_directions);
      for (int id=0; id<27; id++)
      {
         GetDirectionijk(id,ijk);
         // cout << "for id = " << id << ": (" <<ijk[0] << ", " <<ijk[1] << ", " << ijk[2] << ")" << endl;
      }
   }
}

int DiagST::GetDirectionId(const Array<int> & ijk) const
{
   int d = ijk.Size();
   int n=3;
   if (d==2)
   {
      return (ijk[0]+1)*n+(ijk[1]+1);
   }
   else
   {
      return (ijk[0]+1)*n*n+(ijk[1]+1)*n+ijk[2]+1;
   }
}

void DiagST::GetDirectionijk(int id, Array<int> & ijk) const
{
   int d = ijk.Size();
   int n=3;
   if (d==2)
   {
      ijk[0]=id/n - 1;
      ijk[1]=id%n - 1;
   }
   else
   {
      ijk[0]=id/(n*n)-1;
      ijk[1]=(id-(ijk[0]+1)*n*n)/n - 1;
      ijk[2]=(id-(ijk[0]+1)*n*n)%n - 1;
   }
   // cout << "ijk = " ; ijk.Print();
}




void DiagST::TransferSources(int ip0, Vector & sol_ext) const
{
   // Find all neighbors of patch ip
   int nx = nxyz[0];
   int ny = nxyz[1];
   int i0, j0, k0;
   Getijk(ip0, i0,j0,k0);
   cout << "Transfer to : " << endl;
   // loop through possible directions
   for (int i=-1; i<2; i++)
   {
      int i1 = i0 + i;
      if (i1 <0 || i1>=nx) continue;
      for (int j=-1; j<2; j++)
      {
         int j1 = j0 + j;
         if (j1 <0 || j1>=ny) continue;
         cout << "(" << i1 << "," << j1 <<"), ";
         // Find ip 1
         Array<int> ij1(2); ij1[0] = i1; ij1[1]=j1;
         int ip1 = GetPatchId(ij1);
         cout << "ip1 = " << ip1;
         cout << " in the direction of (" << i <<", " <<j <<")" << endl;
         Array<int> directions(2);
         directions[0] = i;
         directions[1] = j;
         Vector cfsol_ext;
         Vector res_ext(sol_ext.Size());
         GetCutOffSolution(sol_ext,cfsol_ext,ip0,directions,true);
         // sol_ext = cfsol_ext;
         // Calculate source to be transfered
         PmlMat[ip0]->Mult(cfsol_ext, res_ext); res_ext*= -1.0;
         Array<int> *Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip0];
         Vector res_local(Dof2PmlDof->Size()); res_local = 0.0;
         res_ext.GetSubVector(*Dof2PmlDof,res_local);
         // Find the direction id to store the transfered source
         Array<int> dij(2); dij[0] = -i; dij[1] = -j; 
         int did = GetDirectionId(dij);
         int jp1 = SourceTransfer(res_local,directions,ip0,*f_transf[ip1][did]);
         MFEM_VERIFY(ip1 == jp1, "Check SourceTransfer patch id");
      }  
   }
}






// void DiagST::GetCutOffSolution(Vector & sol, int ip, int direction, bool local) const
// {
//    int l,k;
//    k=(direction == 1)? ip: ip-1;
//    l=(direction == 1)? ip+1: ip;

//    Mesh * mesh1 = ovlp_prob->fespaces[k]->GetMesh();
//    Mesh * mesh2 = ovlp_prob->fespaces[l]->GetMesh();
   
//    Vector pmin1, pmax1;
//    Vector pmin2, pmax2;
//    mesh1->GetBoundingBox(pmin1, pmax1);
//    mesh2->GetBoundingBox(pmin2, pmax2);

//    Array2D<double> h(dim,2); h = 0.0;
   
//    Vector pmin, pmax;
//    if (direction == 1)
//    {
//       h[0][1] = pmax1[0] - pmin2[0];
//       CutOffFnCoefficient cf(CutOffFncn, pmin1, pmax1, h);
//       pmin = pmin1;
//       pmax = pmax1;
//    }
//    else if (direction == -1)
//    {
//       h[0][0] = pmax1[0] - pmin2[0];
//       pmin = pmin2;
//       pmax = pmax2;
//    }
//    CutOffFnCoefficient cf(CutOffFncn, pmin, pmax, h);

//    double * data = sol.GetData();

//    FiniteElementSpace * fespace;
//    if (!local)
//    {
//       fespace = bf->FESpace();
//    }
//    else
//    {
//       if (direction == 1)
//       {
//          fespace = ovlp_prob->PmlFespaces[k];
//       }
//       else
//       {
//          fespace = ovlp_prob->PmlFespaces[l];
//       }
//    }
   
//    int n = fespace->GetTrueVSize();
//    GridFunction cutF(fespace);
//    cutF.ProjectCoefficient(cf);
//    // char vishost[] = "localhost";
//    // int  visport   = 19916;

   

//    // socketstream sub_sock1(vishost, visport);
//    // sub_sock1 << "solution\n" << *fespace->GetMesh() << cutF << flush;
//    // cin.get();


//    GridFunction solgf_re(fespace, data);

//    // socketstream sub_sock(vishost, visport);
//    // sub_sock << "solution\n" << *fespace->GetMesh() << solgf_re << flush;
//    // cin.get();

//    GridFunction solgf_im(fespace, &data[n]);

//    GridFunctionCoefficient coeff1_re(&solgf_re);
//    GridFunctionCoefficient coeff1_im(&solgf_im);

//    ProductCoefficient prod_re(coeff1_re, cf);
//    ProductCoefficient prod_im(coeff1_im, cf);

//    ComplexGridFunction gf(fespace);
//    gf.ProjectCoefficient(prod_re,prod_im);

//    sol = gf;
//    // socketstream sub_sock2(vishost, visport);
//    // sub_sock2 << "solution\n" << *fespace->GetMesh() << gf.real() << flush;
//    // cin.get();
// }