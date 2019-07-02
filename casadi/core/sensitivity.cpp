//
// Created by Zhou Yu on 4/15/19.
//

#include <iostream>
#include <fstream>
#include <iomanip>      // std::setprecision
#include "sensitivity.hpp"
#include "timing.hpp"

using namespace std;
namespace casadi {

DM NLPsensitivity(const std::string& lsolver, const std::map<std::string, DM>& res,
                  const MX& objective, const MX& constraints, const MX& variables, const MX& parameters,
                  const std::vector<double>& p0, const std::vector<double>& p1) {

  cout << "********************************" << endl;
  cout << "Start of sensitivity calculation" << endl;
  cout << "With linear solver = " << lsolver << endl;


  const MX& f = objective;
  const MX& g = constraints;
  const MX& x = variables;
  const MX& p = parameters;
  //cout << "f = " << f << endl;
  //cout << "g = " << g << endl;

  int ng = g.size1();  // ng = number of constraints g
  int nx = x.size1();  // nx = number of variables x
  int np = p0.size();

  // the symbolic expression for x, λ, ν
  MX lambda = MX::sym("lambda", ng);
  MX v      = MX::sym("v", nx);
  MX V      = MX::diag(v);
  MX X      = MX::diag(x);
  //MX XiV    = mtimes(inv(X), V);
  //MX XiV    = MX::diag(dot(v,1/x));      // equivalent results, not sure about efficiency
  MX XiV    = MX::diag(v/x);

  MX grad = jacobian(g, x);
  // construct the lagrangian function
  MX lagrangian = f + dot(lambda, g) + dot(v, x);
  //MX lagrangian = f + dot(lambda, g);
  MX jac_lagrangian = jacobian(lagrangian, x);
  // writing hessian in the following two different ways does change the numerical results a bit
  // however, it is NOT the reason for inconsistent W with ipopt
  //MX hess = hessian(lagrangian, x);
  MX hess = jacobian(jac_lagrangian, x);

  //cout << "hessian = " << hess << endl;

  Function hess_eval("W", {x, lambda, v, p}, {hess});


  /// Assemble KKT matrix
  // sparse zero matrix
  MX M0 = MX(ng, ng);

  // LHS
  MX KKTprimer = MX::horzcat({MX::vertcat({hess + XiV, grad}),
                              MX::vertcat({grad.T(), M0})});

  MX KKT_noaugment = MX::horzcat({MX::vertcat({hess, grad}),
                                  MX::vertcat({grad.T(), M0})});

  // RHS
  MX phi = MX::vertcat({jac_lagrangian.T(), g});

  /// keep track of the solving time
  FStats time;
  time.tic();

  /// Solve linear system

  MX sensitivity = solve(KKTprimer, -phi, lsolver);
  // can use the following sparse linear solvers if large-scale
  //MX sensitivity = solve(KKTprimer, -phi, "ma27");
  //MX sensitivity = solve(KKTprimer, -phi, "csparse");
  // MX sensitivity = solve(KKTprimer, -phi, "ldl");
  //MX sensitivity = solve(KKTprimer, -phi, "qr");

  // MX p  = MX::sym("p", np);
  time.toc();
  cout << "linear system t_wall time = " << time.t_wall << endl;
  cout << "linear system t_proc time = " << time.t_proc << endl;


  Function sens_eval("sens", {x, lambda, v, p}, {sensitivity});
  int x_tot  = res.at("x").size1();
  int lg_tot = res.at("lam_g").size1();
  int lx_tot = res.at("lam_x").size1();

  // the below is for testing RHS = 0
  // vector<double> x0(x_tot, 1e-12);
  // vector<double> lg0(lg_tot, 1e-12);
  // vector<double> lx0(lx_tot, 1e-12);
  // vector<DM> prim_dual_param{x0, lg0, lx0, p1};


  vector<DM> prim_dual_param{res.at("x"), res.at("lam_g"), res.at("lam_x"), p1};

  cout << "prim_dual_param = ";
  for (int i=0; i<prim_dual_param.size(); ++i) {
    for (int j=0; j<prim_dual_param[i].size1(); ++j) {
      cout << setprecision(20) << double(prim_dual_param[i](j)) << ",    " ;
    }
    cout << endl;
  }

  // solution vector for 2x2 system is [Δx, Δλ]ᵀ
  DM dx_dl = DM::vertcat({sens_eval(prim_dual_param)});


  // take a look at RHS
  Function RHS_eval("RHS", {x, lambda, v, p}, {phi});
  DM RHS = DM::vertcat({RHS_eval(prim_dual_param)});
  cout << "RHS_x = "    << RHS(Slice(0, x_tot)) << endl;
  cout << "RHS_lamg = " << RHS(Slice(x_tot, x_tot + lg_tot)) << endl;


  /// take a look at KKT
  Function KKT_eval("KKT", {x, lambda, v, p}, {KKTprimer});
  DM KKT = DM::vertcat({KKT_eval(prim_dual_param)});
  DM Wa  = KKT(Slice(0, x_tot), Slice(0, x_tot));
  DM A   = KKT(Slice(0, x_tot), Slice(x_tot, x_tot + lg_tot));


  /// evaluate ds
  // the below is for testing RHS = 0
  vector<double> x0(x_tot, 1e-20);
  vector<double> lg0(lg_tot, 1e-20);
  vector<double> lx0(lx_tot, 1e-20);
  vector<DM> prim_dual_param0{x0, lg0, lx0, p1};
  DM RHS0 = DM::vertcat({x0, lg0});
  //cout << "RHS0 (should be all zero) = " << RHS0 << endl;

  //DM KKT_num = solve(KKT, -RHS, lsolver);
  DM KKT_num = solve(KKT, -RHS0, lsolver);
  //cout << "KKT_num = " << KKT_num << endl;

  // cout << "W + Σ = "  ;
  /*
  for (int i=0; i<Wa.size1(); ++i) {
    cout << "row " << i << " = " << Wa(Slice(i, i+1), Slice(0, Wa.size1())) << endl;
  }
  */
  std::cout << std::fixed;
  cout << setprecision(10);

  cout << "W + Σ ="  << Wa << endl;
  cout << "A = "     <<  A << endl;

  Function KKT0_eval("KKT0", {x, lambda, v, p}, {KKT_noaugment});
  DM KKT0 = DM::vertcat({KKT0_eval(prim_dual_param)});
  DM W = KKT0(Slice(0, x_tot), Slice(0, x_tot));
  cout << "W = " << W << endl;
  //cout << "W(52, 52) = " << setprecision(15) << W(52, 52) << endl;

  /*
  cout << "W = ";
  for (int i=0; i<W.size1(); ++i) {
    for (int j=0; j<W.size2(); ++j) {
      cout << "W[" << i << ", " << j << "] = " << setprecision(15) << W(i)(j) << endl;
    }
    //cout << "row " << i << " = " << W(Slice(i, i+1), Slice(0, W.size1())) << endl;
  }
  */



  /// testing for W
  DM hess_raw = DM::vertcat({hess_eval(prim_dual_param)});
  cout << "W0 = "    << hess_raw << endl;







  // compute Δν
  MX dx = MX::sym("dx", nx);
  MX dv = -mtimes(XiV, dx);
  Function dv_eval("dv", {x, v, dx}, {dv});
  vector<DM> x_v_dx{res.at("x"), res.at("lam_x"), dx_dl(Slice(0, nx))};
  DM dv0 = DM::vertcat({dv_eval(x_v_dx)});

  // assemble ds matrix
  DM ds = DM::vertcat({dx_dl, dv0});

  cout << "******************************" << endl;
  cout << "End of sensitivity calculation" << endl;


  /// Output KKT matrix to matlab file

  // Create Matlab script to plot the solution
  ofstream file;
  string filename = "sensitivity_results_KKT_N=1.m";
  file.open(filename.c_str());
  file << "% Results file from " __FILE__ << endl;
  file << "% Generated " __DATE__ " at " __TIME__ << endl;
  file << endl;
  //file << "W_r = " << Wa << ";" << endl;
  //file << "A   = " << A  << ";" << endl;

  file << "W = [" ;
  for (int i = 0; i < W.size1(); ++i) {
    for (int j = 0; j < W.size2(); ++j) {
      file << setprecision(20) << double(W(i, j))  ;
      if (j < W.size2()-1) {
        file << "," ;
      }
    }
    if (i < W.size2()-1) {
      file << ";" << endl;
    }
  }
  file << "];" << endl;




  file << "KKT = [" ;
  for (int i = 0; i < KKT.size1(); ++i) {
    for (int j = 0; j < KKT.size2(); ++j) {
      file << setprecision(20) << double(KKT(i, j))  ;
        if (j < KKT.size2()-1) {
          file << "," ;
        }
    }
    if (i < KKT.size2()-1) {
      file << ";" << endl;
    }
  }
  file << "];" << endl;

  file << "RHS = [" ;
  for (int i = 0; i < RHS.size1(); ++i) {
    file << setprecision(10) << RHS(i, 0) << ";" << endl;
  }
  file << "]" << endl;
  file << "cond(KKT)" << endl;
  file << "eig_KKT = eig(KKT);" << endl;
  file << "spy(KKT)"  << endl;



  //file.close();
  cout << "Results saved to \"" << filename << "\"" << endl;





  return ds;
};







DM NLPsensitivity_p(const std::string& lsolver, const std::map<std::string, DM>& res,
                    const MX& objective, const MX& constraints, const MX& variables, const MX& parameters,
                    const std::vector<double>& p0, const std::vector<double>& p1) {


  cout << "********************************" << endl;
  cout << "Start of sensitivity calculation" << endl;
  cout << "With linear solver = " << lsolver << endl;


  const MX& f = objective;
  const MX& g = constraints;
  const MX& x = variables;
  const MX& p = parameters;


  int ng = g.size1();  // ng = number of constraints g
  int nx = x.size1();  // nx = number of variables x
  int np = p0.size();

  // the symbolic expression for x, λ, ν
  MX lambda = MX::sym("lambda", ng);
  MX v      = MX::sym("v", nx);
  MX V      = MX::diag(v);
  MX X      = MX::diag(x);
  //MX XiV    = mtimes(inv(X), V);
  //MX XiV    = MX::diag(dot(v,1/x));      // equivalent results, not sure about efficiency
  MX XiV    = MX::diag(v/x);

  MX grad = jacobian(g, x);
  // construct the lagrangian function
  MX lagrangian = f + dot(lambda, g) + dot(v, x);
  //MX lagrangian = f + dot(lambda, g);
  MX jac_lagrangian = jacobian(lagrangian, x);
  // writing hessian in the following two different ways does change the numerical results a bit
  // however, it is NOT the reason for inconsistent W with ipopt
  //MX hess = hessian(lagrangian, x);
  MX hess = jacobian(jac_lagrangian, x);

  MX lag_xp = jacobian(jac_lagrangian, p);
  MX g_p = jacobian(g, p);



  //cout << "hessian = " << hess << endl;

  Function hess_eval("W", {x, lambda, v, p}, {hess});


  /// Assemble KKT matrix
  // sparse zero matrix
  MX M0 = MX(ng, ng);

  // LHS
  MX KKTprimer = MX::horzcat({MX::vertcat({hess + XiV, grad}),
                              MX::vertcat({grad.T(), M0})});

  MX KKT_noaugment = MX::horzcat({MX::vertcat({hess, grad}),
                                  MX::vertcat({grad.T(), M0})});

  // RHS
  //MX phi = MX::vertcat({jac_lagrangian.T(), g});
  MX phi = MX::vertcat({lag_xp, g_p});

  /// keep track of the solving time
  FStats time;
  time.tic();

  /// Solve linear system

  MX sensitivity = solve(KKTprimer, -phi, lsolver);

  // MX p  = MX::sym("p", np);
  time.toc();
  cout << "linear system t_wall time = " << time.t_wall << endl;
  cout << "linear system t_proc time = " << time.t_proc << endl;


  Function sens_eval("sens", {x, lambda, v, p}, {sensitivity});
  int x_tot  = res.at("x").size1();
  int lg_tot = res.at("lam_g").size1();
  int lx_tot = res.at("lam_x").size1();

  // the below is for testing RHS = 0
  // vector<double> x0(x_tot, 1e-12);
  // vector<double> lg0(lg_tot, 1e-12);
  // vector<double> lx0(lx_tot, 1e-12);
  // vector<DM> prim_dual_param{x0, lg0, lx0, p1};

  /// choose to evaluate the solution of linear system at p0 or p1
  vector<DM> prim_dual_p0{res.at("x"), res.at("lam_g"), res.at("lam_x"), p0};

  cout << "prim_dual_param = ";
  for (int i=0; i<prim_dual_p0.size(); ++i) {
    for (int j=0; j<prim_dual_p0[i].size1(); ++j) {
      cout << setprecision(20) << double(prim_dual_p0[i](j)) << ",    " ;
    }
    cout << endl;
  }

  // solution vector for 2x2 system is [Δx, Δλ]ᵀ
  DM dx_dl = DM::vertcat({sens_eval(prim_dual_p0)});


  // take a look at RHS
  Function RHS_eval("RHS", {x, lambda, v, p}, {phi});
  DM RHS = DM::vertcat({RHS_eval(prim_dual_p0)});
  cout << "RHS_x = "    << RHS(Slice(0, x_tot)) << endl;
  cout << "RHS_lamg = " << RHS(Slice(x_tot, x_tot + lg_tot)) << endl;


  /// take a look at KKT
  Function KKT_eval("KKT", {x, lambda, v, p}, {KKTprimer});
  DM KKT = DM::vertcat({KKT_eval(prim_dual_p0)});
  DM Wa  = KKT(Slice(0, x_tot), Slice(0, x_tot));
  DM A   = KKT(Slice(0, x_tot), Slice(x_tot, x_tot + lg_tot));


  /// evaluate ds
  // the below is for testing RHS = 0

  std::cout << std::fixed;
  cout << setprecision(10);

  cout << "W + Σ ="  << Wa << endl;
  cout << "A = "     <<  A << endl;

  Function KKT0_eval("KKT0", {x, lambda, v, p}, {KKT_noaugment});
  DM KKT0 = DM::vertcat({KKT0_eval(prim_dual_p0)});
  DM W = KKT0(Slice(0, x_tot), Slice(0, x_tot));
  cout << "W = " << W << endl;

  /// testing for W
  DM hess_raw = DM::vertcat({hess_eval(prim_dual_p0)});
  cout << "W0 = "    << hess_raw << endl;







  // compute Δν
  MX dx = MX::sym("dx", nx);
  MX dv = -mtimes(XiV, dx);
  Function dv_eval("dv", {x, v, dx}, {dv});
  vector<DM> x_v_dx{res.at("x"), res.at("lam_x"), dx_dl(Slice(0, nx), Slice())};
  DM dv0 = DM::vertcat({dv_eval(x_v_dx)});
  cout << "dv = " << dv0 << endl;

  // assemble ds matrix
  DM dsdp = DM::vertcat({dx_dl, dv0});
  cout << "dsdp = " << dsdp << endl;
  vector<double> delta_p;
  for (int i=0; i<p1.size(); ++i) {
    delta_p.push_back(p1[i] - p0[i]);
  }

  DM dp = delta_p ;

  DM ds = mtimes(dsdp, dp);

  cout << "******************************" << endl;
  cout << "End of sensitivity calculation" << endl;

  return ds;




};





}  // namespace casadi




