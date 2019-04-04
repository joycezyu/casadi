/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#include <iostream>
#include <fstream>
#include <ctime>
#include <casadi/casadi.hpp>
//#include <Eigen/Dense>
//#include <casadi/casadi/core/sx.hpp>

using namespace casadi;
using namespace std;

int main() {
  /**
   *  Examples for Pirnay et al. (2012) sIPOPT paper
   *
   *    min     x1^2 + x2^2 + x3^2
   *    s.t.    6*x1 + 3&x2 + 2*x3 - pi = 0
   *            p2*x1 + x2 - x3 - 1 = 0
   *            x1, x2, x3 >= 0
   *
   *
   *
   *  Sensitivity calculates M * ds = - N, where
   *      [ W    A   -I  ]            [grad_x L   ]            [dx       ]
   *  M = [ A^T  0    0  ]   ,  N  =  [       c   ]   ,  ds =  [d_lambda ]
   *      [ V    0    X  ]            [       0   ]            [dv       ]
   *
   *  Instead of solving the nonsymmetric linear system, equivalently we solve
   *
   *  M = [ W + X^(-1)V     A  ]  ,   N =  [grad_x L   ] ,  ds = [dx       ]
   *      [  A^T            0  ]           [       c   ]         [d_lambda ]
   *
   *  and dv = -X^(-1)V dx
   *
   *
  */




  /// Model construction


  MX x1 = MX::sym("x1");
  MX x2 = MX::sym("x2");
  MX x3 = MX::sym("x3");
  MX p  = MX::sym("p", 2);


  // Objective
  MX f = x1*x1 + x2*x2 + x3*x3;

  // Constraints
  MX g = vertcat(
      6*x1 + 3*x2 + 2*x3 - p(0),
      p(1)*x1 +   x2 -   x3 -   1
  );

  MX x = MX::vertcat({x1,x2,x3});
  MXDict nlp = {{"x", x},
                {"p", p},
                {"f", f},
                {"g", g}};

  // Initial guess and bounds for the optimization variables
  vector<double> x0  = {0.15, 0.15, 0.00};
  vector<double> lbx = { 0, 0, 0};
  vector<double> ubx = { inf,  inf,  inf};

  // Nonlinear bounds
  vector<double> lbg = {0.00, 0.00};
  vector<double> ubg = {0.00, 0.00};

  // Original parameter values
  vector<double> p0  = {5.00, 1.00};
  // new parameter values
  vector<double> p1  = {4.50, 1.00};

  // Create NLP solver and buffers
  Function solver = nlpsol("solver", "ipopt", nlp);
  std::map<std::string, DM> arg, res;


  // Solve the NLP
  arg["lbx"] = lbx;
  arg["ubx"] = ubx;
  arg["lbg"] = lbg;
  arg["ubg"] = ubg;
  arg["x0"] = x0;
  arg["p"] = p0;
  res = solver(arg);


  cout << res << endl;
  // Print the solution
  cout << "-----" << endl;
  cout << "Optimal solution for p = " << arg.at("p") << ":" << endl;
  cout << setw(30) << "Objective: " << res.at("f") << endl;
  cout << setw(30) << "Primal solution: " << res.at("x") << endl;
  cout << setw(30) << "Dual solution (x): " << res.at("lam_x") << endl;
  cout << setw(30) << "Dual solution (g): " << res.at("lam_g") << endl;
  cout << setw(30) << "Dual solution (p): " << res.at("lam_p") << endl;

  vector<DM> prim_param{res.at("x"), p1};


  /// sensitivity calculation


  int ng = g.size1();  // ng = number of constraints g
  int nx = x.size1();
  cout << "nx = " << nx << endl;
  MX lambda = MX::sym("lambda", ng);
  MX v      = MX::sym("v", nx);
  MX V      = MX::diag(v);
  MX Xinv   = MX::diag(1/x);
  MX XiV    = mtimes(Xinv, V);
  cout << " XiV = " << XiV << endl;
  MX lagrangian = f + dot(lambda, g) - dot(v, x);

  MX grad = jacobian(g, x);
  std::cout << "jacobians of constraints" << grad << std::endl;

  auto jac_lagrangian = jacobian(lagrangian, x);
  std::cout << "jacobians of lagrangian" << jac_lagrangian << std::endl;


  MX hess = hessian(lagrangian, x);
  std::cout << hess << std::endl;

  // Function_name(input, output, input-name, output-name)
  Function f_jac("grad",{x, p}, {grad}, {"x", "p"}, {"j"});
  auto j = f_jac(prim_param);
  cout << j << endl;



  Function f_hess("hessian",{x, lambda, v}, {hess});
  // using the following format to evaluate hessian if the input contains x and lambda
  // apparently either SX or DM datatype can work with res.at()
  //vector<SX> prim_dual{res.at("x"), res.at("lam_g")};
  vector<DM> prim_dual{res.at("x"), res.at("lam_g"), res.at("lam_x")};
  auto h = f_hess(prim_dual);
  cout << h << endl;

  //cout << SX::vertcat({hess1, grad}) << endl;


  /// Assemble KKT matrix
  // TODO: think about using sparsity zeros??
  //DM M0 = DM::zeros(ng, ng);
  // sparse zero matrix
  MX M0 = MX(ng, ng);
  //MX V = MX::diag(res.at("lam_x"));
  //MX X_inv = MX::diag(1/res.at("x"));
  //MX XiV = mtimes(X_inv, V);
  //cout << "size XiV = " << XiV.size() << endl;
  //cout << "X =   " << X << endl;
  //cout << "inv(X) =   " << inv(X) << endl;

  // LHS
  MX KKTprimer = MX::horzcat({MX::vertcat({hess + XiV, grad}),
                              MX::vertcat({grad.T(), M0})});

  cout << "KKT = " << KKTprimer << endl;
  //Function KKT("KKT",{x, lambda}, {KKTprimer});
  //cout << KKT(prim_dual) << endl;

  /// RHS
  MX phi = MX::vertcat({jac_lagrangian.T(), g});
  MX sensitivity = solve(KKTprimer, -phi);
  //MX sensitivity = solve(KKTprimer, -phi, "cssparse");
  //MX sensitivity = solve(KKTprimer, -phi, "ldl");
  //MX sensitivity = solve(KKTprimer, -phi, "qr");
  Function sens_eval("sens", {x, lambda, v, p}, {sensitivity});
  cout << "-1" << endl;
  vector<DM> prim_dual_param{res.at("x"), res.at("lam_g"), res.at("lam_x"), p1};

  cout << "0" << endl;
  DM dx_dl = DM::vertcat({sens_eval(prim_dual_param)});

  cout << "1" << endl;
  // construct dx (only for primal)
  //cout << "dx = "  << ds(Slice(0, nx)) << endl;
  MX dx = MX::sym("dx", nx);
  MX dv = -mtimes(XiV, dx);
  Function dv_eval("dv", {x, v, dx}, {dv});
  vector<DM> x_v_dx{res.at("x"), res.at("lam_x"), dx_dl(Slice(0, nx))};
  //MX dv = -mtimes(XiV, sensitivity(Slice(0, nx)));   // dv is of datatype MX, need to
  DM dvv = DM::vertcat({dv_eval(x_v_dx)});
  cout << "dv = " << dv << endl;
  cout << "dv_eval = " << dvv << endl;
  //Function dv_eval("dv", {x, lambda, v, p}, {dv});

  //Eigen::MatrixXd dv(1, nx) = XiV * ds(Slice(0, nx));

  DM ds = DM::vertcat({dx_dl, dvv});
  cout << "2" << endl;
  cout << "ds = " << ds << endl;
  cout << ds.size() << endl;

  //Matrix<double> total_ds = MX::vertcat({ds, dv});
  //cout << "dv = " << dv << endl;

  DM s = DM::vertcat({res.at("x"), res.at("lam_g"), res.at("lam_x")});
  cout << s.size() << endl;

  Matrix<double> s1 = s+ds;
  cout << s1 << endl;





  return 0;
}
