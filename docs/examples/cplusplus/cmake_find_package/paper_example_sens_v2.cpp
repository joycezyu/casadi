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
  */



  SX x1 = SX::sym("x1");
  SX x2 = SX::sym("x2");
  SX x3 = SX::sym("x3");
  SX p  = SX::sym("p", 2);


  // Objective
  SX f = x1*x1 + x2*x2 + x3*x3;

  // Constraints
  SX g = vertcat(
      6*x1 + 3*x2 + 2*x3 - p(0),
      p(1)*x1 +   x2 -   x3 -   1
  );

  SX x = SX::vertcat({x1,x2,x3});
  SXDict nlp = {{"x", x},
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

  vector<DM> prim_param{res.at("x"), p1};




  int ng = g.size1();  // ng = number of constraints g
  int nx = x.size1();
  cout << "nx = " << nx << endl;
  SX lambda = SX::sym("lambda", ng);
  SX lagrangian = f + dot(lambda, g);

  SX grad = jacobian(g, x);
  std::cout << "jacobians of constraints" << grad << std::endl;

  auto jac_lagrangian = jacobian(lagrangian, x);
  std::cout << "jacobians of lagrangian" << jac_lagrangian << std::endl;

  // both the following two lines work to get the hessian
  SX hess = jacobian(jac_lagrangian, x);
  SX hess1 = hessian(lagrangian, x);
  std::cout << hess1 << std::endl;

  // Function, input, output, input-name, output-name
  Function f_jac("grad",{x, p}, {grad}, {"x", "p"}, {"j"});
  auto j = f_jac(prim_param);
  cout << j << endl;



  Function f_hess("hessian",{x, lambda}, {hess1});
  // using the following format to evaluate hessian if the input contains x and lambda
  // apparently either SX or DM datatype can work with res.at()
  //vector<SX> prim_dual{res.at("x"), res.at("lam_g")};
  vector<DM> prim_dual{res.at("x"), res.at("lam_g")};
  auto h = f_hess(prim_dual);
  cout << h << endl;

  //cout << SX::vertcat({hess1, grad}) << endl;


  /// Assemble KKT matrix
  // TODO: think about using sparsity zeros??
  DM M11 = DM::zeros(ng, ng); // zero matrix at (1,1)
  DM M21 = DM::zeros(nx, ng); // zero matrix at (2,1)
  DM M12 = DM::zeros(ng, nx); // zero matrix at (1,2)

  DM N0 = DM::zeros(nx, 1);

  DM V = DM::diag(res.at("lam_x"));
  DM X = DM::diag(res.at("x"));

  DM I = DM::eye(nx);



  /// LHS
  SX KKTprimer = SX::horzcat({SX::vertcat({hess1, grad, V}), SX::vertcat({grad.T(), M11, M21}),
                              SX::vertcat({   -I,  M12, X}) });
  cout << KKTprimer << endl;
  //Function KKT("KKT",{x, lambda}, {KKTprimer});
  //cout << KKT(prim_dual) << endl;

  /// RHS
  SX phi = SX::vertcat({jac_lagrangian.T(), g, N0});
  SX sensitivity = solve(KKTprimer, -phi);
  Function sens("sens",{x, lambda, p}, {sensitivity});
  vector<DM> prim_dual_param{res.at("x"), res.at("lam_g"), p1};

  Matrix<double> ds = DM::vertcat({sens(prim_dual_param)});
  cout << "ds = " << ds << endl;
  cout << ds.size() << endl;

  Matrix<double> s = DM::vertcat({res.at("x"), res.at("lam_g"), res.at("lam_x")});
  cout << s.size() << endl;
  cout << s+ds << endl;






  return 0;
}
