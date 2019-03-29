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
   * The following is a basic example for using SX datatype
  SX A = SX::sym("A",3,2);
  SX x = SX::sym("x",2);
  SX j = jacobian(mtimes(A,x),x);
  std::cout << j << std::endl;

  auto g = gradient(dot(A,A),A);
  std::cout << g << std::endl;

  auto H = hessian(dot(x,x),x);
  std::cout << H << std::endl;
  */



  SX x1 = SX::sym("x1");
  SX x2 = SX::sym("x2");
  SX u  = SX::sym("u");
  SX p  = SX::sym("p");


  // Objective
  SX f = x1 * x1;

  // Constraints
  SX g = vertcat(
      x1 + u + p,
      2*x2 + u - 0.5*p,
      x1 - 1
  );

  SX x = SX::vertcat({x1,x2,u});
  SXDict nlp = {{"x", x},
                {"p", p},
                {"f", f},
                {"g", g}};

  // Initial guess and bounds for the optimization variables
  vector<double> x0  = {0.15, 0.15, 0.00};
  //vector<double> lbx = {-inf, -inf, -inf};
  vector<double> lbx = {-1, -1, -1};
  vector<double> ubx = { inf,  inf,  inf};

  // Nonlinear bounds
  vector<double> lbg = {0.00, 0.00, -inf};
  vector<double> ubg = {0.00, 0.00, 0.00};

  // Original parameter values
  vector<double> p0  = {1.00};

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

  // cout << "grad[0] = " << grad[res.at("x")] << endl;

  cout << res << endl;
  // Print the solution
  cout << "-----" << endl;
  cout << "Optimal solution for p = " << arg.at("p") << ":" << endl;
  cout << setw(30) << "Objective: " << res.at("f") << endl;
  cout << setw(30) << "Primal solution: " << res.at("x") << endl;
  cout << setw(30) << "Dual solution (x): " << res.at("lam_x") << endl;
  cout << setw(30) << "Dual solution (g): " << res.at("lam_g") << endl;


  // Evaluate at the optimal solution
  //grad.setInput(solver.output(NLP_X_OPT));
  //grad.at(res);
  //hf.evaluate();



  // SX lagrangian = f + res.at("lam_g") * g + res.at("lam_x") * x;

  int ng(3);  // ng = number of constraints g
  SX lambda = SX::sym("lambda", ng);
  SX lagrangian = f + dot(lambda, g);

  SX grad = jacobian(g, x);
  std::cout << grad << std::endl;

  auto jac_lagrangian = jacobian(lagrangian, x);
  // both the following two lines work to get the hessian
  SX hess = jacobian(jac_lagrangian, x);
  SX hess1 = hessian(lagrangian, x);
  std::cout << hess1 << std::endl;

  // Function, input, output, input-name, output-name
  Function f_jac("grad",{x}, {grad}, {"x"}, {"j"});
  auto j = f_jac(res.at("x"));
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
  DM M0 = DM::zeros(ng, ng); // zero matrix at the bottom right


  SX KKTprimer = SX::horzcat({SX::vertcat({hess1, grad}), SX::vertcat({grad.T(), M0})});
  Function KKT("KKT",{x, lambda}, {KKTprimer});
  cout << KKT(prim_dual) << endl;

  //f_jac.init();
  //vector<double> d{0,0,0};
  //f_jac.setInput(d);
  //f_jac.evaluate();
  //cout << "Function value in x = " << d << " : " << f_jac.output() << "\n";







    return 0;
}
