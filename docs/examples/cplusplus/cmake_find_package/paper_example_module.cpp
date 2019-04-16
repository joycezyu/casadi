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


using namespace casadi;
using namespace std;

int main() {
  /**
   *  Examples for Pirnay et al. (2012) sIPOPT paper
   *
   *    min       x₁² + x₂² + x₃²
   *    s.t.     6x₁ + 3x₂ + 2x₃ - p₁ = 0
   *            p₂x₁ +  x₂ -  x₃ -  1 = 0
   *              x₁,   x₂,   x₃     >= 0
   *
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
  vector<double> lbx = {   0,    0,    0};
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




  /// Sensitivity calculation
  DM ds = NLPsensitivity(res, f, g, x, p, p0, p1);
  DM s  = DM::vertcat({res.at("x"), res.at("lam_g"), res.at("lam_x")});
  DM s1 = s + ds;

  // print the solution for sensitivity calculation
  cout << "ds = " << ds << endl;
  cout << "s1 = " << s1 << endl;


  return 0;
}
