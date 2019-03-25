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

  auto opti = casadi::Opti();

  auto x1 = opti.variable();
  auto x2 = opti.variable();
  auto u = opti.variable();
  auto p = opti.parameter();
  opti.set_value(p, 1);

  opti.minimize(x1 * x1);
  opti.subject_to(x1 + u + p ==0);
  opti.subject_to(2*x2 + u - 0.5*p ==0);
  opti.subject_to(x1 - 1 <= 0);

  opti.solver("ipopt");
  auto sol = opti.solve();


  // Hessian of the Lagrangian
  /*
  Function grad_lag = opti.factory("grad_lag",
                                      {"x1", "p", "lam:f", "lam:g"}, {"grad:gamma:x"},
                                      {{"gamma", {"f", "g"}}});
  */
  // Hess = grad_lag.sparsity_jac("x", "grad_gamma_x", false, true);
  // create_function("nlp_gf_jg", {"x", "p"},
  //                {"f", "g", "grad:f:x", "jac:g:x"});

  // Function gf_jg = create_function("nlp_gf_jg", {"x", "p"},
  //                                 {"f", "g", "grad:f:x", "jac:g:x"});
  // A = gf_jg.sparsity_out("jac_g_x");

  // auto h = casadi::Nlp::getH(opti);
  
  std::cout << sol.value(x1) << ":" << sol.value(x2) << ":" << sol.value(u) << std::endl;

  return 0;
}
