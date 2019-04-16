//
// Created by Zhou Yu on 4/15/19.
//

#include "sensitivity.hpp"

using namespace std;
namespace casadi {

DM NLPsensitivity(std::map<std::string, DM>& res,
                  const MX& objective, const MX& constraints, const MX& variables, const MX& parameters,
                  std::vector<double>& p0, std::vector<double>& p1) {

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
  MX XiV    = mtimes(inv(X), V);
  //MX XiV    = MX::diag(dot(v,1/x));      // equivalent results, not sure about efficiency


  MX grad = jacobian(g, x);
  // construct the lagrangian function
  MX lagrangian = f + dot(lambda, g) - dot(v, x);
  MX jac_lagrangian = jacobian(lagrangian, x);
  MX hess = hessian(lagrangian, x);

  /// Assemble KKT matrix
  // sparse zero matrix
  MX M0 = MX(ng, ng);

  // LHS
  MX KKTprimer = MX::horzcat({MX::vertcat({hess + XiV, grad}),
                              MX::vertcat({grad.T(), M0})});

  // RHS
  MX phi = MX::vertcat({jac_lagrangian.T(), g});

  /// Solve linear system
  MX sensitivity = solve(KKTprimer, -phi);
  // can use the following sparse linear solvers if large-scale
  //MX sensitivity = solve(KKTprimer, -phi, "ma27");
  //MX sensitivity = solve(KKTprimer, -phi, "csparse");
  //MX sensitivity = solve(KKTprimer, -phi, "ldl");
  //MX sensitivity = solve(KKTprimer, -phi, "qr");

  // MX p  = MX::sym("p", np);


  Function sens_eval("sens", {x, lambda, v, p}, {sensitivity});
  vector<DM> prim_dual_param{res.at("x"), res.at("lam_g"), res.at("lam_x"), p1};
  // solution vector for 2x2 system is [Δx, Δλ]ᵀ
  DM dx_dl = DM::vertcat({sens_eval(prim_dual_param)});

  // compute Δν
  MX dx = MX::sym("dx", nx);
  MX dv = -mtimes(XiV, dx);
  Function dv_eval("dv", {x, v, dx}, {dv});
  vector<DM> x_v_dx{res.at("x"), res.at("lam_x"), dx_dl(Slice(0, nx))};
  DM dv0 = DM::vertcat({dv_eval(x_v_dx)});

  // assemble ds matrix
  DM ds = DM::vertcat({dx_dl, dv0});


  return ds;
};



}  // namespace casadi




