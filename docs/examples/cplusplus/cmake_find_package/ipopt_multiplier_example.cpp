//
// Created by Zhou Yu on 5/8/19.
//

#include <casadi/casadi.hpp>
#include <casadi/core/sensitivity.hpp>




using namespace casadi;
using namespace std;

int main() {
  /**
   *  adopted from
   *  https://github.com/casadi/casadi/issues/667
   *  https://list.coin-or.org/pipermail/ipopt/2013-April/003335.html
   *
   *
   *  Examples for nonunique multipliers to ipopt
   *    min      x₀(3 + x₁²)
   *    s.t.     x₀ = 1
   *            -10 ≤ x₁ ≤ 10
   *
   *   because the upper bound and lower bound for x₀ are the same
   *   ipopt (by default) ignores x₀ and the corresponding multiplier is 0
   *
   *
  */


  MX x0 = MX::sym("x0");
  MX x1 = MX::sym("x1");
  MX f = x0 * (3 + x1*x1);

  //MX g = x0 - 1;
  MX g = 0;

  MX x = MX::vertcat({x0, x1});
  MXDict nlp = {{"x", x}, {"g", g}, {"f", f}};
  //MXDict nlp = {{"x", x}, {"f", f}};



  vector<double> xinit = {-1,  -1};
  vector<double> lbx   = { 1, -10};
  vector<double> ubx   = { 1,  10};
  //vector<double> lbx   = { -inf, -10};
  //vector<double> ubx   = {  inf,  10};
  //vector<double> lbg   = { 0};
  //vector<double> ubg   = { 0};
  vector<double> lbg   = { -inf};
  vector<double> ubg   = { inf};


  Dict opts;
  //opts["ipopt.linear_solver"] = "ma57";
  opts["ipopt.linear_system_scaling"] = "none";
  //opts["ipopt.fixed_variable_treatment"] = "make_constraint";
  //opts["ipopt.fixed_variable_treatment"] = "relax_bounds";

  Function solver = nlpsol("solver", "ipopt", nlp, opts);
  std::map<std::string, DM> arg, res;


  // Solve the NLP
  arg["x0"]  = xinit;
  arg["lbx"] = lbx;
  arg["ubx"] = ubx;
  arg["lbg"] = lbg;
  arg["ubg"] = ubg;


  res = solver(arg);
  cout << res << endl;

  vector<DM> input_hess{res.at("x"), res.at("lam_p"), res.at("f"), res.at("lam_g")};
  cout << "ipopt hessian = " << solver.get_function("nlp_hess_l")(input_hess) << endl;

  MX p;
  vector<double> p0, p1;

  /// Sensitivity calculation
  DM ds = NLPsensitivity("ma27", res, f, g, x, p, p0, p1);
  DM s  = DM::vertcat({res.at("x"), res.at("lam_g"), res.at("lam_x")});
  DM s1 = s + ds;

  // print the solution for sensitivity calculation
  cout << "ds = " << ds << endl;
  cout << "s1 = " << s1 << endl;



  return 0;
}