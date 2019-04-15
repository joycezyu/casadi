//
// Created by Zhou Yu on 4/5/19.
//



#include <casadi/casadi.hpp>



using namespace casadi;
using namespace std;

int main() {

  // Degree of interpolating polynomial
  int d = 3;

  // Choose collocation points
  vector<double> tau_root = collocation_points(d, "radau");   // "legendre"
  tau_root.insert(tau_root.begin(), 0);


  // Coefficients of the collocation equation
  vector<vector<double> > C(d+1,vector<double>(d+1,0));

  // Coefficients of the continuity equation
  vector<double> D(d+1,0);

  // Coefficients of the quadrature function
  vector<double> B(d+1,0);

  // For all collocation points
  for(int j=0; j<d+1; ++j) {

    // Construct Lagrange polynomials to get the polynomial basis at the collocation point
    Polynomial p = 1;
    for (int r = 0; r < d + 1; ++r) {
      if (r != j) {
        p *= Polynomial(-tau_root[r], 1) / (tau_root[j] - tau_root[r]);
      }
    }

    // Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    D[j] = p(1.0);

    // Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    Polynomial dp = p.derivative();
    for (int r = 0; r < d + 1; ++r) {
      C[j][r] = dp(tau_root[r]);
    }

    Polynomial pint = p.anti_derivative();
    B[j] = pint(1.0);

  }

  cout << "C = " << C << endl;
  cout << "D = " << D << endl;
  cout << "B = " << B << endl;



  /// Model building

  // Time horizon
  double T = 10.0;

  // Declare model variables

  MX x1 = MX::sym("x1");
  MX x2 = MX::sym("x2");
  MX x = MX::vertcat({x1,x2});
  MX u = MX::sym("u");

  int nx = x.size1();
  int nu = u.size1();

  // model equations
  MX xdot = vertcat(
            (1 - x2*x2)*x1 - x2 + u,
            x1
  );

  // Objective
  MX L = x1*x1 + x2*x2 + u*u;

  // Continuous time dynamics
  Function f("f", {x, u}, {xdot, L});

  // Control discretization
  int N = 20; // number of control intervals
  double h = T/N;   // step size

  // start with an empty NLP
  vector<double> w0, lbw, ubw, lbg, ubg; // w0 is the initial guess
  vector<MX> w, g;
  MX J = 0;  // cost function

  // State at collocation points
  vector<MX> Xkj(d);

  // "lift" initial conditions
  MX Xk = MX::sym("x0", nx);
  w.push_back(Xk);
  lbw.push_back(0);
  lbw.push_back(1);
  ubw.push_back(0);
  ubw.push_back(1);
  w0.push_back(0);
  w0.push_back(1);



  /// Formulate the NLP
  for (int k = 0; k < N; ++k) {
    // New NLP variable for the control
    MX Uk = MX::sym("U_" + std::to_string(k));
    w.push_back(Uk);
    lbw.push_back(-1);
    ubw.push_back(1);
    w0.push_back(0);

    // State at collocation points
    for (int j = 0; j < d; ++j) {
      Xkj[j] = MX::sym("X_" + std::to_string(k) + "_" + std::to_string(j+1), nx);
      w.push_back(Xkj[j]);
      lbw.push_back(-0.25);
      lbw.push_back(-inf);
      ubw.push_back(inf);
      ubw.push_back(inf);
      w0.push_back(0);
      w0.push_back(0);
    }



    // Loop over collocation points
    MX Xk_end = D[0]*Xk;

    for (int j = 0; j < d; ++j) {
      // Expression for the state derivative at the collocation point
      MX xp = C[0][j+1] * Xk;

      for (int r = 0; r < d; ++r) {
        xp += C[r+1][j+1] * Xkj[r];
      }

      // Append collocation equations
      vector<MX> XU{Xkj[j], Uk};
      vector<MX> fL = f(XU);
      MX fj = fL[0];
      MX qj = fL[1];
      cout << "fj = " << fj << endl;
      cout << "qj = " << qj << endl;

      g.push_back(h*fj - xp);
      lbg.push_back(0);  lbg.push_back(0);
      ubg.push_back(0);  ubg.push_back(0);

      // Add contribution to the end state
      Xk_end += D[j+1]*Xkj[j];

      // Add contribution to quadrature function
      J += B[j+1]*qj*h;
      cout << "J = " << J << endl;
    }


    // New NLP variable for state at end of interval
    Xk = MX::sym("X_" + std::to_string(k+1), nx);
    w.push_back(Xk);
    lbw.push_back(-0.25);   lbw.push_back(-inf);
    ubw.push_back(inf);     ubw.push_back(inf);
    w0.push_back(0);        w0.push_back(0);

    // Add equality constraint
    // for continuity between intervals
    g.push_back(Xk_end - Xk);
    lbg.push_back(0);       lbg.push_back(0);
    ubg.push_back(0);       ubg.push_back(0);

  }


  /*
  cout << "w size = " << w.size() << endl;
  cout << "w size = " << MX::vertcat(w).size() << endl;
  cout << "lbw size = " << lbw.size() << endl;
  cout << "ubw size = " << ubw.size() << endl;
  cout << "lbg size = " << lbg.size() << endl;
  cout << "ubg size = " << ubg.size() << endl;
  cout << "g size = " << MX::vertcat(g).size() << endl;
  */

  /// Create an NLP solver
  MXDict nlp = {
                {"x", MX::vertcat(w)},
                {"f", J},
                {"g", MX::vertcat(g)}};

  Function solver = nlpsol("solver", "ipopt", nlp);
  std::map<std::string, DM> arg, res;


  // Solve the NLP
  arg["lbx"] = lbw;
  arg["ubx"] = ubw;
  arg["lbg"] = lbg;
  arg["ubg"] = ubg;
  arg["x0"] = w0;
  res = solver(arg);

  int N_tot = res.at("x").size1();
  DM x1_opt = res.at("x")(Slice(0, N_tot, nu+nx+nx*d));
  DM x2_opt = res.at("x")(Slice(1, N_tot, nu+nx+nx*d));
  DM u_opt = res.at("x")(Slice(2, N_tot, nu+nx+nx*d));


  // Print the solution
  cout << "-----" << endl;
  cout << "Optimal solution" << endl;
  cout << setw(30) << "Objective: " << res.at("f") << endl;
  cout << setw(30) << "Primal solution (x1): " << x1_opt << endl;
  cout << setw(30) << "Primal solution (x2): " << x2_opt << endl;
  cout << setw(30) << "Primal solution (u): "  << u_opt << endl;



  return 0;

}

