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
  double T = 0.2;
  // Control discretization
  int N = 40; // number of control intervals
  double h = T/N;   // step size


  // Declare model variables

  MX CA = MX::sym("CA");
  MX CB = MX::sym("CB");
  MX TR = MX::sym("TR");
  MX TK = MX::sym("TK");
  MX x  = MX::vertcat({CA, CB, TR, TK});

  MX F  = MX::sym("F");
  MX QK = MX::sym("QK");
  MX u  = MX::vertcat({F, QK});

  MX CAin = MX::sym("CAin");
  MX EA3R = MX::sym("EA3R");
  MX p  = MX::vertcat({CAin, EA3R});

  int nx = x.size1();
  int nu = u.size1();
  int np = p.size1();

  // Declare model parameters (fixed) and fixed bounds value
  double CAinit   = 0.8;
  double CBinit   = 0.5;
  double TRinit   = 134.14;
  double TKinit   = 134.0;
  double T0       = 0;

  double Finit    = 18.83;
  double QKinit   = -4495.7;

  double r1       = 1e-7;
  double r2       = 1e-11;

  double CAmin    = 0.1;
  double CAmax    = 1;
  double CBmin    = 0.1;
  double CBmax    = 1;
  double TRmin    = 50;
  double TRmax    = 140;
  double TKmin    = 50;
  double TKmax    = 180;

  double Fmin     = 5;
  double Fmax     = 100;
  double QKmin    = -8500;
  double QKmax    = 0;

  double k01      = 1.287e12;
  double k02      = 1.287e12;
  double k03      = 9.043e9;
  double EA1R     = 9758.3;
  double EA2R     = 9758.3;
  double EA3R_nom = 8560;

  double delHAB   = 4.2;
  double delHBC   = -11;
  double delHAD   = -41.85;
  double rho      = 0.9342;
  double Cp       = 3.01;
  double CpK      = 2.0;
  double Area     = 0.215;
  double VR       = 10.01;
  double mk       = 5.0;
  double Tin      = 130.0;
  double kW       = 4032;

  double CAin_nom = 5.1;
  double CAin_lo  = CAin_nom * (1 - 0.1);


  double CBref    = 0.5;


  vector<double> xinit{CAinit, CBinit, TRinit, TKinit};
  vector<double> uinit{Finit, QKinit};
  vector<double> xmin{CAmin, CBmin, TRmin, TKmin};
  vector<double> xmax{CAmax, CBmax, TRmax, TKmax};
  vector<double> umin{Fmin,  QKmin};
  vector<double> umax{Fmax,  QKmax};






  // Original parameter values
  vector<double> p0  = {CAin_nom, EA3R_nom};
  // new parameter values
  vector<double> p1  = {CAin_lo, EA3R_nom};



  double absT = 273.15;
  MX k1 = k01 * exp( -EA1R / (TR + absT) );
  MX k2 = k02 * exp( -EA2R / (TR + absT) );
  MX k3 = k03 * exp( -EA3R / (TR + absT) );





  // model equations
  MX xdot = vertcat(
     F * (CAin - CA) - k1 * CA - k3 * CA*CA,
    -F * CB          + k1 * CA - k2 * CB,
     F * (Tin - TR)  + kW*Area/(rho*Cp*VR)*(TK-TR) - (k1*CA*delHAB + k2*CB*delHBC + k3*CA*CA*delHAD)/(rho*Cp),
     1 / (mk*CpK)    * (QK + kW*Area*(TR-TK))
  );

  // Objective
  MX L = (CB - CBref) * (CB - CBref) + r1*F*F + r2*QK*QK;

  // Continuous time dynamics
  Function f("f", {x, u}, {xdot, L});



  // start with an empty NLP
  vector<double> w0, lbw, ubw, lbg, ubg; // w0 is the initial guess
  vector<MX> w, g;
  MX J = 0;  // cost function

  // State at collocation points
  vector<MX> Xkj(d);

  // "lift" initial conditions
  MX Xk = MX::sym("x0", nx);
  w.push_back(Xk);

  for (int iw = 0; iw < nx; ++iw) {
    lbw.push_back(xinit[iw]);
    ubw.push_back(xinit[iw]);
    w0.push_back(xinit[iw]);
  }



  /// Formulate the NLP
  for (int k = 0; k < N; ++k) {
    // New NLP variable for the control
    MX Uk = MX::sym("U_" + std::to_string(k), nu);
    w.push_back(Uk);
    for (int iu = 0; iu < nu; ++iu) {
      lbw.push_back(umin[iu]);
      ubw.push_back(umax[iu]);
      w0.push_back(uinit[iu]);
    }

    // State at collocation points
    for (int j = 0; j < d; ++j) {
      Xkj[j] = MX::sym("X_" + std::to_string(k) + "_" + std::to_string(j+1), nx);
      w.push_back(Xkj[j]);
      for (int iw = 0; iw < nx; ++iw) {
        lbw.push_back(xmin[iw]);
        ubw.push_back(xmax[iw]);
        w0.push_back(xinit[iw]);
      }
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

      g.push_back(h*fj - xp);
      for (int iw = 0; iw < nx; ++iw) {
        lbg.push_back(0);
        ubg.push_back(0);
      }

      // Add contribution to the end state
      Xk_end += D[j+1]*Xkj[j];

      // Add contribution to quadrature function
      J += B[j+1]*qj*h;
    }


    // New NLP variable for state at end of interval
    Xk = MX::sym("X_" + std::to_string(k+1), nx);
    w.push_back(Xk);
    for (int iw = 0; iw < nx; ++iw) {
      lbw.push_back(xmin[iw]);
      ubw.push_back(xmax[iw]);
      w0.push_back(xinit[iw]);
    }

    // Add equality constraint
    // for continuity between intervals
    g.push_back(Xk_end - Xk);
    for (int iw = 0; iw < nx; ++iw) {
      lbg.push_back(0);
      ubg.push_back(0);
    }

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
  {"p", p},
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
  DM CA_opt = res.at("x")(Slice(0, N_tot, nu+nx+nx*d));
  DM CB_opt = res.at("x")(Slice(1, N_tot, nu+nx+nx*d));
  DM TR_opt = res.at("x")(Slice(2, N_tot, nu+nx+nx*d));
  DM TK_opt = res.at("x")(Slice(3, N_tot, nu+nx+nx*d));

  DM F_opt = res.at("x")(Slice(4, N_tot, nu+nx+nx*d));
  DM QK_opt = res.at("x")(Slice(5, N_tot, nu+nx+nx*d));



  // Print the solution
  cout << "-----" << endl;
  cout << "Optimal solution" << endl;
  cout << setw(30) << "Objective: " << res.at("f") << endl;
  cout << setw(30) << "Primal solution (CA): " << CA_opt << endl;
  cout << setw(30) << "Primal solution (CB): " << CB_opt << endl;
  cout << setw(30) << "Primal solution (F): "  << F_opt << endl;





  return 0;

}
