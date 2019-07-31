//
// Created by Joyce Yu on 7/22/19.
//


#include <iostream>
#include <fstream>
#include <casadi/casadi.hpp>
#include <casadi/core/sensitivity.hpp>
#include <casadi/core/timing.hpp>



using namespace casadi;
using namespace std;

int main() {

  // Pre-computed radau collocation matrix
  vector<vector<double>> omega(3);
  omega[0] = { 0.19681547722366,  0.39442431473909, 0.37640306270047};
  omega[1] = {-0.06553542585020,  0.29207341166523, 0.51248582618842};
  omega[2] = { 0.02377097434822, -0.04154875212600, 0.11111111111111};


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


  /// Model building

  // Time horizon
  // double T = 0.2;
  // Control discretization
  int N = 25; // number of control intervals
  double h = 10;   // step size  T/N


  // Declare model variables
  MX z10 = MX::sym("z10", N+1);
  MX z20 = MX::sym("z20", N+1);
  MX z30 = MX::sym("z30", N+1);
  MX z40 = MX::sym("z40", N+1);

  MX z1  = MX::sym("z1", N+1, d);
  MX z2  = MX::sym("z2", N+1, d);
  MX z3  = MX::sym("z3", N+1, d);
  MX z4  = MX::sym("z4", N+1, d);

  MX z1dot  = MX::sym("dz1", N+1, d);
  MX z2dot  = MX::sym("dz2", N+1, d);
  MX z3dot  = MX::sym("dz3", N+1, d);
  MX z4dot  = MX::sym("dz4", N+1, d);

  MX v1  = MX::sym("v1", N+1);
  MX v2  = MX::sym("v2", N+1);


  // Declare model parameters (fixed) and fixed bounds value
  double g        = 981;
  double T0       = 0;
  vector<double> smalla{0.233, 0.242, 0.127, 0.127};
  vector<double> bigA{50.27, 50.27, 28.27, 28.27};
  vector<double> xss{14, 14, 14.2, 21.3};
  vector<double> uss{43.4, 35.4};

  vector<double> xmin{-6.5, -6.5, -10.7, -16.8};
  vector<double> xmax{14, 14, 13.8, 6.7};
  vector<double> umin{-43.4, -35.4};
  vector<double> umax{16.6, 24.6};

  double gamma1 = 0.4;
  double gamma2 = 0.4;
  double u1init = -43.4;
  double u2init = -35.4;

  // start with an empty NLP
  vector<double> w0, lbw, ubw, lbg, ubg; // w0 is the initial guess
  vector<MX> w, g;
  MX Cost = 0;  // cost function


  /// Formulate the NLP
  for (int k = 0; k < N; ++k) {
    for (int j = 0; j < d; ++j) {
      // z1dot
      g.push_back(z1dot - (-(smalla[0]/bigA[0])*sqrt(2*g*(z1[i,j]+xss[0])) +
      (smalla[2]/bigA[0])*sqrt(2*g*(z3[i,j]+xss[2])) + (gamma1/bigA[0]*(v1[i]+uss[0]))
                          ));
      lbg.push_back(0);
      ubg.push_back(0);
      // z2dot
      g.push_back(z2dot - (-(smalla[1]/bigA[1])*sqrt(2*g*(z2[i,j]+xss[1])) +
                           (smalla[3]/bigA[1])*sqrt(2*g*(z4[i,j]+xss[3])) + (gamma2/bigA[1]*(v2[i]+uss[1]))
      ));
      lbg.push_back(0);
      ubg.push_back(0);
      // z3dot
      g.push_back(z3dot - (-(smalla[2]/bigA[2])*sqrt(2*g*(z3[i,j]+xss[2])) +
                            ((1-gamma2)/bigA[2]*(v2[i]+uss[1]))
      ));
      lbg.push_back(0);
      ubg.push_back(0);
      // z4dot
      g.push_back(z4dot - (-(smalla[3]/bigA[3])*sqrt(2*g*(z4[i,j]+xss[3])) +
                           ((1-gamma1)/bigA[3]*(v1[i]+uss[0]))
      ));
      lbg.push_back(0);
      ubg.push_back(0);

      g.push_back(z1[i,j] - (z10[i] + h*(omega[0,j]*z1dot[i,0] + omega[1,j]*z1dot[i,1] + omega[2,j]*z1dot[i,2])));
      lbg.push_back(0);
      ubg.push_back(0);

      g.push_back(z2[i,j] - (z20[i] + h*(omega[0,j]*z2dot[i,0] + omega[1,j]*z2dot[i,1] + omega[2,j]*z2dot[i,2])));
      lbg.push_back(0);
      ubg.push_back(0);

      g.push_back(z3[i,j] - (z30[i] + h*(omega[0,j]*z3dot[i,0] + omega[1,j]*z3dot[i,1] + omega[2,j]*z3dot[i,2])));
      lbg.push_back(0);
      ubg.push_back(0);

      g.push_back(z4[i,j] - (z40[i] + h*(omega[0,j]*z4dot[i,0] + omega[1,j]*z4dot[i,1] + omega[2,j]*z4dot[i,2])));
      lbg.push_back(0);
      ubg.push_back(0);

    }
    if (i > 0) {
      g.push_back(z10[i] - z1[i-1, d]);
      lbg.push_back(0);
      ubg.push_back(0);

      g.push_back(z20[i] - z2[i-1, d]);
      lbg.push_back(0);
      ubg.push_back(0);

      g.push_back(z30[i] - z3[i-1, d]);
      lbg.push_back(0);
      ubg.push_back(0);

      g.push_back(z40[i] - z4[i-1, d]);
      lbg.push_back(0);
      ubg.push_back(0);
    }

  }














  //MX w   = MX::vert


  MX CA = MX::sym("CA", N+1);
  MX CB = MX::sym("CB", N+1);
  MX TR = MX::sym("TR", N+1);
  MX TK = MX::sym("TK", N+1);
  MX x  = MX::vertcat({CA, CB, TR, TK});

  MX F  = MX::sym("F", N+1);
  MX QK = MX::sym("QK", N+1);
  MX u  = MX::vertcat({F, QK});

  MX CAin = MX::sym("CAin");
  MX EA3R = MX::sym("EA3R");
  MX p  = MX::vertcat({CAin, EA3R});

  //TODO: the below definitions of nx and nu are updated
  int nx = x.size1()/(N+1);
  int nu = u.size1()/(N+1);
  int np = p.size1();

  // Declare model parameters (fixed) and fixed bounds value

  double g        = 981;
  double T0       = 0;


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
  double EA3R_lo  = EA3R_nom * (1 - 0.01);


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
  double CAin_lo  = CAin_nom * (1 - 0.01);


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
  vector<double> p1  = {CAin_lo, EA3R_lo};



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
  /*
  MX xdot = vertcat(
  F * (CAin - CA) - k01 * exp( -EA1R / (TR + absT) ) * CA - k03 * exp( -EA3R / (TR + absT) ) * CA*CA,
  -F * CB          + k01 * exp( -EA1R / (TR + absT) ) * CA - k02 * exp( -EA2R / (TR + absT) ) * CB,
  F * (Tin - TR)  + kW*Area/(rho*Cp*VR)*(TK-TR) -
  (k01 * exp( -EA1R / (TR + absT) )*CA*delHAB + k02 * exp( -EA2R / (TR + absT) )*CB*delHBC + k03 * exp( -EA3R / (TR + absT) )*CA*CA*delHAD)/(rho*Cp),
  1 / (mk*CpK)    * (QK + kW*Area*(TR-TK))
  );
  */
  // initialize u_prev values
  MX F_prev  = MX::sym("F_prev");
  MX QK_prev = MX::sym("QK_prev");



  // Objective
  MX L = (CB - CBref) * (CB - CBref) + r1 *(F-F_prev)*(F-F_prev) + r2 *(QK-QK_prev)*(QK-QK_prev);
  // MX L = (CB - CBref) * (CB - CBref) + r1*F*F + r2*QK*QK;

  // Continuous time dynamics
  Function f("f", {x, u, F_prev, QK_prev}, {xdot, L});
  /*
  vector<double> xx = {0.8, 0.5, 135, 134};
  vector<double> uu = {5, 0};


  vector<MX> xxx{xx, uu, 10, -100};
  vector<MX> fff = f(xxx);
  MX fj = fff[0];
  MX qj = fff[1];
  cout << "fj = " << fj << endl;
  cout << "qj = " << qj << endl;

  */


  // have to do the initialization _after_ constructing Function f
  F_prev = Finit;
  QK_prev = QKinit;


  // start with an empty NLP
  vector<double> w0, lbw, ubw, lbg, ubg; // w0 is the initial guess
  vector<MX> w, g;
  MX Cost = 0;  // cost function

  // State at collocation points
  vector<MX> Xkj(d);

  // "lift" initial conditions
  MX Xk = MX::sym("x0", nx);
  w.push_back(Xk);
  g.push_back(Xk - xinit);
  for (int iw = 0; iw < nx; ++iw) {
    //lbw.push_back(xinit[iw]);
    //ubw.push_back(xinit[iw]);
    lbw.push_back(xmin[iw]);
    ubw.push_back(xmax[iw]);
    lbg.push_back(0);
    ubg.push_back(0);
    w0.push_back(xinit[iw]);
  }


  /// Formulate the NLP
  for (int k = 0; k < N; ++k) {
    // New NLP variable for the control
    MX Uk = MX::sym("U_" + str(k), nu);
    w.push_back(Uk);
    for (int iu = 0; iu < nu; ++iu) {
      lbw.push_back(umin[iu]);
      ubw.push_back(umax[iu]);
      w0.push_back(uinit[iu]);
    }

    // State at collocation points
    // vector<MX> Xkj(d);
    for (int j = 0; j < d; ++j) {
      Xkj[j] = MX::sym("X_" + str(k) + "_" + str(j+1), nx);
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
      vector<MX> XU{Xkj[j], Uk, F_prev, QK_prev};
      //vector<MX> XU{Xkj[j], Uk};
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
      Cost += B[j+1]*qj*h;
    }




    // New NLP variable for state at end of interval
    Xk = MX::sym("X_" + str(k+1), nx);
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

    // update the previous u
    /*
    MX u_prev = MX::sym("Uprev_" + str(k), nu);
    u_prev = Uk;
    vector<MX> u_prev_split = vertsplit(u_prev);
    F_prev  = u_prev_split[0];
    QK_prev = u_prev_split[1];
    // cout << F_prev << endl;
    */
    vector<MX> u_prev = vertsplit(Uk);
    F_prev  = u_prev[0];
    QK_prev = u_prev[1];



  }

  //cout << "lbw = " << lbw << endl;
  //cout << "ubw = " << ubw << endl;
  //cout << "lbg = " << lbg << endl;
  //cout << "ubg = " << ubg << endl;


  //cout << "w = " << MX::vertcat(w) << endl;
  //cout << "g = " << MX::vertcat(g) << endl;
  cout << "w size = " << w.size() << endl;
  cout << "w size = " << MX::vertcat(w).size() << endl;
  cout << "lbw size = " << lbw.size() << endl;
  cout << "ubw size = " << ubw.size() << endl;
  cout << "lbg size = " << lbg.size() << endl;
  cout << "ubg size = " << ubg.size() << endl;
  cout << "g size = " << MX::vertcat(g).size() << endl;


  MX variables   = MX::vertcat(w);
  MX constraints = MX::vertcat(g);


  // print cost function
  // cout << "cost function = " << Cost << endl;



  /// Create an NLP solver
  MXDict nlp = {
  {"x", variables},
  {"p", p},
  {"f", Cost},
  {"g", constraints}};

  Dict opts;
  //opts["verbose_init"] = true;
  opts["ipopt.linear_solver"] = "ma27";
  opts["ipopt.print_info_string"] = "yes";
  //opts["ipopt.print_level"] = 12;
  opts["ipopt.linear_system_scaling"] = "none";
  //opts["ipopt.fixed_variable_treatment"] = "make_constraint";
  //opts["ipopt.fixed_variable_treatment"] = "relax_bounds";

  Function solver = nlpsol("solver", "ipopt", nlp, opts);
  //MX hess_l = solver.get_function("nlp_hess_l");
  //cout << "ipopt hessian = " << hess_l << endl;
  //cout << "ipopt hessian = " << solver.get_function("nlp_hess_l") << endl;
  //cout << "print solver status" << solver.stats() << endl;
  std::map<std::string, DM> arg;


  // Solve the NLP
  arg["lbx"] = lbw;
  arg["ubx"] = ubw;
  arg["lbg"] = lbg;
  arg["ubg"] = ubg;
  arg["x0"]  = w0;
  arg["p"]   = p0;
  // arg["p"]   = {0, 0};

  /// keep record of timing
  FStats time;
  time.tic();
  auto res = solver(arg);
  time.toc();
  cout << "nlp t_wall time = " << time.t_wall << endl;
  cout << "nlp t_proc time = " << time.t_proc << endl;


  // cout << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1) << "res = " << evalf(res["x"]) << endl;



  int N_tot = res.at("x").size1();
  auto CA_opt = res.at("x")(Slice(0, N_tot, nu+nx+nx*d));
  DM CB_opt = res.at("x")(Slice(1, N_tot, nu+nx+nx*d));
  DM TR_opt = res.at("x")(Slice(2, N_tot, nu+nx+nx*d));
  DM TK_opt = res.at("x")(Slice(3, N_tot, nu+nx+nx*d));

  DM F_opt  = res.at("x")(Slice(4, N_tot, nu+nx+nx*d));
  DM QK_opt = res.at("x")(Slice(5, N_tot, nu+nx+nx*d));


  // cout << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1) << evalf(CA_opt)<< endl;


  // Print the solution
  cout << "-----" << endl;
  cout << "Optimal solution for p = " << arg.at("p") << ":" << endl;
  cout << setw(30) << "Objective: "   << res.at("f") << endl;

  cout << setw(30) << "Primal solution (CA): " << CA_opt << endl;
  cout << setw(30) << "Primal solution (CB): " << CB_opt << endl;
  cout << setw(30) << "Primal solution (TR): " << TR_opt << endl;
  cout << setw(30) << "Primal solution (TK): " << TK_opt << endl;
  cout << setw(30) << "Primal solution (F):  " << F_opt  << endl;
  cout << setw(30) << "Primal solution (QK): " << QK_opt << endl;


  /*
  cout << setw(30) << "x  solution     : [ ";
  for (int i=0; i<res.at("x").size1(); ++i) {
    cout  << setprecision(20) << double(res.at("x")(i)) << "  ";
  }
  cout << "]" << endl;


  cout << setw(30) << "Primal solution (CA): [";

  for (int i=0; i<CA_opt.size1(); ++i) {
    //cout  << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1)  << evalf(CA_opt(i))<< "  ";
    cout  << setprecision(20) << double(CA_opt(i)) << "  ";
  }
  cout << "]" << endl;

  cout << setw(30) << "Primal solution (CB): [" ;
  for (int i=0; i<CB_opt.size1(); ++i) {
    cout  << setprecision(20) << double(CB_opt(i)) << "  ";
  }
  cout << "]" << endl;


  cout << setw(30) << "Primal solution (TR): [" ;
  for (int i=0; i<TR_opt.size1(); ++i) {
    cout  << setprecision(20) << double(TR_opt(i)) << "  ";
  }
  cout << "]" << endl;

  cout << setw(30) << "Primal solution (TK): [";
  for (int i=0; i<TK_opt.size1(); ++i) {
    cout  << setprecision(20) << double(TK_opt(i)) << "  ";
  }
  cout << "]" << endl;

  cout << setw(30) << "Primal solution (F): [";
  for (int i=0; i<F_opt.size1(); ++i) {
    cout  << setprecision(20) << double(F_opt(i)) << "  ";
  }
  cout << "]" << endl;


  cout << setw(30) << "Primal solution (QK): [";
  for (int i=0; i<QK_opt.size1(); ++i) {
    cout  << setprecision(20) << double(QK_opt(i)) << "  ";
  }
  cout << "]" << endl;




  cout << setw(30) << "lam_g  solution     : [ ";
  for (int i=0; i<res.at("lam_g").size1(); ++i) {
    cout  << setprecision(20) << double(res.at("lam_g")(i)) << "  ";
  }
  cout << "]" << endl;

  cout << setw(30) << "lam_x  solution     : [ ";
  for (int i=0; i<res.at("lam_x").size1(); ++i) {
    cout  << setprecision(20) << double(res.at("lam_x")(i)) << "  ";
  }
  cout << "]" << endl;

  */



  //cout << setw(30) << "lam_g  solution     : " << res.at("lam_g") << endl;
  //cout << setw(30) << "lam_x  solution     : " << res.at("lam_x") << endl;





  //vector<DM> input_hess{res.at("x"), p1,  res.at("lam_g")};
  //cout << "ipopt hessian = " << solver.get_function("nlp_hess_l")(input_hess) << endl;
  // nlp_hess_l:(x[58],p[2],lam_f,lam_g[48])->(hess_gamma_x_x[58x58,82nz]) MXFunction
  // nlp_grad_f:(x[58],p[2])->(f,grad_f_x[58]) MXFunction


  vector<DM> input_grad_f{res.at("x"), p1};
  // cout << "ipopt nlp_grad_f = " << solver.get_function("nlp_grad_f")(input_grad_f) << endl;
  DM grad_f = DM::vertcat({solver.get_function("nlp_grad_f")(input_grad_f)[0]});
  //vector<DM> input_hess{res.at("x"), p1, grad_f, res.at("lam_g")};
  vector<DM> input_hess{res.at("x"), p1, res.at("f"), res.at("lam_g")};
  cout << "ipopt hessian = " << solver.get_function("nlp_hess_l")(input_hess) << endl;

  //cout << "ipopt interface lagrangian" << solver.get_function("hess_lag")(input_hess) << endl;


  ///****************************************************
  /// Sensitivity calculation


  int ng = MX::vertcat(g).size1();   // ng = number of constraints g
  int nw = MX::vertcat(w).size1();  // nw = number of variables x


  //DM ds = NLPsensitivity("csparse", res, Cost, constraints, variables, p, p0, p1);
  DM ds = NLPsensitivity_p("csparse", res, Cost, constraints, variables, p, p0, p1);
  DM s  = DM::vertcat({res.at("x"), res.at("lam_g"), res.at("lam_x")});
  DM s1 = s + ds;
  // int s_tot = s1.size1();
  // cout << "s vector dimension = " << s_tot << endl;
  // cout << "x vector dimension = " << N_tot << endl;

  // cout << "ds = " << ds(Slice(0, N_tot)) << endl;

  DM CA_pert = s1(Slice(0, N_tot, nu+nx+nx*d));
  DM CB_pert = s1(Slice(1, N_tot, nu+nx+nx*d));
  DM TR_pert = s1(Slice(2, N_tot, nu+nx+nx*d));
  DM TK_pert = s1(Slice(3, N_tot, nu+nx+nx*d));

  DM F_pert  = s1(Slice(4, N_tot, nu+nx+nx*d));
  DM QK_pert = s1(Slice(5, N_tot, nu+nx+nx*d));


  DM CA_ds = ds(Slice(0, N_tot, nu+nx+nx*d));
  DM CB_ds = ds(Slice(1, N_tot, nu+nx+nx*d));
  DM TR_ds = ds(Slice(2, N_tot, nu+nx+nx*d));
  DM TK_ds = ds(Slice(3, N_tot, nu+nx+nx*d));

  DM F_ds  = ds(Slice(4, N_tot, nu+nx+nx*d));
  DM QK_ds = ds(Slice(5, N_tot, nu+nx+nx*d));



  cout << setw(30) << "ds(CA): " << CA_ds << endl;
  cout << setw(30) << "ds(CB): " << CB_ds << endl;
  cout << setw(30) << "ds(TR): " << TR_ds << endl;
  cout << setw(30) << "ds(TK): " << TK_ds << endl;
  cout << setw(30) << "ds(F):  " << F_ds  << endl;
  cout << setw(30) << "ds(QK): " << QK_ds << endl;



  cout << setw(30) << "Perturbed solution (CA): " << CA_pert << endl;
  cout << setw(30) << "Perturbed solution (CB): " << CB_pert << endl;
  cout << setw(30) << "Perturbed solution (TR): " << TR_pert << endl;
  cout << setw(30) << "Perturbed solution (TK): " << TK_pert << endl;
  cout << setw(30) << "Perturbed solution (F):  " << F_pert  << endl;
  cout << setw(30) << "Perturbed solution (QK): " << QK_pert << endl;



  arg["p"]   = p1;
  res = solver(arg);

  // Print the new solution
  cout << "-----" << endl;
  cout << "Optimal solution for p = " << arg.at("p") << ":" << endl;
  cout << setw(30) << "Objective: " << res.at("f") << endl;
  //cout << setw(30) << "Primal solution: " << res.at("x") << endl;
  //cout << setw(30) << "Dual solution (x): " << res.at("lam_x") << endl;
  //cout << setw(30) << "Dual solution (g): " << res.at("lam_g") << endl;


  DM CA_optm = res.at("x")(Slice(0, N_tot, nu+nx+nx*d));
  DM CB_optm = res.at("x")(Slice(1, N_tot, nu+nx+nx*d));
  DM TR_optm = res.at("x")(Slice(2, N_tot, nu+nx+nx*d));
  DM TK_optm = res.at("x")(Slice(3, N_tot, nu+nx+nx*d));

  DM F_optm  = res.at("x")(Slice(4, N_tot, nu+nx+nx*d));
  DM QK_optm = res.at("x")(Slice(5, N_tot, nu+nx+nx*d));



  cout << setw(30) << "Optimal solution for p1 (CA): " << CA_optm << endl;
  cout << setw(30) << "Optimal solution for p1 (CB): " << CB_optm << endl;
  cout << setw(30) << "Optimal solution for p1 (TR): " << TR_optm << endl;
  cout << setw(30) << "Optimal solution for p1 (TK): " << TK_optm << endl;
  cout << setw(30) << "Optimal solution for p1 (F):  " << F_optm  << endl;
  cout << setw(30) << "Optimal solution for p1 (QK): " << QK_optm << endl;



  return 0;

}

