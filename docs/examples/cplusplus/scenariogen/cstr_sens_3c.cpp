//
// Created by Zhou Yu on 4/5/19.
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
  double T = 0.2;
  // Control discretization
  int horN = 3; // number of control intervals
  double h = T/horN;   // step size
  //cout << "h = " << h << endl;

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
  double EA3R_lo  = EA3R_nom * (1 - 0.01);
  double EA3R_up  = EA3R_nom * (1 + 0.01);

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
  double CAin_up  = CAin_nom * (1 + 0.1);


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
  vector<double> p2  = {CAin_up, EA3R_nom};



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
  for (int k = 0; k < horN; ++k) {
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


  cout << "primal variables = " << variables << endl;

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




  vector<DM> input_grad_f{res.at("x"), p1};
  // cout << "ipopt nlp_grad_f = " << solver.get_function("nlp_grad_f")(input_grad_f) << endl;
  DM grad_f = DM::vertcat({solver.get_function("nlp_grad_f")(input_grad_f)[0]});
  //vector<DM> input_hess{res.at("x"), p1, grad_f, res.at("lam_g")};
  vector<DM> input_hess{res.at("x"), p1, res.at("f"), res.at("lam_g")};
  cout << "ipopt hessian = " << solver.get_function("nlp_hess_l")(input_hess) << endl;

  //cout << "ipopt interface lagrangian" << solver.get_function("hess_lag")(input_hess) << endl;


  ///****************************************************
  /// scenario generation with Schur-complement

  int nr = 1;

  //vector<int> NACindex(int nx, int nu, int nr) {
  vector<int> NAC_Ctrl_index;
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nu; ++j) {
      NAC_Ctrl_index.push_back(nx*(i+1) + nx*d*i + nu*i + j);
    };
  };
    //return res;
  //};

  /// NAC_Ctrl_index already considers for all the u's
  cout << "NAC index list = " << NAC_Ctrl_index << endl;


  /// scenario construction
  int length = lbw.size() + lbg.size();

  int ns = 3;
  int mNAC = 2;
  vector<DM> N(ns);
  N[0] = DM(length,mNAC);
  N[1] = DM(length,mNAC);
  N[2] = DM(length,mNAC);
  for (auto ind:NAC_Ctrl_index) {
    // scenario 1
    // need NACs for NAC constraint index 0 and 1
    // index 0 = NAC constraint 0, index 1 = NAC constraint 1
    N[0](ind, 0) = 1;
    N[0](ind, 1) = 1;
    // scenario 2 only for NAC index 0
    N[1](ind, 0) = -1;
    // scenario 3 only for NAC index 1
    N[2](ind, 1) = -1;

  }


  cout << "all N blocks = " << N << endl;

  ///start counting time
  //FStats NACtime;
  //NACtime.tic();

  /// construct the multiplier gamma vector
  DM gamma(mNAC,1);
  vector<vector<DM>> KR(ns);



  vector<DM> KR1 = getKKTaRHS(res, Cost, constraints, variables, p, p0, p0);
  vector<DM> KR2 = getKKTaRHS(res, Cost, constraints, variables, p, p0, p1);
  vector<DM> KR3 = getKKTaRHS(res, Cost, constraints, variables, p, p0, p2);

  vector<DM> K(ns), R(ns);
  K[0] = KR1[0];
  R[0] = KR1[1];
  K[1] = KR2[0];
  R[1] = KR2[1];
  K[2] = KR3[0];
  R[2] = KR3[1];


  ///start counting time
  FStats NACtime;
  NACtime.tic();

  // LHS
  vector<DM> KiN(ns);
  DM totLHS(mNAC, mNAC);
  for (int is = 0; is < ns; ++is) {
    KiN[is] = solve(K[is], N[is], "ma27");
    totLHS += mtimes(N[is].T(), KiN[is]);
  }

  //auto linear_solver = Linsol("linear_solver", "ma27", K1.sparsity());
  //linear_solver.sfact(K1);

  // RHS
  vector<DM> Kir(ns);
  DM totRHS(mNAC, 1);
  for (int is = 0; is < ns; ++is) {
    Kir[is] = solve(K[is], R[is], "ma27");
    totRHS += mtimes(N[is].T(), Kir[is]);
  }

  gamma = solve(totLHS, -totRHS, "ma27");

  cout << "gamma = " << gamma << endl;

  vector<DM> schurR(ns);
  vector<DM> ds(ns);
  for (int is = 0; is < ns; ++is) {
    schurR[is] = R[is] + mtimes(N[is], gamma);
    ds[is] = solve(K[is], -schurR[is], "ma27");
  }

  /*
  DM newR1 = R[0] + mtimes(N[0], gamma);
  DM ds1 = solve(K[0], -newR1, "ma27");
  cout << "ds1 = " << ds1 << endl;

  DM newR2 = R[1] + mtimes(N[1], gamma);
  DM ds2 = solve(K[1], -newR2, "ma27");
  cout << "ds2 = " << ds2 << endl;

  DM newR3 = R[2] + mtimes(N[2], gamma);
  DM ds3 = solve(K[2], -newR3, "ma27");
  cout << "ds3 = " << ds3 << endl;
  */

  /// end time
  NACtime.toc();
  cout << "total t_wall time = " << NACtime.t_wall << endl;
  cout << "total t_proc time = " << NACtime.t_proc << endl;


  DM s  = DM::vertcat({res.at("x"), res.at("lam_g")});
  cout << "nominal scenario optimal solution s = " << s << endl;
  DM s1 = s + ds[0];
  DM s2 = s + ds[1];
  DM s3 = s + ds[2];
  vector<DM> s_pert{s1, s2, s3};

  vector<DM> CA_pert(ns), CB_pert(ns), TR_pert(ns), TK_pert(ns), F_pert(ns), QK_pert(ns);
  vector<DM> CA_ds(ns), CB_ds(ns), TR_ds(ns), TK_ds(ns), F_ds(ns), QK_ds(ns);

  for (int is = 0; is < ns; ++is) {

    CA_pert[is] = s_pert[is](Slice(0, N_tot, nu+nx+nx*d));
    CB_pert[is] = s_pert[is](Slice(1, N_tot, nu+nx+nx*d));
    TR_pert[is] = s_pert[is](Slice(2, N_tot, nu+nx+nx*d));
    TK_pert[is] = s_pert[is](Slice(3, N_tot, nu+nx+nx*d));
    F_pert[is]  = s_pert[is](Slice(4, N_tot, nu+nx+nx*d));
    QK_pert[is] = s_pert[is](Slice(5, N_tot, nu+nx+nx*d));


    CA_ds[is] = ds[is](Slice(0, N_tot, nu+nx+nx*d));
    CB_ds[is] = ds[is](Slice(1, N_tot, nu+nx+nx*d));
    TR_ds[is] = ds[is](Slice(2, N_tot, nu+nx+nx*d));
    TK_ds[is] = ds[is](Slice(3, N_tot, nu+nx+nx*d));
    F_ds[is]  = ds[is](Slice(4, N_tot, nu+nx+nx*d));
    QK_ds[is] = ds[is](Slice(5, N_tot, nu+nx+nx*d));

    cout << setw(30) << " For scenario s = " << is << endl;
    cout << setw(30) << "ds(CA): " << CA_ds[is] << endl;
    cout << setw(30) << "ds(CB): " << CB_ds[is] << endl;
    cout << setw(30) << "ds(TR): " << TR_ds[is] << endl;
    cout << setw(30) << "ds(TK): " << TK_ds[is] << endl;
    cout << setw(30) << "ds(F):  " << F_ds[is]  << endl;
    cout << setw(30) << "ds(QK): " << QK_ds[is] << endl;



    cout << setw(30) << "Perturbed solution (CA): " << CA_pert[is] << endl;
    cout << setw(30) << "Perturbed solution (CB): " << CB_pert[is] << endl;
    cout << setw(30) << "Perturbed solution (TR): " << TR_pert[is] << endl;
    cout << setw(30) << "Perturbed solution (TK): " << TK_pert[is] << endl;
    cout << setw(30) << "Perturbed solution (F):  " << F_pert[is]  << endl;
    cout << setw(30) << "Perturbed solution (QK): " << QK_pert[is] << endl;

  }




  return 0;

}
