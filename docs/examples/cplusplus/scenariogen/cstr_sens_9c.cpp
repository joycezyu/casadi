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

  cout << " B = " << B << endl;
  cout << " C = " << C << endl;
  cout << " D = " << D << endl;

  /// Step 1
  /// Nominal scenario
  /// Model building

  // Time horizon
  double T = 0.2;
  // Control discretization
  int horN = 1; // number of control intervals
  double h = T/horN;   // step size

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

  double r1       = 1e-7;   //1e-7
  double r2       = 1e-11;  //1e-11

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



  /// uncertain parameter distribution
  vector<double> p_CAin{CAin_nom, CAin_lo, CAin_up};
  vector<double> p_EA3R{EA3R_nom, EA3R_lo, EA3R_up};

  vector<vector<double>> p_c(p_EA3R.size() * p_CAin.size());

  for (int i = 0; i < p_CAin.size(); ++i) {
    for (int j = 0; j < p_EA3R.size(); ++j) {
      p_c[i * p_EA3R.size() + j] = {p_CAin[i], p_EA3R[j]};
    }
  }

  cout << "p_c = " << p_c << endl;


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

  // initialize u_prev values
  MX F_prev  = MX::sym("F_prev");
  MX QK_prev = MX::sym("QK_prev");



  // Objective
  MX L = (CB - CBref) * (CB - CBref) + r1 *(F-F_prev)*(F-F_prev) + r2 *(QK-QK_prev)*(QK-QK_prev);

  // Continuous time dynamics
  Function f_xdot("xdot", {x, u}, {xdot});
  Function f_L("L", {x, u, F_prev, QK_prev}, {L});



  // have to do the initialization _after_ constructing Function f
  F_prev = Finit;
  QK_prev = QKinit;


  // start with an empty NLP
  vector<double> w0, lbw, ubw, lbg, ubg; // w0 is the initial guess
  vector<MX> w, g;
  vector<MX> g_ineq;
  MX Cost = 0;  // cost function

  // State at collocation points
  //vector<MX> Xkj(d);



  /// keep track of variables
  vector<MX> Xk(horN+1);
  vector<MX> Uk(horN);
  vector<MX> Xk_end(horN);
  // vector<MX> Uk_prev(horN - 1);

  vector<vector<MX>> Xkj(horN);





  // "lift" initial conditions
  Xk[0] = MX::sym("x0", nx);
  w.push_back(Xk[0]);
  g.push_back(Xk[0] - xinit);
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
    Uk[k] = MX::sym("U_" + str(k), nu);
    w.push_back(Uk[k]);
    for (int iu = 0; iu < nu; ++iu) {
      lbw.push_back(umin[iu]);
      ubw.push_back(umax[iu]);
      w0.push_back(uinit[iu]);
    }

    // State at collocation points
    // vector<MX> Xkj(d);
    for (int j = 0; j < d; ++j) {
      Xkj[k].push_back(MX::sym("X_" + str(k) + "_" + str(j+1), nx));
      w.push_back(Xkj[k].back());
      for (int iw = 0; iw < nx; ++iw) {
        lbw.push_back(xmin[iw]);
        ubw.push_back(xmax[iw]);
        w0.push_back(xinit[iw]);
      }
    }



    // Loop over collocation points
    Xk_end[k] = D[0]*Xk[k];

    for (int j = 0; j < d; ++j) {
      // Expression for the state derivative at the collocation point
      MX xp = C[0][j+1] * Xk[k];

      for (int r = 0; r < d; ++r) {
        xp += C[r+1][j+1] * Xkj[k][r];
      }

      // Append collocation equations
      // TODO
      vector<MX> XUprev{Xkj[k][j], Uk[k], F_prev, QK_prev};
      vector<MX> XU{Xkj[k][j], Uk[k]};
      MX fj = f_xdot(XU)[0];
      MX Lj = f_L(XUprev)[0];


      g.push_back(h*fj - xp);
      for (int iw = 0; iw < nx; ++iw) {
        lbg.push_back(0);
        ubg.push_back(0);
      }

      // Add contribution to the end state
      Xk_end[k] += D[j+1]*Xkj[k][j];

      // Add contribution to quadrature function
      Cost += B[j+1]*Lj*h;
    }

    /// alternative way of writing cost
    // apparently the belowed code do things differently than above
    //vector<MX> XUprev1{Xk_end[k], Uk[k], F_prev, QK_prev};
    //MX Lk = f_L(XUprev1)[0];
    //Cost += Lk * h;




    // New NLP variable for state at end of interval
    Xk[k+1] = MX::sym("X_" + str(k+1), nx);
    w.push_back(Xk[k+1]);
    for (int iw = 0; iw < nx; ++iw) {
      lbw.push_back(xmin[iw]);
      ubw.push_back(xmax[iw]);
      w0.push_back(xinit[iw]);
    }

    /// add inequ here for the bound constraints
    g_ineq.push_back(Xk[k+1]);


    // Add equality constraint
    // for continuity between intervals
    g.push_back(Xk_end[k] - Xk[k+1]);
    for (int iw = 0; iw < nx; ++iw) {
      lbg.push_back(0);
      ubg.push_back(0);
    }

    // update the previous u
    vector<MX> u_prev = vertsplit(Uk[k]);
    F_prev  = u_prev[0];
    QK_prev = u_prev[1];




    cout << "Xk = " << Xk[k] << endl;
    cout << "Uk = " << Uk[k] << endl;


  }

  cout << "w size = " << w.size() << endl;
  cout << "w size = " << MX::vertcat(w).size() << endl;
  cout << "lbw size = " << lbw.size() << endl;
  cout << "ubw size = " << ubw.size() << endl;
  cout << "lbg size = " << lbg.size() << endl;
  cout << "ubg size = " << ubg.size() << endl;
  cout << "g size = " << MX::vertcat(g).size() << endl;


  cout << "all constraints = " << g << endl;


  MX variables   = MX::vertcat(w);
  MX constraints = MX::vertcat(g);

  /// add inequalities
  MX inequalities = MX::vertcat(g_ineq);


  cout << "primal variables = " << variables << endl;




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
  std::map<std::string, DM> arg;


  // Solve the NLP
  arg["lbx"] = lbw;
  arg["ubx"] = ubw;
  arg["lbg"] = lbg;
  arg["ubg"] = ubg;
  arg["x0"]  = w0;
  arg["p"]   = p_c[2];
  //arg["p"]   = p0;
  // arg["p"]   = {0, 0};

  /// keep record of timing
  FStats time;
  time.tic();
  auto res = solver(arg);
  time.toc();
  cout << "nlp t_wall time = " << time.t_wall << endl;
  cout << "nlp t_proc time = " << time.t_proc << endl;



  int N_tot = res.at("x").size1();
  auto CA_opt = res.at("x")(Slice(0, N_tot, nu+nx+nx*d));
  DM CB_opt = res.at("x")(Slice(1, N_tot, nu+nx+nx*d));
  DM TR_opt = res.at("x")(Slice(2, N_tot, nu+nx+nx*d));
  DM TK_opt = res.at("x")(Slice(3, N_tot, nu+nx+nx*d));

  DM F_opt  = res.at("x")(Slice(4, N_tot, nu+nx+nx*d));
  DM QK_opt = res.at("x")(Slice(5, N_tot, nu+nx+nx*d));



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




  //vector<DM> input_grad_f{res.at("x"), p1};
  //DM grad_f = DM::vertcat({solver.get_function("nlp_grad_f")(input_grad_f)[0]});
  //vector<DM> input_hess{res.at("x"), p1, res.at("f"), res.at("lam_g")};
  //cout << "ipopt hessian = " << solver.get_function("nlp_hess_l")(input_hess) << endl;



  /// Compute the sign of dg for inequality constraints
  DM dg_p0 = getDg(res, inequalities, Cost, constraints, variables, p, p_c[0], p_c[0]);

  cout << "inequality constraints = " << inequalities << endl;
  cout << "dg_p0 = " << dg_p0 << endl;

  ///****************************************************
  /// Step 2
  /// scenario generation with Schur-complement

  int nr = 1;

  //vector<int> NACindex(int nx, int nu, int nr) {
  vector<int> NAC_Ctrl_index;
  for (int i = 0; i < nr; ++i) {
    NAC_Ctrl_index.push_back(nx*(i+1) + nx*d*i + nu*i );
  };
    //return res;
  //};

  /// NAC_Ctrl_index already considers for all the u's
  cout << "NAC index list = " << NAC_Ctrl_index << endl;


  /// scenario construction
  int n, m;
  n = lbw.size();
  m = lbg.size();
  int length = n + m;

  int ns = 9;
  // TODO
  // generate a function that computes mNAC from nr and np (robust horizon length and number of parameter)
  int mNAC = 8;
  vector<DM> N(ns);

  for (int is = 0; is < ns; ++is) {
    N[is] = DM(length,mNAC*nu);
  }

  for (int i:NAC_Ctrl_index) {
    for (int j = 0; j < nu; ++j) {
      //for (int s = 0; s < ns; ++s) {
        // scenario 1
        // need NACs for NAC constraint index 0 and 1
        // index 0 = NAC constraint 0, index 1 = NAC constraint 1
      N[0](i + j, 0 + j*mNAC) = 1;
      N[0](i + j, 1 + j*mNAC) = 1;
      N[0](i + j, 2 + j*mNAC) = 1;
      N[0](i + j, 3 + j*mNAC) = 1;
      N[0](i + j, 4 + j*mNAC) = 1;
      N[0](i + j, 5 + j*mNAC) = 1;
      N[0](i + j, 6 + j*mNAC) = 1;
      N[0](i + j, 7 + j*mNAC) = 1;



        // scenario 2 only for NAC index 0
        N[1](i + j, 0 + j*mNAC) = -1;
        // scenario 3 only for NAC index 1
        N[2](i + j, 1 + j*mNAC) = -1;

      N[3](i + j, 2 + j*mNAC) = -1;
      N[4](i + j, 3 + j*mNAC) = -1;
      N[5](i + j, 4 + j*mNAC) = -1;
      N[6](i + j, 5 + j*mNAC) = -1;
      N[7](i + j, 6 + j*mNAC) = -1;
      N[8](i + j, 7 + j*mNAC) = -1;




      // }

    }


  }


  cout << "all N blocks = " << N << endl;


  /// construct the multiplier gamma vector
  DM gamma(mNAC*nu,1);

  // fetch the KKT matrix and RHS for each scenario
  vector<vector<DM>> KR(ns);
  vector<DM> K(ns), R(ns);
  for (int is = 0; is < ns; ++is) {
    KR[is] = getKKTaRHS(res, Cost, constraints, variables, p, p_c[0], p_c[is]);
    K[is] = KR[is][0];
    R[is] = KR[is][1];
  }


  cout << "check on K matrix = " << K << endl;
  cout << "check on R matrix = " << R << endl;



  ///start counting time
  FStats NACtime;
  NACtime.tic();


  /*
  /// try assembling the KKT matrix before solve

  Linsol linear_solver = Linsol("linear_solver", "ma27", K[0].sparsity());

  vector<Linsol> linearsolver(ns);
  for (int is = 0; is < ns; ++is) {
    linearsolver[is] = Linsol("linear_solver", "ma27", K[is].sparsity());
    //linearsolver[is].nfact(K[is]);
  }
  */




  // LHS
  vector<DM> KiN(ns);
  DM totLHS(mNAC*nu, mNAC*nu);
  for (int is = 0; is < ns; ++is) {
    //KiN[is] = linearsolver[is].solve(K[is], N[is]);
    KiN[is] = solve(K[is], N[is], "ma27");
    totLHS += mtimes(N[is].T(), KiN[is]);
  }


  //auto linear_solver = Linsol("linear_solver", "ma27", K1.sparsity());
  //linear_solver.sfact(K1);

  // RHS
  vector<DM> Kir(ns);
  DM totRHS(mNAC*nu, 1);
  for (int is = 0; is < ns; ++is) {
    // Kir[is] = linearsolver[is].solve(K[is], R[is]);
    Kir[is] = solve(K[is], R[is], "ma27");
    totRHS += mtimes(N[is].T(), Kir[is]);
  }

  gamma = solve(totLHS, -totRHS, "ma27");

  cout << "gamma = " << gamma << endl;

  vector<DM> schurR(ns);
  vector<DM> ds(ns), s_c(ns), s_pert;
  for (int is = 0; is < ns; ++is) {
    schurR[is] = R[is] + mtimes(N[is], gamma);
    // ds[is] = linearsolver[is].solve(K[is], -schurR[is]);
    ds[is] = solve(K[is], -schurR[is], "ma27");
  }


  /// end time
  NACtime.toc();
  cout << "total t_wall time = " << NACtime.t_wall << endl;
  cout << "total t_proc time = " << NACtime.t_proc << endl;


  DM s  = DM::vertcat({res.at("x"), res.at("lam_g")});
  cout << "nominal scenario optimal solution s = " << s << endl;

  for (int is = 0; is < ns; ++is) {
    s_c[is] = s + ds[is];
    s_pert.push_back(s_c[is]);
  }


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


  /* step 2 outputs
   * ds for each scenario
   * use them in step 3 NLP problem
   */

  /// step 2.5
  /// have the index list for worse case scenarios
  vector<int> worstcase{7};


  /// Step 3
  /// Calculate the approximate multistage solutions based on sensitivity

  MX Cost_sens = 0;  // cost function for step 3


  vector<vector<MX>> Xkj_sens(horN);
  vector<MX> Uk_sens(horN);


  for (int is = 0; is < ns; ++is) {
    F_prev = Finit;
    QK_prev = QKinit;

    /// if non-critical scenario
    /// then use the following method and based on nominal scenario

    if (is != worstcase[0]) {

      for (int k = 0; k < horN; ++k) {
        Uk_sens[k] =
        Uk[k] + ds[is](Slice(nx * (k + 1) + nu * k + nx * d * k, nx * (k + 1) + nu * (k + 1) + nx * d * k, 1));

        for (int j = 0; j < d; ++j) {

          // We need to update Xkj_sens and Uk_sens for each scenario

          Xkj_sens[k].push_back(Xkj[k][j] +
                                ds[is](Slice(nx * (k + 1) + nu * (k + 1) + nx * (j + d * k),
                                             nx * (k + 1) + nu * (k + 1) + nx * (j + 1 + d * k), 1)));



          // Append collocation equations
          vector<MX> XUprev_sens{Xkj_sens[k][j], Uk_sens[k], F_prev, QK_prev};

          MX Lj_sens = f_L(XUprev_sens)[0];



          // Add contribution to quadrature function
          Cost_sens += B[j + 1] * Lj_sens * h;


        } // collocation

        // update the previous u
        vector<MX> u_prev = vertsplit(Uk_sens[k]);
        F_prev = u_prev[0];
        QK_prev = u_prev[1];

      } // horizon

    }

    /// if critical-scenario, need to add separate variables for cost, and additional NAC constraints

    else {



    }





  } // scenario



  cout << "Uk_sens = " << Uk_sens << endl;
  cout << "cost_sens = " << Cost_sens << endl;


  /// re-solve
  nlp = {
  {"x", variables},
  {"p", p},
  {"f", Cost_sens},
  {"g", constraints}};


  // need to update solver
  solver = nlpsol("solver", "ipopt", nlp, opts);


  FStats newtime;
  newtime.tic();
  res = solver(arg);
  newtime.toc();
  cout << "nlp t_wall time = " << newtime.t_wall << endl;
  cout << "nlp t_proc time = " << newtime.t_proc << endl;



  int newN_tot = res.at("x").size1();
  auto newCA_opt = res.at("x")(Slice(0, newN_tot, nu+nx+nx*d));
  DM newCB_opt = res.at("x")(Slice(1, newN_tot, nu+nx+nx*d));
  DM newTR_opt = res.at("x")(Slice(2, newN_tot, nu+nx+nx*d));
  DM newTK_opt = res.at("x")(Slice(3, newN_tot, nu+nx+nx*d));

  DM newF_opt  = res.at("x")(Slice(4, newN_tot, nu+nx+nx*d));
  DM newQK_opt = res.at("x")(Slice(5, newN_tot, nu+nx+nx*d));



  // Print the solution
  cout << "-----" << endl;
  cout << " Step 3 re-solve NLP results" << endl;
  cout << "Optimal solution for p = " << arg.at("p") << ":" << endl;
  cout << setw(30) << "Objective: "   << res.at("f") << endl;

  cout << setw(30) << "Primal solution (CA): " << newCA_opt << endl;
  cout << setw(30) << "Primal solution (CB): " << newCB_opt << endl;
  cout << setw(30) << "Primal solution (TR): " << newTR_opt << endl;
  cout << setw(30) << "Primal solution (TK): " << newTK_opt << endl;
  cout << setw(30) << "Primal solution (F):  " << newF_opt  << endl;
  cout << setw(30) << "Primal solution (QK): " << newQK_opt << endl;





  return 0;

}

