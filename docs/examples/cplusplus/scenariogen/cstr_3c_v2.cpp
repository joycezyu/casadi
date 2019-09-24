//
// Created by Zhou Yu on 08/06/19.
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
  int horN = 5; // number of control intervals

  double h = T/horN;   // step size
  // Number of scenarios
  int ns = 1;

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


  MX p_CAinit = MX::sym("p_CAinit");
  MX p_CBinit = MX::sym("p_CBinit");
  MX p_TRinit = MX::sym("p_TRinit");
  MX p_TKinit = MX::sym("p_TKinit");
  MX p_xinit  = MX::vertcat({p_CAinit, p_CBinit, p_TRinit, p_TKinit});



  int nx = x.size1();
  int nu = u.size1();
  //int np = p.size1();

  // Declare model parameters (fixed) and fixed bounds value
  double CAinit0  = 0.8;
  double CBinit0  = 0.5;
  double TRinit0  = 134.14;
  double TKinit0  = 134.0;
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




  vector<double> xinit0{CAinit0, CBinit0, TRinit0, TKinit0};
  vector<double> xinit1{0.8, 0.5, 140, 134.0};
  vector<double> uinit{Finit, QKinit};
  vector<double> xmin{CAmin, CBmin, TRmin, TKmin};
  vector<double> xmax{CAmax, CBmax, TRmax, TKmax};
  vector<double> umin{Fmin,  QKmin};
  vector<double> umax{Fmax,  QKmax};



  // set up the params associated with each scenario
  vector<MX> CAins{CAin_nom, CAin_lo,  CAin_up};
  vector<MX> EA3Rs{EA3R_nom, EA3R_nom, EA3R_nom};
  vector<MX> param(ns);
  for (int is = 0; is < ns; ++is) {
    param[is] = MX::vertcat({CAins[is], EA3Rs[is]});
  }




  double absT = 273.15;
  MX k1 = k01 * exp( -EA1R / (TR + absT) );
  MX k2 = k02 * exp( -EA2R / (TR + absT) );

  //vector<MX> k3(ns);
  //for (int is = 0; is < ns; ++is) {
  //  k3[is] = k03 * exp( -EA3R[is] / (TR + absT) );
  //}
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
  //MX F_prev  = MX::sym("F_prev");
  //MX QK_prev = MX::sym("QK_prev");
  MX u_prev = MX::sym("u_prev", nu);



  // Objective
  MX L = (CB - CBref) * (CB - CBref) + r1 *(F-u_prev(0))*(F-u_prev(0)) + r2 *(QK-u_prev(1))*(QK-u_prev(1));
  // MX L = (CB - CBref) * (CB - CBref) + r1*F*F + r2*QK*QK;

  // Continuous time dynamics
  //Function dynamics("f", {x, u, u_prev}, {xdot, L});
  //Function dynamics("f", {x, u, u_prev, p}, {xdot, L});

  Function f_xdot("xdot", {x, u, p}, {xdot});
  Function f_L("L", {x, u, u_prev}, {L});


  // have to do the initialization _after_ constructing Function f
  //u_prev(0) = Finit;
  //u_prev(1) = QKinit;


  // start with an empty NLP
  vector<double> w0, lbw, ubw, lbg, ubg; // w0 is the initial guess
  vector<MX> w, g;
  MX Cost = 0;  // cost function

  // State at collocation points
  vector<MX> Xkj(d);



  /// Preparation for model building
  vector<MX> Xk(ns);
  vector<vector<MX>> Uk(ns);
  vector<MX> Xk_end(ns);
  vector<MX> Uk_prev(ns);

  /// NAC
  /*

  for (int is = 0; is < ns; ++is) {
    Uk[is] = MX::sym("U^" + str(is) + "_0", nu);
    w.push_back(Uk[is]);
  }

  g.push_back(Uk[0] - Uk[1]);
  lbg.push_back(0);
  ubg.push_back(0);
  lbg.push_back(0);
  ubg.push_back(0);

  g.push_back(Uk[0] - Uk[2]);
  lbg.push_back(0);
  ubg.push_back(0);
  lbg.push_back(0);
  ubg.push_back(0);

  */

  // add everything for each scenario
  for (int is = 0; is < ns; ++is) {

    // "lift" initial conditions
    Xk[is] = MX::sym("x0^"+ str(is), nx);
    w.push_back(Xk[is]);
    g.push_back(Xk[is] - p_xinit);
    for (int iw = 0; iw < nx; ++iw) {
      //lbw.push_back(xinit[iw]);
      //ubw.push_back(xinit[iw]);
      lbw.push_back(xmin[iw]);
      ubw.push_back(xmax[iw]);
      lbg.push_back(0);
      ubg.push_back(0);
      w0.push_back(xinit0[iw]);
    }

    Uk_prev[is] = MX::sym("u^" + str(is) + "_prev", nu);
    Uk_prev[is] = MX::vertcat({Finit, QKinit});
    // Uk_prev[is](1) = QKinit;

    cout << "Uk_prev = " << Uk_prev[is] << endl;



    /// Formulate the NLP
    for (int k = 0; k < horN; ++k) {
      // New NLP variable for the control
      Uk[is].push_back(MX::sym("U^" + str(is) + "_" + str(k), nu));
      cout << "U = " << Uk[is].back() << endl;
      w.push_back(Uk[is].back());
      for (int iu = 0; iu < nu; ++iu) {
        lbw.push_back(umin[iu]);
        ubw.push_back(umax[iu]);
        w0.push_back(uinit[iu]);
      }


      // State at collocation points
      // vector<MX> Xkj(d);
      for (int j = 0; j < d; ++j) {
        Xkj[j] = MX::sym("X^" + str(is) + "_" + str(k) + "_" + str(j + 1), nx);
        w.push_back(Xkj[j]);
        for (int iw = 0; iw < nx; ++iw) {
          lbw.push_back(xmin[iw]);
          ubw.push_back(xmax[iw]);
          w0.push_back(xinit0[iw]);
        }
      }


      // Loop over collocation points
      Xk_end[is] = D[0] * Xk[is];

      for (int j = 0; j < d; ++j) {
        // Expression for the state derivative at the collocation point
        MX xp = C[0][j + 1] * Xk[is];

        for (int r = 0; r < d; ++r) {
          xp += C[r + 1][j + 1] * Xkj[r];
        }

        // Append collocation equations
        //vector<MX> XUp{Xkj[j], Uk[is].back(), Uk_prev[is], param[is]};
        //vector<MX> XU{Xkj[j], Uk[is], Uk_prev[is]};
        //vector<MX> XU{Xkj[j], Uk};
        //vector<MX> fL = dynamics(XUp);
        //MX fj = fL[0];
        //MX Lj = fL[1];


        vector<MX> XUprev{Xkj[j], Uk[is].back(), Uk_prev[is]};
        vector<MX> XU{Xkj[j], Uk[is].back(), param[is]};
        MX fj = f_xdot(XU)[0];
        MX Lj = f_L(XUprev)[0];

        cout << "fj = " << fj << endl;
        //cout << "Lj = " << Lj << endl;

        g.push_back(h * fj - xp);
        for (int iw = 0; iw < nx; ++iw) {
          lbg.push_back(0);
          ubg.push_back(0);
        }

        // Add contribution to the end state
        Xk_end[is] += D[j + 1] * Xkj[j];

        // Add contribution to quadrature function
        Cost += B[j + 1] * Lj * h;
      }


      // New NLP variable for state at end of interval
      Xk[is] = MX::sym("X^" + str(is) + "_" + str(k + 1), nx);
      w.push_back(Xk[is]);
      for (int iw = 0; iw < nx; ++iw) {
        lbw.push_back(xmin[iw]);
        ubw.push_back(xmax[iw]);
        w0.push_back(xinit0[iw]);
      }

      // Add equality constraint
      // for continuity between intervals
      g.push_back(Xk_end[is] - Xk[is]);
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
      //vector<MX> u_prev = vertsplit(Uk);
      //F_prev  = u_prev[0];
      //QK_prev = u_prev[1];
      Uk_prev[is] = Uk[is].back();

      cout << "Uk_prev = " << Uk_prev[is] << endl;


    }
    /// NAC
    // Robust horizon = 1
    if (is != 0) {
      g.push_back(Uk[0][0] - Uk[is][0]);

      for (int iu = 0; iu < nu; ++iu) {
        lbg.push_back(0);
        ubg.push_back(0);
      }
    }



  }

  cout << " print out Uk[0]" << Uk[0] << endl;

  //cout << "lbw = " << lbw << endl;
  //cout << "ubw = " << ubw << endl;
  //cout << "lbg = " << lbg << endl;
  //cout << "ubg = " << ubg << endl;


  cout << "w = " << MX::vertcat(w) << endl;
  cout << "g = " << MX::vertcat(g) << endl;
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
  /*
  MXDict nlp = {
  {"x", variables},
  {"p", p},
  {"f", Cost},
  {"g", constraints}};
  */

  MXDict nlp = {
  {"x", variables},
  {"p", p_xinit},
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
  arg["p"]   = xinit0;
  //arg["p"]   = param[0];
  // arg["p"]   = {0, 0};

  /// keep record of timing
  FStats time;
  time.tic();
  auto res = solver(arg);
  time.toc();
  cout << "nlp t_wall time = " << time.t_wall << endl;
  cout << "nlp t_proc time = " << time.t_proc << endl;


  // cout << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1) << "res = " << evalf(res["x"]) << endl;


  /// Print the solution
  cout << "-----" << endl;
  //cout << "Optimal solution for p = " << arg.at("p") << ":" << endl;
  cout << setw(30) << "Objective: "   << res.at("f") << endl;

  int N_tot = res.at("x").size1();
  int N_per_s = N_tot / ns;
  vector<DM> CA_opt(ns), CB_opt(ns), TR_opt(ns), TK_opt(ns), F_opt(ns), QK_opt(ns);

  for (int is = 0; is < ns; ++is) {
    CA_opt[is] = res.at("x")(Slice(    N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    CB_opt[is] = res.at("x")(Slice(1 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    TR_opt[is] = res.at("x")(Slice(2 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    TK_opt[is] = res.at("x")(Slice(3 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    F_opt[is]  = res.at("x")(Slice(4 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    QK_opt[is] = res.at("x")(Slice(5 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));


    cout << setw(30) << " For scenario s = " << is << endl;
    cout << setw(30) << "CA: " << CA_opt[is] << endl;
    cout << setw(30) << "CB: " << CB_opt[is] << endl;
    cout << setw(30) << "TR: " << TR_opt[is] << endl;
    cout << setw(30) << "TK: " << TK_opt[is] << endl;
    cout << setw(30) << "F:  " << F_opt[is]  << endl;
    cout << setw(30) << "QK: " << QK_opt[is] << endl;


  }


  /*

  ///****************************************************
  /// Sensitivity calculation


  int ng = MX::vertcat(g).size1();   // ng = number of constraints g
  int nw = MX::vertcat(w).size1();  // nw = number of variables x

  DM ds = NLPsensitivity_p_factor(res, Cost, constraints, variables, p_xinit, xinit0, xinit1);
  DM s  = DM::vertcat({res.at("x"), res.at("lam_g"), res.at("lam_x")});
  DM s1 = s + ds;
  // int s_tot = s1.size1();
  // cout << "s vector dimension = " << s_tot << endl;
  // cout << "x vector dimension = " << N_tot << endl;

  // cout << "ds = " << ds(Slice(0, N_tot)) << endl;

  vector<DM> CA_pert(ns), CB_pert(ns), TR_pert(ns), TK_pert(ns), F_pert(ns), QK_pert(ns);
  vector<DM> CA_ds(ns), CB_ds(ns), TR_ds(ns), TK_ds(ns), F_ds(ns), QK_ds(ns);

  for (int is = 0; is < ns; ++is) {
    CA_pert[is] = s1(Slice(    N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    CB_pert[is] = s1(Slice(1 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    TR_pert[is] = s1(Slice(2 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    TK_pert[is] = s1(Slice(3 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    F_pert[is]  = s1(Slice(4 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    QK_pert[is] = s1(Slice(5 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));


    CA_ds[is] = ds(Slice(    N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    CB_ds[is] = ds(Slice(1 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    TR_ds[is] = ds(Slice(2 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    TK_ds[is] = ds(Slice(3 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    F_ds[is]  = ds(Slice(4 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));
    QK_ds[is] = ds(Slice(5 + N_per_s * is, N_per_s * (is+1), nu+nx+nx*d));




    cout << setw(30) << " For scenario s = " << is << endl;

    cout << setw(30) << "ds(CA): " << CA_ds[is] << endl;
    cout << setw(30) << "ds(CB): " << CB_ds[is] << endl;
    cout << setw(30) << "ds(TR): " << TR_ds[is] << endl;
    cout << setw(30) << "ds(TK): " << TK_ds[is] << endl;
    cout << setw(30) << "ds(F):  " << F_ds[is]  << endl;
    cout << setw(30) << "ds(QK): " << QK_ds[is] << endl;


    cout << setw(30) << "Perturbed solution CA: " << CA_pert[is] << endl;
    cout << setw(30) << "Perturbed solution CB: " << CB_pert[is] << endl;
    cout << setw(30) << "Perturbed solution TR: " << TR_pert[is] << endl;
    cout << setw(30) << "Perturbed solution TK: " << TK_pert[is] << endl;
    cout << setw(30) << "Perturbed solution F:  " << F_pert[is]  << endl;
    cout << setw(30) << "Perturbed solution QK: " << QK_pert[is] << endl;


  }

  */



  return 0;

}

