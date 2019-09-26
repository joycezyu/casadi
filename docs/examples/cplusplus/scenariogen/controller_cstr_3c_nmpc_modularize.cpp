//
// Created by Zhou Yu on 09/25/19.
//


#include <iostream>
#include <fstream>
#include <casadi/casadi.hpp>
#include <casadi/core/sensitivity.hpp>
#include <casadi/core/timing.hpp>
#include "mymodel/cstr_model.hpp"







using namespace std;
namespace casadi {
/*
  struct model_setup {
    vector<double> w0;
    vector<double> lbw;
    vector<double> ubw;
    vector<double> lbg;
    vector<double> ubg;
    vector<MX> w;
    vector<MX> g;
    MX Cost = 0;
  };

  model_setup cstr_model(double time_horizon, int horizon_length, const MX& p_xinit,
                                    vector<MX>& states, vector<MX>& controls, MX param,
                                    int index_scenario) {
    model_setup model;



    /// collocation

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
    //double T = 0.2;

    double h = time_horizon/horizon_length;   // step size



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
    //int np = p.size1();

    // Declare model parameters (fixed) and fixed bounds value
    //double T0       = 0;

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
    //double EA3R_nom = 8560;
    //double EA3R_lo  = EA3R_nom * (1 - 0.01);
    //double EA3R_up  = EA3R_nom * (1 + 0.01);


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

    //double CAin_nom = 5.1;
    //double CAin_lo  = CAin_nom * (1 - 0.1);
    //double CAin_up  = CAin_nom * (1 + 0.1);


    double CBref    = 0.5;


    //vector<double> xinit0{CAinit0, CBinit0, TRinit0, TKinit0};

    //TODO: later update xinit1 with the solution from the last nlp iteration
    vector<double> xinit1{0.8, 0.5, 134.14, 134.0};
    vector<double> uinit{Finit, QKinit};
    vector<double> xmin{CAmin, CBmin, TRmin, TKmin};
    vector<double> xmax{CAmax, CBmax, TRmax, TKmax};
    vector<double> umin{Fmin,  QKmin};
    vector<double> umax{Fmax,  QKmax};



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

    MX u_prev = MX::sym("u_prev", nu);

    // Objective
    MX L = (CB - CBref) * (CB - CBref) + r1 *(F-u_prev(0))*(F-u_prev(0)) + r2 *(QK-u_prev(1))*(QK-u_prev(1));

    Function f_xdot("xdot", {x, u, p}, {xdot});
    Function f_L("L", {x, u, u_prev}, {L});



    /// "lift" initial conditions
    MX Xk = MX::sym("x0^"+ str(index_scenario), nx);
    states.push_back(Xk);
    model.w.push_back(Xk);
    model.g.push_back(Xk - p_xinit);
    for (int iw = 0; iw < nx; ++iw) {
      //lbw.push_back(xinit[iw]);
      //ubw.push_back(xinit[iw]);
      model.lbw.push_back(xmin[iw]);
      model.ubw.push_back(xmax[iw]);
      model.lbg.push_back(0);
      model.ubg.push_back(0);
      model.w0.push_back(xinit1[iw]);
    }

    MX Uk_prev = MX::sym("u^" + str(index_scenario) + "_prev", nu);
    Uk_prev = MX::vertcat({Finit, QKinit});

    cout << "Uk_prev = " << Uk_prev << endl;



    /// Formulate the NLP
    for (int k = 0; k < horizon_length; ++k) {
      // New NLP variable for the control
      MX Uk = MX::sym("U^" + str(index_scenario) + "_" + str(k), nu);
      controls.push_back(Uk);
      cout << "U = " << Uk << endl;
      model.w.push_back(Uk);
      for (int iu = 0; iu < nu; ++iu) {
        model.lbw.push_back(umin[iu]);
        model.ubw.push_back(umax[iu]);
        model.w0.push_back(uinit[iu]);
      }


      // State at collocation points
      vector<MX> Xkj(d);
      for (int j = 0; j < d; ++j) {
        Xkj[j] = MX::sym("X^" + str(index_scenario) + "_" + str(k) + "_" + str(j + 1), nx);
        model.w.push_back(Xkj[j]);
        for (int iw = 0; iw < nx; ++iw) {
          model.lbw.push_back(xmin[iw]);
          model.ubw.push_back(xmax[iw]);
          model.w0.push_back(xinit1[iw]);
        }
      }


      // Loop over collocation points
      MX Xk_end = D[0] * Xk;

      for (int j = 0; j < d; ++j) {
        // Expression for the state derivative at the collocation point
        MX xp = C[0][j + 1] * Xk;

        for (int r = 0; r < d; ++r) {
          xp += C[r + 1][j + 1] * Xkj[r];
        }

        vector<MX> XUprev{Xkj[j], Uk, Uk_prev};
        vector<MX> XU{Xkj[j], Uk, param};
        MX fj = f_xdot(XU)[0];
        MX Lj = f_L(XUprev)[0];

        cout << "fj = " << fj << endl;
        //cout << "Lj = " << Lj << endl;

        model.g.push_back(h * fj - xp);
        for (int iw = 0; iw < nx; ++iw) {
          model.lbg.push_back(0);
          model.ubg.push_back(0);
        }

        // Add contribution to the end state
        Xk_end += D[j + 1] * Xkj[j];

        // Add contribution to quadrature function
        model.Cost += B[j + 1] * Lj * h;
      }


      // New NLP variable for state at end of interval
      Xk = MX::sym("X^" + str(index_scenario) + "_" + str(k + 1), nx);
      model.w.push_back(Xk);
      for (int iw = 0; iw < nx; ++iw) {
        model.lbw.push_back(xmin[iw]);
        model.ubw.push_back(xmax[iw]);
        model.w0.push_back(xinit1[iw]);
      }

      // Add equality constraint
      // for continuity between intervals
      model.g.push_back(Xk_end - Xk);
      for (int iw = 0; iw < nx; ++iw) {
        model.lbg.push_back(0);
        model.ubg.push_back(0);
      }
      Uk_prev = Uk;

      cout << "Uk_prev = " << Uk_prev << endl;


    }

    return model;
  }

}

using namespace casadi;

*/
  int main() {

    int nx = 4;
    int nu = 2;
    int d = 3;

    // initial condition for the model
    double CAinit0 = 0.8;
    double CBinit0 = 0.5;
    double TRinit0 = 134.14;
    double TKinit0 = 134.0;
    vector<double> xinit0{CAinit0, CBinit0, TRinit0, TKinit0};


    MX p_CAinit = MX::sym("p_CAinit");
    MX p_CBinit = MX::sym("p_CBinit");
    MX p_TRinit = MX::sym("p_TRinit");
    MX p_TKinit = MX::sym("p_TKinit");
    MX p_xinit = MX::vertcat({p_CAinit, p_CBinit, p_TRinit, p_TKinit});


    double EA3R_nom = 8560;
    double EA3R_lo = EA3R_nom * (1 - 0.01);
    double EA3R_up = EA3R_nom * (1 + 0.01);


    double CAin_nom = 5.1;
    double CAin_lo = CAin_nom * (1 - 0.1);
    double CAin_up = CAin_nom * (1 + 0.1);

    /// number of scenarios
    int ns = 3;

    double T = 0.2;
    /// horizon length
    int horN = 5;

    // set up the params associated with each scenario
    vector<MX> CAins{CAin_nom, CAin_lo, CAin_up};
    vector<MX> EA3Rs{EA3R_nom, EA3R_nom, EA3R_nom};
    vector<MX> param(ns);
    for (int is = 0; is < ns; ++is) {
      param[is] = MX::vertcat({CAins[is], EA3Rs[is]});
    }


    /// Preparation for model building
    vector<vector<MX>> Xk(ns);
    vector<vector<MX>> Uk(ns);
    // vector<MX> Xk_end(ns);
    //vector<MX> Uk_prev(ns);

    // start with an empty NLP
    vector<double> w0, lbw, ubw, lbg, ubg; // w0 is the initial guess
    vector<MX> w, g;
    MX Cost = 0;  // cost function




    model_setup controller;


    for (int is = 0; is < ns; ++is) {
      controller = cstr_model(T, horN, p_xinit, Xk[is], Uk[is], param[is], is);
      w.insert(w.end(), controller.w.begin(), controller.w.end());
      g.insert(g.end(), controller.g.begin(), controller.g.end());
      w0.insert(w0.end(), controller.w0.begin(), controller.w0.end());
      lbw.insert(lbw.end(), controller.lbw.begin(), controller.lbw.end());
      ubw.insert(ubw.end(), controller.ubw.begin(), controller.ubw.end());
      lbg.insert(lbg.end(), controller.lbg.begin(), controller.lbg.end());
      ubg.insert(ubg.end(), controller.ubg.begin(), controller.ubg.end());
      Cost += controller.Cost / ns;


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


    MX variables = MX::vertcat(w);
    MX constraints = MX::vertcat(g);

    MXDict nlp = {
    {"x", variables},
    {"p", p_xinit},
    {"f", Cost},
    {"g", constraints}};


    Dict opts;
    opts["ipopt.linear_solver"] = "ma27";
    opts["ipopt.print_info_string"] = "yes";
    opts["ipopt.linear_system_scaling"] = "none";

    Function solver = nlpsol("solver", "ipopt", nlp, opts);
    std::map<std::string, DM> arg;


    /// Solve the NLP
    arg["lbx"] = lbw;
    arg["ubx"] = ubw;
    arg["lbg"] = lbg;
    arg["ubg"] = ubg;
    arg["x0"] = w0;
    arg["p"] = xinit0;

    /// keep record of timing
    FStats time;
    time.tic();
    auto res = solver(arg);
    time.toc();
    cout << "nlp t_wall time = " << time.t_wall << endl;
    cout << "nlp t_proc time = " << time.t_proc << endl;



    /// Print the solution
    cout << "-----" << endl;
    //cout << "Optimal solution for p = " << arg.at("p") << ":" << endl;
    cout << setw(30) << "Objective: " << res.at("f") << endl;

    int N_tot = res.at("x").size1();
    int N_per_s = N_tot / ns;
    vector<DM> CA_opt(ns), CB_opt(ns), TR_opt(ns), TK_opt(ns), F_opt(ns), QK_opt(ns);

    for (int is = 0; is < ns; ++is) {
      CA_opt[is] = res.at("x")(Slice(N_per_s * is, N_per_s * (is + 1), nu + nx + nx * d));
      CB_opt[is] = res.at("x")(Slice(1 + N_per_s * is, N_per_s * (is + 1), nu + nx + nx * d));
      TR_opt[is] = res.at("x")(Slice(2 + N_per_s * is, N_per_s * (is + 1), nu + nx + nx * d));
      TK_opt[is] = res.at("x")(Slice(3 + N_per_s * is, N_per_s * (is + 1), nu + nx + nx * d));
      F_opt[is] = res.at("x")(Slice(4 + N_per_s * is, N_per_s * (is + 1), nu + nx + nx * d));
      QK_opt[is] = res.at("x")(Slice(5 + N_per_s * is, N_per_s * (is + 1), nu + nx + nx * d));


      cout << setw(30) << " For scenario s = " << is << endl;
      cout << setw(30) << "CA: " << CA_opt[is] << endl;
      cout << setw(30) << "CB: " << CB_opt[is] << endl;
      cout << setw(30) << "TR: " << TR_opt[is] << endl;
      cout << setw(30) << "TK: " << TK_opt[is] << endl;
      cout << setw(30) << "F:  " << F_opt[is] << endl;
      cout << setw(30) << "QK: " << QK_opt[is] << endl;


    }


    cout << F_opt[0](0) << endl;

    /// controls that should be implemented in the plant
    DM first_step_control = DM::vertcat({F_opt[0](0), QK_opt[0](0)});
    cout << first_step_control << endl;


    /// plant simulation
    vector<MX> X;
    vector<MX> U;

    vector<double> w0_plt, lbw_plt, ubw_plt, lbg_plt, ubg_plt; // w0 is the initial guess
    vector<MX> w_plt, g_plt;
    MX Cost_plt = 0;  // cost function

    int rd_index = rand() % ns;
    cout << "random number = " << rd_index << endl;

    double step_length = T / horN;
    cout << "each step length:" << step_length << endl;

    model_setup plant = cstr_model(step_length, 1, p_xinit, X, U, param[rd_index], rd_index);

    cout << "each step length:" << step_length << endl;
    w_plt.insert(w_plt.end(), plant.w.begin(), plant.w.end());
    g_plt.insert(g_plt.end(), plant.g.begin(), plant.g.end());
    w0_plt.insert(w0_plt.end(), plant.w0.begin(), plant.w0.end());
    lbw_plt.insert(lbw_plt.end(), plant.lbw.begin(), plant.lbw.end());
    ubw_plt.insert(ubw_plt.end(), plant.ubw.begin(), plant.ubw.end());
    lbg_plt.insert(lbg_plt.end(), plant.lbg.begin(), plant.lbg.end());
    ubg_plt.insert(ubg_plt.end(), plant.ubg.begin(), plant.ubg.end());


    g_plt.push_back(U[0] - first_step_control);
    for (int iu = 0; iu < nu; ++iu) {
      lbg_plt.push_back(0);
      ubg_plt.push_back(0);
    }


    MX variables_plt = MX::vertcat(w_plt);
    MX constraints_plt = MX::vertcat(g_plt);

    MXDict nlp_plt = {
    {"x", variables_plt},
    {"p", p_xinit},
    {"f", Cost_plt},
    {"g", constraints_plt}};


    Function solver_plt = nlpsol("solver", "ipopt", nlp_plt, opts);
    std::map<std::string, DM> arg_plt;


    /// Solve the NLP
    arg_plt["lbx"] = lbw_plt;
    arg_plt["ubx"] = ubw_plt;
    arg_plt["lbg"] = lbg_plt;
    arg_plt["ubg"] = ubg_plt;
    arg_plt["x0"] = w0_plt;
    arg_plt["p"] = xinit0;

    auto res_plt = solver_plt(arg_plt);



    /// show the plant simulation result
    int N_tot_plt = res_plt.at("x").size1();
    auto CA_plt = res_plt.at("x")(Slice(0, N_tot_plt, nu + nx + nx * d));
    DM CB_plt = res_plt.at("x")(Slice(1, N_tot_plt, nu + nx + nx * d));
    DM TR_plt = res_plt.at("x")(Slice(2, N_tot_plt, nu + nx + nx * d));
    DM TK_plt = res_plt.at("x")(Slice(3, N_tot_plt, nu + nx + nx * d));

    DM F_plt = res_plt.at("x")(Slice(4, N_tot_plt, nu + nx + nx * d));
    DM QK_plt = res_plt.at("x")(Slice(5, N_tot_plt, nu + nx + nx * d));


    // Print the solution
    cout << "-----" << endl;
    cout << "Optimal solution for p = " << arg_plt.at("p") << ":" << endl;
    cout << setw(30) << "Objective: " << res_plt.at("f") << endl;

    cout << setw(30) << "Simulated (CA): " << CA_plt << endl;
    cout << setw(30) << "Simulated (CB): " << CB_plt << endl;
    cout << setw(30) << "Simulated (TR): " << TR_plt << endl;
    cout << setw(30) << "Simulated (TK): " << TK_plt << endl;
    cout << setw(30) << "Simulated (F):  " << F_plt << endl;
    cout << setw(30) << "Simulated (QK): " << QK_plt << endl;

    cout << "CA[1] = " << double(CA_plt(1)) << endl;


    int rolling_horizon = 1;

    vector<vector<double>> states_plant(rolling_horizon, vector<double>(nx, 0));

    states_plant[0] = {double(CA_plt(1)), double(CB_plt(1)), double(TR_plt(1)), double(TK_plt(1))};
    //states_plant[0]= {nextCA, nextCB, nextTR, nextTK};
    cout << states_plant << endl;





    /*
     * controller model: nlp and arg
     * plant model: nlp_plt and arg_plt
     * update controller initial states:  arg["p"]   = xinit0;
     * update plant initial states and also control:  arg_plt["p"] = xinit0;
     */









    return 0;

  } // main
} // casadi


