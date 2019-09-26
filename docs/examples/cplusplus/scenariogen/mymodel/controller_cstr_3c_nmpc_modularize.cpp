//
// Created by Zhou Yu on 09/25/19.
//


#include <iostream>
#include <fstream>
#include <casadi/casadi.hpp>
#include <casadi/core/sensitivity.hpp>
#include <casadi/core/timing.hpp>
#include "cstr_model.hpp"



using namespace casadi;

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



