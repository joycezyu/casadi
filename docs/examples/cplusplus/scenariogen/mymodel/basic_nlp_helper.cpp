//
// Created by Zhou Yu on 9/27/19.
//

#ifndef CASADI_BASIC_NLP_HELP_CPP
#define CASADI_BASIC_NLP_HELP_CPP

#include <casadi/casadi.hpp>
#include "cstr_model.hpp"
#include "scenario_gen_helper.cpp"

namespace casadi {

  struct nlp_setup {
    MXDict nlp;
    Function solver;
    std::map<std::string, DM> arg;
  };



  nlp_setup plant_simulate(double step_length, MX p_xinit, vector<double> xup_init,
                                int nx, int nu, int np ) {
    nlp_setup simulate;

    MX u0 = MX::sym("u0", nu);

    vector<MX> X;
    vector<MX> U;

    vector<double> w0_plt, lbw_plt, ubw_plt, lbg_plt, ubg_plt; // w0 is the initial guess
    vector<MX> w_plt, g_plt;
    MX Cost_plt = 0;  // cost function

    //srand(10);
    //int rd_index = rand() % ns;
    // rand_seed is taken care of in the outer loop


    MX param_plt = MX::sym("param_plt", np);
    //model_setup plant = cstr_model(step_length, 1, p_xinit, X, U, param[rd_index], rd_index);
    model_setup plant = cstr_model(step_length, 1, p_xinit, X, U, param_plt, 0);

    w_plt.insert(w_plt.end(), plant.w.begin(), plant.w.end());
    g_plt.insert(g_plt.end(), plant.g.begin(), plant.g.end());
    w0_plt.insert(w0_plt.end(), plant.w0.begin(), plant.w0.end());
    lbw_plt.insert(lbw_plt.end(), plant.lbw.begin(), plant.lbw.end());
    ubw_plt.insert(ubw_plt.end(), plant.ubw.begin(), plant.ubw.end());
    lbg_plt.insert(lbg_plt.end(), plant.lbg.begin(), plant.lbg.end());
    ubg_plt.insert(ubg_plt.end(), plant.ubg.begin(), plant.ubg.end());


    g_plt.push_back(U[0] - u0);
    for (int iu = 0; iu < nu; ++iu) {
      lbg_plt.push_back(0);
      ubg_plt.push_back(0);
    }


    MX variables_plt = MX::vertcat(w_plt);
    MX constraints_plt = MX::vertcat(g_plt);

    MX p_plt = MX::vertcat({p_xinit, u0, param_plt});

    MXDict nlp_plt = {
    {"x", variables_plt},
    {"p", p_plt},
    {"f", Cost_plt},
    {"g", constraints_plt}};

    Dict opts;
    opts["ipopt.linear_solver"] = "ma27";
    opts["ipopt.print_info_string"] = "yes";
    opts["ipopt.linear_system_scaling"] = "none";

    Function solver_plt = nlpsol("solver", "ipopt", nlp_plt, opts);
    std::map<std::string, DM> arg_plt;
    arg_plt["lbx"] = lbw_plt;
    arg_plt["ubx"] = ubw_plt;
    arg_plt["lbg"] = lbg_plt;
    arg_plt["ubg"] = ubg_plt;
    arg_plt["x0"] = w0_plt;
    arg_plt["p"] = xup_init;

    simulate.nlp = nlp_plt;
    simulate.arg = arg_plt;
    simulate.solver = solver_plt;

    return simulate;
  }




  nlp_setup nmpc_nominal(double time_horizon, int horizon_length, MX p, MX param, vector<double> p0) {

    nlp_setup nmpc_nom;

    vector<MX> Xk;
    vector<MX> Uk;

    // start with an empty NLP
    vector<double> w0, lbw, ubw, lbg, ubg; // w0 is the initial guess
    vector<MX> w, g;
    MX Cost = 0;  // cost function


    model_setup controller;

    controller = cstr_model(time_horizon, horizon_length, p, Xk, Uk, param, 0);
    w.insert(w.end(), controller.w.begin(), controller.w.end());
    g.insert(g.end(), controller.g.begin(), controller.g.end());
    w0.insert(w0.end(), controller.w0.begin(), controller.w0.end());
    lbw.insert(lbw.end(), controller.lbw.begin(), controller.lbw.end());
    ubw.insert(ubw.end(), controller.ubw.begin(), controller.ubw.end());
    lbg.insert(lbg.end(), controller.lbg.begin(), controller.lbg.end());
    ubg.insert(ubg.end(), controller.ubg.begin(), controller.ubg.end());
    Cost = controller.Cost;



    MX variables = MX::vertcat(w);
    MX constraints = MX::vertcat(g);

    MXDict nlp = {
    {"x", variables},
    {"p", p},
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
    arg["p"] = p0;

    nmpc_nom.nlp = nlp;
    nmpc_nom.arg = arg;
    nmpc_nom.solver = solver;

    return nmpc_nom;


  }



  nlp_setup nmpc_nominal_p(double time_horizon, int horizon_length, const vector<double>& xinit, MX p,
                           const vector<double>& p0) {

    nlp_setup nmpc_nom;

    vector<MX> Xk;
    vector<MX> Uk;

    // start with an empty NLP
    vector<double> w0, lbw, ubw, lbg, ubg; // w0 is the initial guess
    vector<MX> w, g;
    MX Cost = 0;  // cost function


    model_setup controller;

    controller = cstr_model_p(time_horizon, horizon_length, xinit, Xk, Uk, p, 0);
    w.insert(w.end(), controller.w.begin(), controller.w.end());
    g.insert(g.end(), controller.g.begin(), controller.g.end());
    w0.insert(w0.end(), controller.w0.begin(), controller.w0.end());
    lbw.insert(lbw.end(), controller.lbw.begin(), controller.lbw.end());
    ubw.insert(ubw.end(), controller.ubw.begin(), controller.ubw.end());
    lbg.insert(lbg.end(), controller.lbg.begin(), controller.lbg.end());
    ubg.insert(ubg.end(), controller.ubg.begin(), controller.ubg.end());
    Cost = controller.Cost;



    MX variables = MX::vertcat(w);
    MX constraints = MX::vertcat(g);

    MXDict nlp = {
    {"x", variables},
    {"p", controller.p_uncertain},
    {"f", Cost},
    {"g", constraints}};

    cout << "print out p = " << nlp.at("p") << endl;

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
    arg["p"] = p0;

    cout << "print out lbx = " << arg.at("lbx") << endl;
    cout << "print out ubx = " << arg.at("ubx") << endl;

    cout << "print out p0 = " << arg.at("p") << endl;


    nmpc_nom.nlp = nlp;
    nmpc_nom.arg = arg;
    nmpc_nom.solver = solver;

    return nmpc_nom;


  }




  nlp_setup gen_step3(double time_horizon, int horizon_length, MX p_xinit, vector<MX> params, vector<double> xinit0,
                       int ns, int worst_case, const vector<DM>& delta_s) {

    nlp_setup step3;

    vector<MX> Xk;
    vector<MX> Uk;

    // start with an empty NLP
    vector<double> w0, lbw, ubw, lbg, ubg; // w0 is the initial guess
    vector<MX> w, g;
    MX Cost = 0;  // cost function

    cout << "checkpoint 5" << endl;

    model_setup controller;

    controller = scenario_gen_helper(time_horizon, horizon_length, p_xinit, params, ns, worst_case, delta_s);
    w.insert(w.end(), controller.w.begin(), controller.w.end());
    g.insert(g.end(), controller.g.begin(), controller.g.end());
    w0.insert(w0.end(), controller.w0.begin(), controller.w0.end());
    lbw.insert(lbw.end(), controller.lbw.begin(), controller.lbw.end());
    ubw.insert(ubw.end(), controller.ubw.begin(), controller.ubw.end());
    lbg.insert(lbg.end(), controller.lbg.begin(), controller.lbg.end());
    ubg.insert(ubg.end(), controller.ubg.begin(), controller.ubg.end());
    Cost = controller.Cost;


    cout << "checkpoint 6" << endl;

    MX variables = MX::vertcat(w);
    MX constraints = MX::vertcat(g);

    MXDict nlp = {
    {"x", variables},
    {"p", controller.p_uncertain},
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

    step3.nlp = nlp;
    step3.arg = arg;
    step3.solver = solver;

    return step3;


  }



  vector<vector<DM>> nlp_res_reader(const std::map<std::string, DM>& result,
                                    int nx, int nu, int d, int num_scenarios = 1) {
      vector<vector<DM>> traj(num_scenarios, vector<DM>(nx+nu));
      int N_tot = result.at("x").size1();
      int N_per_s = N_tot / num_scenarios;

      for (int is = 0; is < num_scenarios; ++is) {
          for (int i = 0; i < nx+nu; ++i) {
              traj[is][i] = result.at("x")(Slice(i + N_per_s * is, N_per_s * (is + 1), nu + nx + nx * d));
          }
      }
      return traj;
  }


}

#endif // CASADI_BASIC_NLP_HELP_CPP