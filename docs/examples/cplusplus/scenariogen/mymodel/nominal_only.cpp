//
// Created by Zhou Yu on 09/25/19.
//


#include <iostream>
#include <fstream>
#include <casadi/casadi.hpp>
#include <casadi/core/sensitivity.hpp>
#include <casadi/core/timing.hpp>
#include "cstr_model.hpp"
#include "basic_nlp_helper.cpp"



using namespace casadi;

  int main() {

    int nx = 4;
    int nu = 2;
    int np = 2;
    int d = 3;

    // initial condition for the model
    double CAinit0 = 0.8;
    double CBinit0 = 0.5;
    double TRinit0 = 134.14;
    double TKinit0 = 134.0;
    vector<double> xinit0{CAinit0, CBinit0, TRinit0, TKinit0};

    double Finit    = 18.83;
    double QKinit   = -4495.7;
    vector<double> uinit0{Finit, QKinit};


    MX p_CAinit = MX::sym("p_CAinit");
    MX p_CBinit = MX::sym("p_CBinit");
    MX p_TRinit = MX::sym("p_TRinit");
    MX p_TKinit = MX::sym("p_TKinit");
    MX p_xinit = MX::vertcat({p_CAinit, p_CBinit, p_TRinit, p_TKinit});


    double EA3R_nom = 8560;
    double EA3R_lo = EA3R_nom * (1 - 0.01);
    double EA3R_up = EA3R_nom * (1 + 0.01);


    double CAin_nom = 5.1;
    double CAin_lo = CAin_nom * (1 - 0.3);
    double CAin_up = CAin_nom * (1 + 0.3);

    /// number of scenarios
    int ns = 3;

    double T = 0.2;
    /// horizon length
    int horN = 40;

    // set up the params associated with each scenario
    vector<MX> CAins{CAin_nom, CAin_lo, CAin_up};
    vector<MX> EA3Rs{EA3R_nom, EA3R_nom, EA3R_nom};
    vector<MX> param(ns);
    for (int is = 0; is < ns; ++is) {
      param[is] = MX::vertcat({CAins[is], EA3Rs[is]});
    }


    /// Preparation for model building
    vector<MX> Xk;
    vector<MX> Uk;

    // start with an empty NLP
    vector<double> w0, lbw, ubw, lbg, ubg; // w0 is the initial guess
    vector<MX> w, g;
    MX Cost = 0;  // cost function




    model_setup controller;
    //MX theta = MX::sym("theta");



    controller = cstr_model(T, horN, p_xinit, Xk, Uk, param[0], 0);
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
    {"p", p_xinit},
    {"f", Cost},
    {"g", constraints}};

    cout << "print out nlp variables" << nlp.at("x") << endl;




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

    vector<vector<DM>> mpc_traj = nlp_res_reader(res, nx, nu, d, ns);

    for (int is = 0; is < ns; ++is) {

      cout << setw(30) << " For scenario s = " << is << endl;
      cout << setw(30) << "CA: " << mpc_traj[is][0] << endl;
      cout << setw(30) << "CB: " << mpc_traj[is][1] << endl;
      cout << setw(30) << "TR: " << mpc_traj[is][2] << endl;
      cout << setw(30) << "TK: " << mpc_traj[is][3] << endl;
      cout << setw(30) << "F:  " << mpc_traj[is][4] << endl;
      cout << setw(30) << "QK: " << mpc_traj[is][5] << endl;


    }


    cout << "F0_0 = " << mpc_traj[0][4](0) << endl;

    /// controls that should be implemented in the plant
    MX u0 = MX::sym("u0", nu);
    //uinit0 = {double(mpc_traj[0][4](0)), double(mpc_traj[0][5](0))};
    //cout << "u0_0 = " << uinit0 << endl;


    /// plant simulation
    srand(1);
    int rd_index = rand() % ns;
    cout << "random number = " << rd_index << endl;

    double step_length = T / horN;
    cout << "each step length:" << step_length << endl;

    // create x_u_init = xinit0 + uinit0 + param_realized
    vector<double> x_u_init = xinit0;
    x_u_init.insert(x_u_init.end(), uinit0.begin(), uinit0.end() );


    vector<double> param_realized;
    param_realized = {double(param[rd_index](0)), double(param[rd_index](1))} ;
    cout << "param_realized = " << param_realized << endl;
    x_u_init.insert(x_u_init.end(), param_realized.begin(), param_realized.end() );


    nlp_setup plant = plant_simulate(step_length, p_xinit, x_u_init, nx, nu, np);


    auto res_plt = plant.solver(plant.arg);



    /// show the plant simulation result
    vector<DM> plant_traj = nlp_res_reader(res_plt, nx, nu, d)[0];


    // Print the solution
    cout << "-----" << endl;
    cout << "Optimal solution for p = " << plant.arg.at("p") << ":" << endl;
    cout << setw(30) << "Objective: " << res_plt.at("f") << endl;

    cout << setw(30) << "Simulated (CA): " << plant_traj[0] << endl;
    cout << setw(30) << "Simulated (CB): " << plant_traj[1] << endl;
    cout << setw(30) << "Simulated (TR): " << plant_traj[2] << endl;
    cout << setw(30) << "Simulated (TK): " << plant_traj[3] << endl;
    cout << setw(30) << "Simulated (F):  " << plant_traj[4] << endl;
    cout << setw(30) << "Simulated (QK): " << plant_traj[5] << endl;

    cout << "CA[1] = " << double(plant_traj[0](1)) << endl;


    int rolling_horizon = 20;

    vector<vector<double>> states_plant(rolling_horizon+1, vector<double>(nx, 0));
    vector<vector<double>> controls_mpc(rolling_horizon+1, vector<double>(nu, 0));

    states_plant[0] = xinit0;
    controls_mpc[0] = uinit0;
    //states_plant[0] = {double(plant_traj[0](1)), double(plant_traj[1](1)),
     //                  double(plant_traj[2](1)), double(plant_traj[3](1))};
    //states_plant[0]= {nextCA, nextCB, nextTR, nextTK};
    cout << states_plant << endl;







    /*
     * controller model: nlp and arg
     * plant model: nlp_plt and arg_plt
     * update controller initial states:  arg["p"]   = xinit0;
     * update plant initial states and also control:  arg_plt["p"] = xinit0;
     */


    double setpoint_error = 0;
    vector<int> rand_seed(rolling_horizon);

    for (int i = 0; i < rolling_horizon; ++i) {
      /// first solve plant
      x_u_init = states_plant[i];
      x_u_init.insert(x_u_init.end(), controls_mpc[i].begin(), controls_mpc[i].end());

      // add plant param realized
      rd_index = rand() % ns;
      rand_seed[i] = rd_index;
      param_realized = {double(param[rd_index](0)), double(param[rd_index](1))} ;
      cout << "param_realized = " << param_realized << endl;
      x_u_init.insert(x_u_init.end(), param_realized.begin(), param_realized.end() );


      plant.arg["p"] = x_u_init;
      res_plt = plant.solver(plant.arg);
      plant_traj = nlp_res_reader(res_plt, nx, nu, d)[0];

      // then fetch the new states
      states_plant[i+1] = {double(plant_traj[0](1)), double(plant_traj[1](1)),
                           double(plant_traj[2](1)), double(plant_traj[3](1))};
      xinit0 = states_plant[i+1];

      /// then solve the mpc
      arg["p"] = xinit0;
      res = solver(arg);
      mpc_traj = nlp_res_reader(res, nx, nu, d, ns);

      // fetch the controls
      controls_mpc[i+1] = {double(mpc_traj[0][4](0)), double(mpc_traj[0][5](0))};
      uinit0 = controls_mpc[i+1];

      setpoint_error += pow((xinit0[1] - 0.5), 2);

    }

    cout << "setpoint error = "  << setpoint_error << endl;
    cout << "states profile = "  << states_plant << endl;
    cout << "control profile = " << controls_mpc << endl;
    cout << "random seed = "     << rand_seed    << endl;







    return 0;

  } // main


