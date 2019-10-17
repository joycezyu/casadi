//
// Created by Zhou Yu on 10/17/19.
//


#include <casadi/casadi.hpp>
#include "cstr_model.hpp"


namespace casadi {

  nlp_setup multistage_3c_nmpc(double time_horizon, int horizon_length, MX p, MX param, vector<double> p0,
                               int index_k) {

    nlp_setup multistage_3c_nmpc;

    vector<MX> Xk;
    vector<MX> Uk;

    // start with an empty NLP
    vector<double> w0, lbw, ubw, lbg, ubg; // w0 is the initial guess
    vector<MX> w, g;
    MX Cost = 0;  // cost function


    model_setup controller;

    controller = cstr_model(time_horizon, horizon_length, p, Xk, Uk, param, 0, index_k);
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







};