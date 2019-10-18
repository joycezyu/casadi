//
// Created by Zhou Yu on 10/17/19.
//


#include <casadi/casadi.hpp>
#include "cstr_model.hpp"


namespace casadi {

  nlp_setup multistage_nmpc(double time_horizon, int horizon_length, MX p, vector<MX> param, vector<double> p0,
                               int nu, int ns, int index_k) {

    nlp_setup multistage_nmpc;


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
    //MX theta = MX::sym("theta");


    for (int is = 0; is < ns; ++is) {
      controller = cstr_model(time_horizon, horizon_length, p, Xk[is], Uk[is], param[is], is, index_k);
      w.insert(w.end(), controller.w.begin(), controller.w.end());
      g.insert(g.end(), controller.g.begin(), controller.g.end());
      w0.insert(w0.end(), controller.w0.begin(), controller.w0.end());
      lbw.insert(lbw.end(), controller.lbw.begin(), controller.lbw.end());
      ubw.insert(ubw.end(), controller.ubw.begin(), controller.ubw.end());
      lbg.insert(lbg.end(), controller.lbg.begin(), controller.lbg.end());
      ubg.insert(ubg.end(), controller.ubg.begin(), controller.ubg.end());
      Cost += controller.Cost / ns;

      /// the inner max operator
      //g.push_back(controller.Cost - theta);
      //lbg.push_back(-inf);
      //ubg.push_back(0);

      /// NAC
      // Robust horizon = 1
      if (is != 0) {
        g.push_back(Uk[0][0] - Uk[is][0]);

        for (int iu = 0; iu < nu; ++iu) {
          lbg.push_back(0);
          ubg.push_back(0);
        }
      }


    }  // per scenario


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

    multistage_nmpc.nlp = nlp;
    multistage_nmpc.arg = arg;
    multistage_nmpc.solver = solver;

    return multistage_nmpc;


  }







};