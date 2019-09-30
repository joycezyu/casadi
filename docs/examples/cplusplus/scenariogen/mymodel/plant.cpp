//
// Created by Zhou Yu on 9/27/19.
//


#include <casadi/casadi.hpp>
#include "cstr_model.hpp"

namespace casadi {

  vector<double> plant_simulate(double step_length, MX p_xinit,
                                int nx, int nu, int np ) {
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