//
// Created by Zhou Yu on 9/25/19.
//

#ifndef CASADI_CSTR_MODEL_HPP
#define CASADI_CSTR_MODEL_HPP

#include <iostream>
#include <fstream>
#include <casadi/casadi.hpp>


namespace casadi {
  using namespace std;

  struct model_setup {
    vector<double> w0;
    vector<double> lbw;
    vector<double> ubw;
    vector<double> lbg;
    vector<double> ubg;
    vector <MX> w;
    vector <MX> g;
    MX Cost = 0;
    MX p;
  };

  model_setup cstr_model(double time_horizon, int horizon_length, const MX &p_xinit,
                         vector <MX> &states, vector <MX> &controls, MX param,
                         int index_scenario);
} // casadi



#endif //CASADI_CSTR_MODEL_HPP
