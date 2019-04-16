//
// Created by Zhou Yu on 4/15/19.
//

#include "sensitivity.hpp"

using namespace std;
namespace casadi {

DM NLPsensitivity(std::map<std::string, DM>& res, int nw, int ng,
                  std::vector<double>& p0, std::vector<double>& p1) {
  DM ds = {1};
  return ds;
};



}  // namespace casadi




