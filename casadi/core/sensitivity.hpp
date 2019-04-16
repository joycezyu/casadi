//
// Created by Zhou Yu on 4/15/19.
//

#ifndef CASADI_SENSITIVITY_HPP
#define CASADI_SENSITIVITY_HPP

#include "casadi/core/function.hpp"

namespace casadi {

  CASADI_EXPORT
DM NLPsensitivity(std::map<std::string, DM>& res, int nw, int ng,
                    std::vector<double>& p0, std::vector<double>& p1);



}  // namespace casadi







#endif //CASADI_SENSITIVITY_HPP
