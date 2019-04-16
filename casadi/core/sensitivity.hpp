//
// Created by Zhou Yu on 4/15/19.
//

#ifndef CASADI_SENSITIVITY_HPP
#define CASADI_SENSITIVITY_HPP

#include "casadi/core/function.hpp"



/**
*  Sensitivity calculates MΔs = -N, where
*      [ W    A   -I ]            [∇ₓL]           [Δx]
*  M = [ Aᵀ   0    0 ]   ,  N  =  [ c ]   ,  Δs = [Δλ]
*      [ V    0    X ]            [ 0 ]           [Δν]
*
*  Instead of solving the nonsymmetric linear system, equivalently we solve
*
*  M = [ W + X⁻¹V  A ]  ,   N =  [∇ₓL] ,  Δs = [Δx]
*      [  Aᵀ       0 ]           [ c ]         [Δλ]
*
*  and Δν = -X⁻¹V Δx
*
*
*/




namespace casadi {

  CASADI_EXPORT
DM NLPsensitivity(std::map<std::string, DM>& res,
                  const MX& objective, const MX& constraints, const MX& variables, const MX& parameters,
                  std::vector<double>& p0, std::vector<double>& p1);



}  // namespace casadi







#endif //CASADI_SENSITIVITY_HPP
