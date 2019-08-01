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
DM NLPsensitivity(const std::map<std::string, DM>& res,
                  const MX& objective, const MX& constraints, const MX& variables, const MX& parameters,
                  const std::vector<double>& p0, const std::vector<double>& p1,
                  const std::string& lsolver="ma27");


  CASADI_EXPORT
DM NLPsensitivity_p(const std::map<std::string, DM>& res,
                    const MX& objective, const MX& constraints, const MX& variables, const MX& parameters,
                    const std::vector<double>& p0, const std::vector<double>& p1,
                    const std::string& lsolver="ma27");

  CASADI_EXPORT
  DM NLPsensitivity_p_factor(const std::map<std::string, DM>& res,
                      const MX& objective, const MX& constraints, const MX& variables, const MX& parameters,
                      const std::vector<double>& p0, const std::vector<double>& p1,
                      const std::string& lsolver="ma27", bool prefactor = true);


}  // namespace casadi







#endif //CASADI_SENSITIVITY_HPP
