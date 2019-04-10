//
// Created by Zhou Yu on 4/5/19.
//



#include <casadi/casadi.hpp>



using namespace casadi;
using namespace std;

int main() {

  // Degree of interpolating polynomial
  int d = 3;

  // Choose collocation points
  vector<double> tau_root = collocation_points(d, "radau");
  //vector<double> tau_root = collocation_points(d, "legendre");
  tau_root.insert(tau_root.begin(), 0);


  // Coefficients of the collocation equation
  vector<vector<double> > C(d+1,vector<double>(d+1,0));

  // Coefficients of the continuity equation
  vector<double> D(d+1,0);

  // Coefficients of the quadrature function
  vector<double> B(d+1,0);

  // For all collocation points
  for(int j=0; j<d+1; ++j) {

    // Construct Lagrange polynomials to get the polynomial basis at the collocation point
    Polynomial p = 1;
    for (int r = 0; r < d + 1; ++r) {
      if (r != j) {
        p *= Polynomial(-tau_root[r], 1) / (tau_root[j] - tau_root[r]);
      }
    }

    // Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    D[j] = p(1.0);

    // Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    Polynomial dp = p.derivative();
    for (int r = 0; r < d + 1; ++r) {
      C[j][r] = dp(tau_root[r]);
    }

    Polynomial pint = p.anti_derivative();
    B[j] = pint(1.0);

  }

  cout << "C = " << C << endl;
  cout << "D = " << D << endl;
  cout << "B = " << B << endl;



  return 0;

}

