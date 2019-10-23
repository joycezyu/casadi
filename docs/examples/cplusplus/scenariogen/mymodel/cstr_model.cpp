//
// Created by Zhou Yu on 9/25/19.
//

#include <iostream>
#include <fstream>
#include <casadi/casadi.hpp>
#include "cstr_model.hpp"




namespace casadi {

  using namespace std;

  model_setup cstr_model(double time_horizon, int horizon_length, const MX& p_xinit,
                         vector<MX>& states, vector<MX>& controls, MX param,
                         int index_scenario, int index_k) {
    model_setup model;



    /// collocation

    // Pre-computed radau collocation matrix
    vector<vector<double>> omega(3);
    omega[0] = { 0.19681547722366,  0.39442431473909, 0.37640306270047};
    omega[1] = {-0.06553542585020,  0.29207341166523, 0.51248582618842};
    omega[2] = { 0.02377097434822, -0.04154875212600, 0.11111111111111};


    // Degree of interpolating polynomial
    int d = 3;

    // Choose collocation points
    vector<double> tau_root = collocation_points(d, "radau");   // "legendre"
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


    /// Model building
    // Time horizon
    //double T = 0.2;

    double h = time_horizon/horizon_length;   // step size



    // Declare model variables

    MX CA = MX::sym("CA");
    MX CB = MX::sym("CB");
    MX TR = MX::sym("TR");
    MX TK = MX::sym("TK");
    MX x  = MX::vertcat({CA, CB, TR, TK});

    MX F  = MX::sym("F");
    MX QK = MX::sym("QK");
    MX u  = MX::vertcat({F, QK});

    MX CAin = MX::sym("CAin");
    MX EA3R = MX::sym("EA3R");
    MX p  = MX::vertcat({CAin, EA3R});

    //p = model.p_uncertain;





    int nx = x.size1();
    int nu = u.size1();
    //int np = p.size1();

    // Declare model parameters (fixed) and fixed bounds value
    //double T0       = 0;

    double Finit    = 18.83;
    double QKinit   = -4495.7;

    double r1       = 1e-7;
    double r2       = 1e-11;

    double CAmin    = 0.1;
    double CAmax    = 1;
    double CBmin    = 0.1;
    double CBmax    = 1;
    double TRmin    = 50;
    double TRmax    = 140;
    double TKmin    = 50;
    double TKmax    = 180;

    double Fmin     = 5;
    double Fmax     = 100;
    double QKmin    = -8500;
    double QKmax    = 0;

    double k01      = 1.287e12;
    double k02      = 1.287e12;
    double k03      = 9.043e9;
    double EA1R     = 9758.3;
    double EA2R     = 9758.3;
    //double EA3R_nom = 8560;
    //double EA3R_lo  = EA3R_nom * (1 - 0.01);
    //double EA3R_up  = EA3R_nom * (1 + 0.01);


    double delHAB   = 4.2;
    double delHBC   = -11;
    double delHAD   = -41.85;
    double rho      = 0.9342;
    double Cp       = 3.01;
    double CpK      = 2.0;
    double Area     = 0.215;
    double VR       = 10.01;
    double mk       = 5.0;
    double Tin      = 130.0;
    double kW       = 4032;

    //double CAin_nom = 5.1;
    //double CAin_lo  = CAin_nom * (1 - 0.1);
    //double CAin_up  = CAin_nom * (1 + 0.1);


    double CBref    = 0.5;
    if (index_k >= 20) {
      CBref = 0.7;
    }


    //vector<double> xinit0{CAinit0, CBinit0, TRinit0, TKinit0};

    //TODO: later update xinit1 with the solution from the last nlp iteration
    vector<double> xinit1{0.8, 0.5, 134.14, 134.0};
    vector<double> uinit{Finit, QKinit};
    vector<double> xmin{CAmin, CBmin, TRmin, TKmin};
    vector<double> xmax{CAmax, CBmax, TRmax, TKmax};
    vector<double> umin{Fmin,  QKmin};
    vector<double> umax{Fmax,  QKmax};



    double absT = 273.15;
    MX k1 = k01 * exp( -EA1R / (TR + absT) );
    MX k2 = k02 * exp( -EA2R / (TR + absT) );
    MX k3 = k03 * exp( -EA3R / (TR + absT) );

    // model equations
    MX xdot = vertcat(
    F * (CAin - CA) - k1 * CA - k3 * CA*CA,
    -F * CB          + k1 * CA - k2 * CB,
    F * (Tin - TR)  + kW*Area/(rho*Cp*VR)*(TK-TR) - (k1*CA*delHAB + k2*CB*delHBC + k3*CA*CA*delHAD)/(rho*Cp),
    1 / (mk*CpK)    * (QK + kW*Area*(TR-TK))
    );

    MX u_prev = MX::sym("u_prev", nu);

    // Objective
    MX L = (CB - CBref) * (CB - CBref) + r1 *(F-u_prev(0))*(F-u_prev(0)) + r2 *(QK-u_prev(1))*(QK-u_prev(1));

    Function f_xdot("xdot", {x, u, p}, {xdot});
    Function f_L("L", {x, u, u_prev}, {L});



    /// "lift" initial conditions
    MX Xk = MX::sym("x0^"+ str(index_scenario), nx);
    states.push_back(Xk);
    model.w.push_back(Xk);
    model.g.push_back(Xk - p_xinit);
    for (int iw = 0; iw < nx; ++iw) {
      //lbw.push_back(xinit[iw]);
      //ubw.push_back(xinit[iw]);
      model.lbw.push_back(xmin[iw]);
      model.ubw.push_back(xmax[iw]);
      model.lbg.push_back(0);
      model.ubg.push_back(0);
      model.w0.push_back(xinit1[iw]);
    }

    MX Uk_prev = MX::sym("u^" + str(index_scenario) + "_prev", nu);
    Uk_prev = MX::vertcat({Finit, QKinit});

    cout << "Uk_prev = " << Uk_prev << endl;



    /// Formulate the NLP
    for (int k = 0; k < horizon_length; ++k) {
      // New NLP variable for the control
      MX Uk = MX::sym("U^" + str(index_scenario) + "_" + str(k), nu);
      controls.push_back(Uk);
      cout << "U = " << Uk << endl;
      model.w.push_back(Uk);
      for (int iu = 0; iu < nu; ++iu) {
        model.lbw.push_back(umin[iu]);
        model.ubw.push_back(umax[iu]);
        model.w0.push_back(uinit[iu]);
      }


      // State at collocation points
      vector<MX> Xkj(d);
      for (int j = 0; j < d; ++j) {
        Xkj[j] = MX::sym("X^" + str(index_scenario) + "_" + str(k) + "_" + str(j + 1), nx);
        model.w.push_back(Xkj[j]);
        for (int iw = 0; iw < nx; ++iw) {
          model.lbw.push_back(xmin[iw]);
          model.ubw.push_back(xmax[iw]);
          //model.lbw.push_back(-inf);
          //model.ubw.push_back( inf);
          model.w0.push_back(xinit1[iw]);
        }
      }


      // Loop over collocation points
      MX Xk_end = D[0] * Xk;

      for (int j = 0; j < d; ++j) {
        // Expression for the state derivative at the collocation point
        MX xp = C[0][j + 1] * Xk;

        for (int r = 0; r < d; ++r) {
          xp += C[r + 1][j + 1] * Xkj[r];
        }

        vector<MX> XUprev{Xkj[j], Uk, Uk_prev};
        vector<MX> XU{Xkj[j], Uk, param};
        MX fj = f_xdot(XU)[0];
        MX Lj = f_L(XUprev)[0];

        cout << "fj = " << fj << endl;
        //cout << "Lj = " << Lj << endl;

        model.g.push_back(h * fj - xp);
        for (int iw = 0; iw < nx; ++iw) {
          model.lbg.push_back(0);
          model.ubg.push_back(0);
        }

        // Add contribution to the end state
        Xk_end += D[j + 1] * Xkj[j];

        // Add contribution to quadrature function
        model.Cost += B[j + 1] * Lj * h;
      }


      // New NLP variable for state at end of interval
      Xk = MX::sym("X^" + str(index_scenario) + "_" + str(k + 1), nx);
      model.w.push_back(Xk);
      for (int iw = 0; iw < nx; ++iw) {
        model.lbw.push_back(xmin[iw]);
        model.ubw.push_back(xmax[iw]);
        //model.lbw.push_back(-inf);
        //model.ubw.push_back( inf);
        model.w0.push_back(xinit1[iw]);
      }

      // Add equality constraint
      // for continuity between intervals
      model.g.push_back(Xk_end - Xk);
      for (int iw = 0; iw < nx; ++iw) {
        model.lbg.push_back(0);
        model.ubg.push_back(0);
      }
      Uk_prev = Uk;

      cout << "Uk_prev = " << Uk_prev << endl;


    }

    return model;
  }


  model_setup cstr_model_p(double time_horizon, int horizon_length, const vector<double>& xinit,
                         vector <MX> &states, vector <MX> &controls, MX param,
                         int index_scenario, int index_k) {
    model_setup model;




    /// collocation

    // Pre-computed radau collocation matrix
    vector<vector<double>> omega(3);
    omega[0] = { 0.19681547722366,  0.39442431473909, 0.37640306270047};
    omega[1] = {-0.06553542585020,  0.29207341166523, 0.51248582618842};
    omega[2] = { 0.02377097434822, -0.04154875212600, 0.11111111111111};


    // Degree of interpolating polynomial
    int d = 3;

    // Choose collocation points
    vector<double> tau_root = collocation_points(d, "radau");   // "legendre"
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


    /// Model building
    // Time horizon
    //double T = 0.2;

    double h = time_horizon/horizon_length;   // step size



    // Declare model variables

    MX CA = MX::sym("CA");
    MX CB = MX::sym("CB");
    MX TR = MX::sym("TR");
    MX TK = MX::sym("TK");
    MX x  = MX::vertcat({CA, CB, TR, TK});

    MX F  = MX::sym("F");
    MX QK = MX::sym("QK");
    MX u  = MX::vertcat({F, QK});

    //MX CAin = MX::sym("CAin");
    //MX EA3R = MX::sym("EA3R");
    //MX p  = MX::vertcat({CAin, EA3R});

    //p = model.p_uncertain;

    MX CAin = param(0);
    MX EA3R = param(1);




    int nx = x.size1();
    int nu = u.size1();
    //int np = p.size1();

    // Declare model parameters (fixed) and fixed bounds value
    //double T0       = 0;

    double Finit    = 18.83;
    double QKinit   = -4495.7;

    double r1       = 1e-7;  // 1e-7
    double r2       = 1e-11;  // 1e-11

    double CAmin    = 0.1;
    double CAmax    = 1;
    double CBmin    = 0.1;
    double CBmax    = 1;
    double TRmin    = 50;
    double TRmax    = 140;
    double TKmin    = 50;
    double TKmax    = 180;

    double Fmin     = 5;
    double Fmax     = 100;
    double QKmin    = -8500;
    double QKmax    = 0;

    double k01      = 1.287e12;
    double k02      = 1.287e12;
    double k03      = 9.043e9;
    double EA1R     = 9758.3;
    double EA2R     = 9758.3;
    //double EA3R_nom = 8560;
    //double EA3R_lo  = EA3R_nom * (1 - 0.01);
    //double EA3R_up  = EA3R_nom * (1 + 0.01);


    double delHAB   = 4.2;
    double delHBC   = -11;
    double delHAD   = -41.85;
    double rho      = 0.9342;
    double Cp       = 3.01;
    double CpK      = 2.0;
    double Area     = 0.215;
    double VR       = 10.01;
    double mk       = 5.0;
    double Tin      = 130.0;
    double kW       = 4032;

    //double CAin_nom = 5.1;
    //double CAin_lo  = CAin_nom * (1 - 0.1);
    //double CAin_up  = CAin_nom * (1 + 0.1);


    double CBref    = 0.5;
    if (index_k >= 20) {
      CBref = 0.7;
    }


    //vector<double> xinit0{CAinit0, CBinit0, TRinit0, TKinit0};

    //TODO: later update xinit1 with the solution from the last nlp iteration
    vector<double> xinit1{0.8, 0.5, 134.14, 134.0};
    vector<double> uinit{Finit, QKinit};
    vector<double> xmin{CAmin, CBmin, TRmin, TKmin};
    vector<double> xmax{CAmax, CBmax, TRmax, TKmax};
    vector<double> umin{Fmin,  QKmin};
    vector<double> umax{Fmax,  QKmax};



    double absT = 273.15;
    MX k1 = k01 * exp( -EA1R / (TR + absT) );
    MX k2 = k02 * exp( -EA2R / (TR + absT) );
    MX k3 = k03 * exp( -EA3R / (TR + absT) );

    // model equations
    MX xdot = vertcat(
    F * (CAin - CA) - k1 * CA - k3 * CA*CA,
    -F * CB          + k1 * CA - k2 * CB,
    F * (Tin - TR)  + kW*Area/(rho*Cp*VR)*(TK-TR) - (k1*CA*delHAB + k2*CB*delHBC + k3*CA*CA*delHAD)/(rho*Cp),
    1 / (mk*CpK)    * (QK + kW*Area*(TR-TK))
    );

    MX u_prev = MX::sym("u_prev", nu);

    // Objective
    MX L = (CB - CBref) * (CB - CBref) + r1 *(F-u_prev(0))*(F-u_prev(0)) + r2 *(QK-u_prev(1))*(QK-u_prev(1));

    Function f_xdot("xdot", {x, u, param}, {xdot});
    Function f_L("L", {x, u, u_prev}, {L});



    /// "lift" initial conditions
    MX Xk = MX::sym("x0^"+ str(index_scenario), nx);
    states.push_back(Xk);
    model.w.push_back(Xk);
    model.g.push_back(Xk - xinit);
    for (int iw = 0; iw < nx; ++iw) {
      //lbw.push_back(xinit[iw]);
      //ubw.push_back(xinit[iw]);
      model.lbw.push_back(xmin[iw]);
      model.ubw.push_back(xmax[iw]);
      model.lbg.push_back(0);
      model.ubg.push_back(0);
      model.w0.push_back(xinit1[iw]);
    }

    MX Uk_prev = MX::sym("u^" + str(index_scenario) + "_prev", nu);
    Uk_prev = MX::vertcat({Finit, QKinit});

    cout << "Uk_prev = " << Uk_prev << endl;



    /// Formulate the NLP
    for (int k = 0; k < horizon_length; ++k) {
      // New NLP variable for the control
      MX Uk = MX::sym("U^" + str(index_scenario) + "_" + str(k), nu);
      controls.push_back(Uk);
      cout << "U = " << Uk << endl;
      model.w.push_back(Uk);
      for (int iu = 0; iu < nu; ++iu) {
        model.lbw.push_back(umin[iu]);
        model.ubw.push_back(umax[iu]);
        //model.lbw.push_back(-inf);
        //model.ubw.push_back( inf);
        model.w0.push_back(uinit[iu]);
      }


      // State at collocation points
      vector<MX> Xkj(d);
      for (int j = 0; j < d; ++j) {
        Xkj[j] = MX::sym("X^" + str(index_scenario) + "_" + str(k) + "_" + str(j + 1), nx);
        model.w.push_back(Xkj[j]);
        for (int iw = 0; iw < nx; ++iw) {
          model.lbw.push_back(xmin[iw]);
          model.ubw.push_back(xmax[iw]);
          //model.lbw.push_back(-inf);
          //model.ubw.push_back( inf);
          model.w0.push_back(xinit1[iw]);
        }
      }


      // Loop over collocation points
      MX Xk_end = D[0] * Xk;

      for (int j = 0; j < d; ++j) {
        // Expression for the state derivative at the collocation point
        MX xp = C[0][j + 1] * Xk;

        for (int r = 0; r < d; ++r) {
          xp += C[r + 1][j + 1] * Xkj[r];
        }

        vector<MX> XUprev{Xkj[j], Uk, Uk_prev};
        vector<MX> XU{Xkj[j], Uk, param};
        MX fj = f_xdot(XU)[0];
        MX Lj = f_L(XUprev)[0];

        cout << "fj = " << fj << endl;
        //cout << "Lj = " << Lj << endl;

        model.g.push_back(h * fj - xp);
        for (int iw = 0; iw < nx; ++iw) {
          model.lbg.push_back(0);
          model.ubg.push_back(0);
        }

        // Add contribution to the end state
        Xk_end += D[j + 1] * Xkj[j];

        // Add contribution to quadrature function
        model.Cost += B[j + 1] * Lj * h;
      }


      // New NLP variable for state at end of interval
      Xk = MX::sym("X^" + str(index_scenario) + "_" + str(k + 1), nx);
      model.w.push_back(Xk);
      for (int iw = 0; iw < nx; ++iw) {
        model.lbw.push_back(xmin[iw]);
        model.ubw.push_back(xmax[iw]);
        //model.lbw.push_back(-inf);
        //model.ubw.push_back( inf);
        model.w0.push_back(xinit1[iw]);
      }

      // Add equality constraint
      // for continuity between intervals
      model.g.push_back(Xk_end - Xk);
      for (int iw = 0; iw < nx; ++iw) {
        model.lbg.push_back(0);
        model.ubg.push_back(0);
      }
      Uk_prev = Uk;

      cout << "Uk_prev = " << Uk_prev << endl;


    }

    model.p_uncertain = param;

    return model;
  }




  model_setup cstr_model_soft(double time_horizon, int horizon_length, const MX& p_xinit,
                         vector<MX>& states, vector<MX>& controls, MX param,
                         int index_scenario, int index_k) {
    model_setup model;



    /// collocation

    // Pre-computed radau collocation matrix
    vector<vector<double>> omega(3);
    omega[0] = { 0.19681547722366,  0.39442431473909, 0.37640306270047};
    omega[1] = {-0.06553542585020,  0.29207341166523, 0.51248582618842};
    omega[2] = { 0.02377097434822, -0.04154875212600, 0.11111111111111};


    // Degree of interpolating polynomial
    int d = 3;

    // Choose collocation points
    vector<double> tau_root = collocation_points(d, "radau");   // "legendre"
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


    /// Model building
    // Time horizon
    //double T = 0.2;

    double h = time_horizon/horizon_length;   // step size



    // Declare model variables

    MX CA = MX::sym("CA");
    MX CB = MX::sym("CB");
    MX TR = MX::sym("TR");
    MX TK = MX::sym("TK");
    MX x  = MX::vertcat({CA, CB, TR, TK});

    MX F  = MX::sym("F");
    MX QK = MX::sym("QK");
    MX u  = MX::vertcat({F, QK});

    MX CAin = MX::sym("CAin");
    MX EA3R = MX::sym("EA3R");
    MX p  = MX::vertcat({CAin, EA3R});

    //p = model.p_uncertain;





    int nx = x.size1();
    int nu = u.size1();
    //int np = p.size1();

    // Declare model parameters (fixed) and fixed bounds value
    //double T0       = 0;

    double Finit    = 18.83;
    double QKinit   = -4495.7;

    double r1       = 1e-7;
    double r2       = 1e-11;

    double CAmin    = 0.1;
    double CAmax    = 1;
    double CBmin    = 0.1;
    double CBmax    = 1;
    double TRmin    = 50;
    double TRmax    = 140;
    double TKmin    = 50;
    double TKmax    = 180;

    double Fmin     = 5;
    double Fmax     = 100;
    double QKmin    = -8500;
    double QKmax    = 0;

    double k01      = 1.287e12;
    double k02      = 1.287e12;
    double k03      = 9.043e9;
    double EA1R     = 9758.3;
    double EA2R     = 9758.3;
    //double EA3R_nom = 8560;
    //double EA3R_lo  = EA3R_nom * (1 - 0.01);
    //double EA3R_up  = EA3R_nom * (1 + 0.01);


    double delHAB   = 4.2;
    double delHBC   = -11;
    double delHAD   = -41.85;
    double rho      = 0.9342;
    double Cp       = 3.01;
    double CpK      = 2.0;
    double Area     = 0.215;
    double VR       = 10.01;
    double mk       = 5.0;
    double Tin      = 130.0;
    double kW       = 4032;

    double M        = 1e3;   // penalty for state violations

    //double CAin_nom = 5.1;
    //double CAin_lo  = CAin_nom * (1 - 0.1);
    //double CAin_up  = CAin_nom * (1 + 0.1);


    double CBref    = 0.5;
    if (index_k >= 20) {
      CBref = 0.7;
    }


    //vector<double> xinit0{CAinit0, CBinit0, TRinit0, TKinit0};

    //TODO: later update xinit1 with the solution from the last nlp iteration
    vector<double> xinit1{0.8, 0.5, 134.14, 134.0};
    vector<double> uinit{Finit, QKinit};
    vector<double> xmin{CAmin, CBmin, TRmin, TKmin};
    vector<double> xmax{CAmax, CBmax, TRmax, TKmax};
    vector<double> umin{Fmin,  QKmin};
    vector<double> umax{Fmax,  QKmax};



    double absT = 273.15;
    MX k1 = k01 * exp( -EA1R / (TR + absT) );
    MX k2 = k02 * exp( -EA2R / (TR + absT) );
    MX k3 = k03 * exp( -EA3R / (TR + absT) );

    // model equations
    MX xdot = vertcat(
    F * (CAin - CA) - k1 * CA - k3 * CA*CA,
    -F * CB          + k1 * CA - k2 * CB,
    F * (Tin - TR)  + kW*Area/(rho*Cp*VR)*(TK-TR) - (k1*CA*delHAB + k2*CB*delHBC + k3*CA*CA*delHAD)/(rho*Cp),
    1 / (mk*CpK)    * (QK + kW*Area*(TR-TK))
    );

    MX u_prev = MX::sym("u_prev", nu);

    // Objective
    MX L = (CB - CBref) * (CB - CBref) + r1 *(F-u_prev(0))*(F-u_prev(0)) + r2 *(QK-u_prev(1))*(QK-u_prev(1));

    Function f_xdot("xdot", {x, u, p}, {xdot});
    Function f_L("L", {x, u, u_prev}, {L});




    /// "lift" initial conditions
    MX Xk = MX::sym("x0^"+ str(index_scenario), nx);
    states.push_back(Xk);
    model.w.push_back(Xk);
    model.g.push_back(Xk - p_xinit);
    for (int iw = 0; iw < nx; ++iw) {
      //lbw.push_back(xinit[iw]);
      //ubw.push_back(xinit[iw]);
      model.lbw.push_back(xmin[iw]);
      model.ubw.push_back(xmax[iw]);
      model.lbg.push_back(0);
      model.ubg.push_back(0);
      model.w0.push_back(xinit1[iw]);
    }

    MX Uk_prev = MX::sym("u^" + str(index_scenario) + "_prev", nu);
    Uk_prev = MX::vertcat({Finit, QKinit});

    cout << "Uk_prev = " << Uk_prev << endl;



    /// Formulate the NLP
    for (int k = 0; k < horizon_length; ++k) {
      // New NLP variable for the control
      MX Uk = MX::sym("U^" + str(index_scenario) + "_" + str(k), nu);
      controls.push_back(Uk);
      cout << "U = " << Uk << endl;
      model.w.push_back(Uk);
      for (int iu = 0; iu < nu; ++iu) {
        model.lbw.push_back(umin[iu]);
        model.ubw.push_back(umax[iu]);
        model.w0.push_back(uinit[iu]);
      }


      // State at collocation points
      vector<MX> Xkj(d);
      // variables for soft constraints
      vector<MX> dx_up(d), dx_lo(d);

      for (int j = 0; j < d; ++j) {
        Xkj[j] = MX::sym("X^" + str(index_scenario) + "_" + str(k) + "_" + str(j + 1), nx);
        model.w.push_back(Xkj[j]);

        dx_up[j] = MX::sym("dx_up^" + str(index_scenario) + "_" + str(k) + "_" + str(j + 1), nx);
        dx_lo[j] = MX::sym("dx_lo^" + str(index_scenario) + "_" + str(k) + "_" + str(j + 1), nx);


        // soft constraints
        model.g.push_back( Xkj[j] - dx_up[j]);
        model.g.push_back(-Xkj[j] - dx_lo[j]);

        for (int iw = 0; iw < nx; ++iw) {
          //model.lbw.push_back(xmin[iw]);
          //model.ubw.push_back(xmax[iw]);
          model.lbw.push_back(-inf);
          model.ubw.push_back( inf);
          model.w0.push_back(xinit1[iw]);

          // upper bound
          model.lbg.push_back(-inf);
          model.ubg.push_back(xmax[iw]);
          model.Cost += M * dx_up[j](iw);

          // lower bound
          model.lbg.push_back(-inf);
          model.ubg.push_back(-xmin[iw]);
          model.Cost += M * dx_lo[j](iw);


        }

        model.w.push_back(dx_up[j]);
        model.w.push_back(dx_lo[j]);
        for (int iw = 0; iw < nx; ++iw) {
          model.lbw.push_back(0);
          model.ubw.push_back(inf);
          model.w0.push_back(0);

          model.lbw.push_back(0);
          model.ubw.push_back(inf);
          model.w0.push_back(0);
        }

      }


      // Loop over collocation points
      MX Xk_end = D[0] * Xk;

      for (int j = 0; j < d; ++j) {
        // Expression for the state derivative at the collocation point
        MX xp = C[0][j + 1] * Xk;

        for (int r = 0; r < d; ++r) {
          xp += C[r + 1][j + 1] * Xkj[r];
        }

        vector<MX> XUprev{Xkj[j], Uk, Uk_prev};
        vector<MX> XU{Xkj[j], Uk, param};
        MX fj = f_xdot(XU)[0];
        MX Lj = f_L(XUprev)[0];

        cout << "fj = " << fj << endl;
        //cout << "Lj = " << Lj << endl;

        model.g.push_back(h * fj - xp);
        for (int iw = 0; iw < nx; ++iw) {
          model.lbg.push_back(0);
          model.ubg.push_back(0);
        }

        // Add contribution to the end state
        Xk_end += D[j + 1] * Xkj[j];

        // Add contribution to quadrature function
        model.Cost += B[j + 1] * Lj * h;
      }


      // New NLP variable for state at end of interval
      Xk = MX::sym("X^" + str(index_scenario) + "_" + str(k + 1), nx);
      model.w.push_back(Xk);
      for (int iw = 0; iw < nx; ++iw) {
        //model.lbw.push_back(xmin[iw]);
        //model.ubw.push_back(xmax[iw]);
        model.lbw.push_back(-inf);
        model.ubw.push_back( inf);
        model.w0.push_back(xinit1[iw]);
      }

      // Add equality constraint
      // for continuity between intervals
      model.g.push_back(Xk_end - Xk);
      for (int iw = 0; iw < nx; ++iw) {
        model.lbg.push_back(0);
        model.ubg.push_back(0);
      }
      Uk_prev = Uk;

      cout << "Uk_prev = " << Uk_prev << endl;


    }

    cout << "Cost function = " << model.Cost << endl;
    cout << "variables = " << model.w << endl;
    cout << "lbw = " << model.lbw << endl;
    cout << "ubw = " << model.ubw << endl;



    cout << "***********************" << endl;
    cout << "END OF CSTR_MODEL_SOFT" << endl;


    return model;
  }



  model_setup cstr_model_plant(double time_horizon, int horizon_length, const MX& p_xinit,
                         vector<MX>& states, vector<MX>& controls, MX param,
                         int index_scenario, int index_k) {
    model_setup model;



    /// collocation

    // Pre-computed radau collocation matrix
    vector<vector<double>> omega(3);
    omega[0] = { 0.19681547722366,  0.39442431473909, 0.37640306270047};
    omega[1] = {-0.06553542585020,  0.29207341166523, 0.51248582618842};
    omega[2] = { 0.02377097434822, -0.04154875212600, 0.11111111111111};


    // Degree of interpolating polynomial
    int d = 3;

    // Choose collocation points
    vector<double> tau_root = collocation_points(d, "radau");   // "legendre"
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


    /// Model building
    // Time horizon
    //double T = 0.2;

    double h = time_horizon/horizon_length;   // step size



    // Declare model variables

    MX CA = MX::sym("CA");
    MX CB = MX::sym("CB");
    MX TR = MX::sym("TR");
    MX TK = MX::sym("TK");
    MX x  = MX::vertcat({CA, CB, TR, TK});

    MX F  = MX::sym("F");
    MX QK = MX::sym("QK");
    MX u  = MX::vertcat({F, QK});

    MX CAin = MX::sym("CAin");
    MX EA3R = MX::sym("EA3R");
    MX p  = MX::vertcat({CAin, EA3R});

    //p = model.p_uncertain;





    int nx = x.size1();
    int nu = u.size1();
    //int np = p.size1();

    // Declare model parameters (fixed) and fixed bounds value
    //double T0       = 0;

    double Finit    = 18.83;
    double QKinit   = -4495.7;

    double r1       = 1e-7;
    double r2       = 1e-11;

    double CAmin    = 0.1;
    double CAmax    = 1;
    double CBmin    = 0.1;
    double CBmax    = 1;
    double TRmin    = 50;
    double TRmax    = 140;
    double TKmin    = 50;
    double TKmax    = 180;

    double Fmin     = 5;
    double Fmax     = 100;
    double QKmin    = -8500;
    double QKmax    = 0;

    double k01      = 1.287e12;
    double k02      = 1.287e12;
    double k03      = 9.043e9;
    double EA1R     = 9758.3;
    double EA2R     = 9758.3;
    //double EA3R_nom = 8560;
    //double EA3R_lo  = EA3R_nom * (1 - 0.01);
    //double EA3R_up  = EA3R_nom * (1 + 0.01);


    double delHAB   = 4.2;
    double delHBC   = -11;
    double delHAD   = -41.85;
    double rho      = 0.9342;
    double Cp       = 3.01;
    double CpK      = 2.0;
    double Area     = 0.215;
    double VR       = 10.01;
    double mk       = 5.0;
    double Tin      = 130.0;
    double kW       = 4032;

    //double CAin_nom = 5.1;
    //double CAin_lo  = CAin_nom * (1 - 0.1);
    //double CAin_up  = CAin_nom * (1 + 0.1);


    double CBref    = 0.5;
    if (index_k >= 20) {
      CBref = 0.7;
    }


    //vector<double> xinit0{CAinit0, CBinit0, TRinit0, TKinit0};

    //TODO: later update xinit1 with the solution from the last nlp iteration
    vector<double> xinit1{0.8, 0.5, 134.14, 134.0};
    vector<double> uinit{Finit, QKinit};
    vector<double> xmin{CAmin, CBmin, TRmin, TKmin};
    vector<double> xmax{CAmax, CBmax, TRmax, TKmax};
    vector<double> umin{Fmin,  QKmin};
    vector<double> umax{Fmax,  QKmax};



    double absT = 273.15;
    MX k1 = k01 * exp( -EA1R / (TR + absT) );
    MX k2 = k02 * exp( -EA2R / (TR + absT) );
    MX k3 = k03 * exp( -EA3R / (TR + absT) );

    // model equations
    MX xdot = vertcat(
    F * (CAin - CA) - k1 * CA - k3 * CA*CA,
    -F * CB          + k1 * CA - k2 * CB,
    F * (Tin - TR)  + kW*Area/(rho*Cp*VR)*(TK-TR) - (k1*CA*delHAB + k2*CB*delHBC + k3*CA*CA*delHAD)/(rho*Cp),
    1 / (mk*CpK)    * (QK + kW*Area*(TR-TK))
    );

    MX u_prev = MX::sym("u_prev", nu);

    // Objective
    MX L = (CB - CBref) * (CB - CBref) + r1 *(F-u_prev(0))*(F-u_prev(0)) + r2 *(QK-u_prev(1))*(QK-u_prev(1));

    Function f_xdot("xdot", {x, u, p}, {xdot});
    Function f_L("L", {x, u, u_prev}, {L});



    /// "lift" initial conditions
    MX Xk = MX::sym("x0^"+ str(index_scenario), nx);
    states.push_back(Xk);
    model.w.push_back(Xk);
    model.g.push_back(Xk - p_xinit);
    for (int iw = 0; iw < nx; ++iw) {
      //lbw.push_back(xinit[iw]);
      //ubw.push_back(xinit[iw]);
      //model.lbw.push_back(xmin[iw]);
      //model.ubw.push_back(xmax[iw]);
      model.lbw.push_back(-inf);
      model.ubw.push_back(inf);
      model.lbg.push_back(0);
      model.ubg.push_back(0);
      model.w0.push_back(xinit1[iw]);
    }

    MX Uk_prev = MX::sym("u^" + str(index_scenario) + "_prev", nu);
    Uk_prev = MX::vertcat({Finit, QKinit});

    cout << "Uk_prev = " << Uk_prev << endl;



    /// Formulate the NLP
    for (int k = 0; k < horizon_length; ++k) {
      // New NLP variable for the control
      MX Uk = MX::sym("U^" + str(index_scenario) + "_" + str(k), nu);
      controls.push_back(Uk);
      cout << "U = " << Uk << endl;
      model.w.push_back(Uk);
      for (int iu = 0; iu < nu; ++iu) {
        model.lbw.push_back(umin[iu]);
        model.ubw.push_back(umax[iu]);
        model.w0.push_back(uinit[iu]);
      }


      // State at collocation points
      vector<MX> Xkj(d);
      for (int j = 0; j < d; ++j) {
        Xkj[j] = MX::sym("X^" + str(index_scenario) + "_" + str(k) + "_" + str(j + 1), nx);
        model.w.push_back(Xkj[j]);
        for (int iw = 0; iw < nx; ++iw) {
          //model.lbw.push_back(xmin[iw]);
          //model.ubw.push_back(xmax[iw]);
          model.lbw.push_back(-inf);
          model.ubw.push_back( inf);
          model.w0.push_back(xinit1[iw]);
        }
      }


      // Loop over collocation points
      MX Xk_end = D[0] * Xk;

      for (int j = 0; j < d; ++j) {
        // Expression for the state derivative at the collocation point
        MX xp = C[0][j + 1] * Xk;

        for (int r = 0; r < d; ++r) {
          xp += C[r + 1][j + 1] * Xkj[r];
        }

        vector<MX> XUprev{Xkj[j], Uk, Uk_prev};
        vector<MX> XU{Xkj[j], Uk, param};
        MX fj = f_xdot(XU)[0];
        MX Lj = f_L(XUprev)[0];

        cout << "fj = " << fj << endl;
        //cout << "Lj = " << Lj << endl;

        model.g.push_back(h * fj - xp);
        for (int iw = 0; iw < nx; ++iw) {
          model.lbg.push_back(0);
          model.ubg.push_back(0);
        }

        // Add contribution to the end state
        Xk_end += D[j + 1] * Xkj[j];

        // Add contribution to quadrature function
        model.Cost += B[j + 1] * Lj * h;
      }


      // New NLP variable for state at end of interval
      Xk = MX::sym("X^" + str(index_scenario) + "_" + str(k + 1), nx);
      model.w.push_back(Xk);
      for (int iw = 0; iw < nx; ++iw) {
        //model.lbw.push_back(xmin[iw]);
        //model.ubw.push_back(xmax[iw]);
        model.lbw.push_back(-inf);
        model.ubw.push_back( inf);
        model.w0.push_back(xinit1[iw]);
      }

      // Add equality constraint
      // for continuity between intervals
      model.g.push_back(Xk_end - Xk);
      for (int iw = 0; iw < nx; ++iw) {
        model.lbg.push_back(0);
        model.ubg.push_back(0);
      }
      Uk_prev = Uk;

      cout << "Uk_prev = " << Uk_prev << endl;


    }

    return model;
  }


}

