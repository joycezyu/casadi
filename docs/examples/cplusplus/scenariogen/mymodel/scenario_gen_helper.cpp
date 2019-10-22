//
// Created by Zhou Yu on 9/30/19.
//


#include <iostream>
#include <fstream>
#include <casadi/casadi.hpp>
#include "cstr_model.hpp"



namespace casadi {

  model_setup scenario_gen_helper(double time_horizon, int horizon_length, const MX& p_xinit,
                         vector<MX> param, int ns,
                         vector<int> worst_case, const vector<DM>& delta_s, int index_k) {
    model_setup model;

    cout << "checkpoint s1" << endl;

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


    cout << "checkpoint s2" << endl;


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
    if (index_k >= 60) {
      CBref = 0.7;
    }

    cout << "checkpoint s3" << endl;

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


    /// Preparation for model building
    //vector<MX> Xk(ns);
    //vector<vector<MX>> Uk(ns);
    //vector<MX> Xk_end(ns);
    //vector<MX> Uk_prev(ns);

    vector<MX> Xk(horizon_length+1);
    vector<MX> Uk0(horizon_length);  // this is for the nominal scenario
    vector<MX> Uk(horizon_length);   // this is for critical scenario
    vector<MX> Xk_end(horizon_length);

    vector<vector<MX>> Xkj(horizon_length, vector<MX>(d));
    MX Cost_sens = 0;

    cout << "checkpoint s4" << endl;

    vector<vector<MX>> Xkj_sens(horizon_length);
    vector<MX> Uk_sens(horizon_length);


    MX Uk_prev = MX::sym("u^0_prev", nu);
    Uk_prev = MX::vertcat({Finit, QKinit});


    /// add everything for each scenario

    // create a dummy u for the NAC
    //MX dummy_u = MX::sym("dummy_u", nu);

    /// first add the nominal case
    for (int is = 0; is < ns; ++is) {
      if (is == 0) {

        cout << "checkpoint nominal case added" << endl;

        /// "lift" initial conditions
        Xk[0] = MX::sym("x0^" + str(is), nx);
        //states.push_back(Xk[is]);
        model.w.push_back(Xk[0]);
        model.g.push_back(Xk[0] - p_xinit);
        for (int iw = 0; iw < nx; ++iw) {
          //lbw.push_back(xinit[iw]);
          //ubw.push_back(xinit[iw]);
          model.lbw.push_back(xmin[iw]);
          model.ubw.push_back(xmax[iw]);
          model.lbg.push_back(0);
          model.ubg.push_back(0);
          model.w0.push_back(xinit1[iw]);
        }

        //Uk_prev[is] = MX::sym("u^" + str(is) + "_prev", nu);
        //Uk_prev[is] = MX::vertcat({Finit, QKinit});

        //cout << "Uk_prev = " << Uk_prev[is] << endl;


        /// Formulate the NLP
        for (int k = 0; k < horizon_length; ++k) {
          // New NLP variable for the control
          Uk0[k] = MX::sym("U^" + str(is) + "_" + str(k), nu);
          //controls.push_back(Uk[is][k]);
          cout << "U = " << Uk0[k] << endl;
          model.w.push_back(Uk0[k]);
          for (int iu = 0; iu < nu; ++iu) {
            model.lbw.push_back(umin[iu]);
            model.ubw.push_back(umax[iu]);
            model.w0.push_back(uinit[iu]);
          }


          // State at collocation points
          for (int j = 0; j < d; ++j) {
            Xkj[k][j] = MX::sym("X^" + str(is) + "_" + str(k) + "_" + str(j + 1), nx);
            model.w.push_back(Xkj[k][j]);
            for (int iw = 0; iw < nx; ++iw) {
              model.lbw.push_back(xmin[iw]);
              model.ubw.push_back(xmax[iw]);
              model.w0.push_back(xinit1[iw]);
            }
          }


          // Loop over collocation points
          Xk_end[k] = D[0] * Xk[k];


          for (int j = 0; j < d; ++j) {
            // Expression for the state derivative at the collocation point
            MX xp = C[0][j + 1] * Xk[k];

            for (int r = 0; r < d; ++r) {
              xp += C[r + 1][j + 1] * Xkj[k][r];
            }

            vector<MX> XUprev{Xkj[k][j], Uk0[k], Uk_prev};
            vector<MX> XU{Xkj[k][j], Uk0[k], param[is]};
            MX fj = f_xdot(XU)[0];
            MX Lj = f_L(XUprev)[0];

            //cout << "fj = " << fj << endl;
            //cout << "Lj = " << Lj << endl;

            model.g.push_back(h * fj - xp);
            for (int iw = 0; iw < nx; ++iw) {
              model.lbg.push_back(0);
              model.ubg.push_back(0);
            }

            // Add contribution to the end state
            Xk_end[k] += D[j + 1] * Xkj[k][j];

            // Add contribution to quadrature function
            model.Cost += B[j + 1] * Lj * h;
          }


          // New NLP variable for state at end of interval
          Xk[k+1] = MX::sym("X^" + str(is) + "_" + str(k + 1), nx);
          model.w.push_back(Xk[k+1]);
          for (int iw = 0; iw < nx; ++iw) {
            model.lbw.push_back(xmin[iw]);
            model.ubw.push_back(xmax[iw]);
            model.w0.push_back(xinit1[iw]);
          }

          // Add equality constraint
          // for continuity between intervals
          model.g.push_back(Xk_end[k] - Xk[k+1]);
          for (int iw = 0; iw < nx; ++iw) {
            model.lbg.push_back(0);
            model.ubg.push_back(0);
          }
          Uk_prev = Uk0[k];

          //cout << "Uk_prev = " << Uk_prev << endl;
        }

        /// NAC
        /*
        // Robust horizon = 1
        model.g.push_back(Uk[0] - dummy_u);
        model.w.push_back(dummy_u);

        for (int iu = 0; iu < nu; ++iu) {
          model.lbg.push_back(0);
          model.ubg.push_back(0);
          model.lbw.push_back(umin[iu]);
          model.ubw.push_back(umax[iu]);
          model.w0.push_back(uinit[iu]);
        }
        */

      }
    }


    /// Second add the non-critical scenarios

    for (int is = 0; is < ns; ++is) {
      if (is > 0 and std::find(worst_case.begin(), worst_case.end(), is) == worst_case.end()) {
        //if (is != worst_case and is > 0) {   // this excludes the nominal case
        //if (is != worst_case ) {  //  this includes the nominal case
        cout << "checkpoint s6" << endl;

        cout << "checkpoint non-critical case added" << endl;

        // nominal+non-critical scenarios
        //cout << "checkpoint s6.0" << endl;

        Uk_prev = MX::vertcat({Finit, QKinit});
        cout << "Uk_prev = " << Uk_prev << endl;

        //cout << "checkpoint s6.1" << endl;
        int left_u, right_u, left_x, right_x;
        for (int k = 0; k < horizon_length; ++k) {
          //cout << "print out delta_s = " << delta_s << endl;

          left_u  = nx * (k + 1) + nu * k + nx * d * k;
          right_u = nx * (k + 1) + nu * (k + 1) + nx * d * k;
          //cout << "print out slice left and right index = " << left_u << ", " <<  right_u << endl;
          //cout << "print out corresponding delta_s = " <<  delta_s[is](Slice(left_u, right_u)) << endl;
          Uk_sens[k] = Uk0[k] + delta_s[is](Slice(left_u, right_u));
          //cout << "checkpoint s6.01" << endl;

          for (int j = 0; j < d; ++j) {


            // We need to update Xkj_sens and Uk_sens for each scenario
            left_x = nx * (k + 1) + nu * (k + 1) + nx * (j + d * k);
            right_x = nx * (k + 1) + nu * (k + 1) + nx * (j + 1 + d * k);

            Xkj_sens[k].push_back(Xkj[k][j] + delta_s[is](Slice(left_x, right_x)));

            //cout << "checkpoint s6.2" << endl;

            // Append collocation equations
            vector<MX> XUprev_sens{Xkj_sens[k][j], Uk_sens[k], Uk_prev};


            MX Lj_sens = f_L(XUprev_sens)[0];



            // Add contribution to quadrature function
            Cost_sens += B[j + 1] * Lj_sens * h;
            //cout << "checkpoint s6.3" << endl;


          } // collocation

          // update the previous u
          Uk_prev = Uk_sens[k];

        } // horizon



        /// NAC
        /*  not needed if sensitivity already takes care of NAC
        // Robust horizon = 1
        if (is != 0) {
          model.g.push_back(Uk[0][0] - Uk[is][0]);

          for (int iu = 0; iu < nu; ++iu) {
            model.lbg.push_back(0);
            model.ubg.push_back(0);
          }
        }
         */

        // Robust horizon = 1
        //model.g.push_back(Uk_sens[0] - dummy_u);

        //for (int iu = 0; iu < nu; ++iu) {
        //  model.lbg.push_back(0);
        //  model.ubg.push_back(0);
        //}


      }

    }



    /// finally add the worst cases
    for (int is = 0; is < ns; ++is) {
      // if is is in the worst_case vector
      if (is != 0 and std::find(worst_case.begin(), worst_case.end(), is) != worst_case.end() ) {

        cout << "checkpoint worst case added" << endl;

        /// "lift" initial conditions
        Xk[0] = MX::sym("x0^" + str(is), nx);
        //states.push_back(Xk[is]);
        model.w.push_back(Xk[0]);
        model.g.push_back(Xk[0] - p_xinit);
        for (int iw = 0; iw < nx; ++iw) {
          //lbw.push_back(xinit[iw]);
          //ubw.push_back(xinit[iw]);
          model.lbw.push_back(xmin[iw]);
          model.ubw.push_back(xmax[iw]);
          model.lbg.push_back(0);
          model.ubg.push_back(0);
          model.w0.push_back(xinit1[iw]);
        }

        //Uk_prev[is] = MX::sym("u^" + str(is) + "_prev", nu);
        //Uk_prev[is] = MX::vertcat({Finit, QKinit});

        //cout << "Uk_prev = " << Uk_prev[is] << endl;

        Uk_prev = MX::vertcat({Finit, QKinit});


        /// Formulate the NLP
        for (int k = 0; k < horizon_length; ++k) {
          // New NLP variable for the control
          Uk[k] = MX::sym("U^" + str(is) + "_" + str(k), nu);
          //controls.push_back(Uk[is][k]);
          cout << "U = " << Uk[k] << endl;
          model.w.push_back(Uk[k]);
          for (int iu = 0; iu < nu; ++iu) {
            model.lbw.push_back(umin[iu]);
            model.ubw.push_back(umax[iu]);
            model.w0.push_back(uinit[iu]);
          }


          // State at collocation points
          for (int j = 0; j < d; ++j) {
            Xkj[k][j] = MX::sym("X^" + str(is) + "_" + str(k) + "_" + str(j + 1), nx);
            model.w.push_back(Xkj[k][j]);
            for (int iw = 0; iw < nx; ++iw) {
              model.lbw.push_back(xmin[iw]);
              model.ubw.push_back(xmax[iw]);
              model.w0.push_back(xinit1[iw]);
            }
          }


          // Loop over collocation points
          Xk_end[k] = D[0] * Xk[k];


          for (int j = 0; j < d; ++j) {
            // Expression for the state derivative at the collocation point
            MX xp = C[0][j + 1] * Xk[k];

            for (int r = 0; r < d; ++r) {
              xp += C[r + 1][j + 1] * Xkj[k][r];
            }

            vector<MX> XUprev{Xkj[k][j], Uk[k], Uk_prev};
            vector<MX> XU{Xkj[k][j], Uk[k], param[is]};
            MX fj = f_xdot(XU)[0];
            MX Lj = f_L(XUprev)[0];

            //cout << "fj = " << fj << endl;
            //cout << "Lj = " << Lj << endl;

            model.g.push_back(h * fj - xp);
            for (int iw = 0; iw < nx; ++iw) {
              model.lbg.push_back(0);
              model.ubg.push_back(0);
            }

            // Add contribution to the end state
            Xk_end[k] += D[j + 1] * Xkj[k][j];

            // Add contribution to quadrature function
            model.Cost += B[j + 1] * Lj * h;
          }


          // New NLP variable for state at end of interval
          Xk[k+1] = MX::sym("X^" + str(is) + "_" + str(k + 1), nx);
          model.w.push_back(Xk[k+1]);
          for (int iw = 0; iw < nx; ++iw) {
            model.lbw.push_back(xmin[iw]);
            model.ubw.push_back(xmax[iw]);
            model.w0.push_back(xinit1[iw]);
          }

          // Add equality constraint
          // for continuity between intervals
          model.g.push_back(Xk_end[k] - Xk[k+1]);
          for (int iw = 0; iw < nx; ++iw) {
            model.lbg.push_back(0);
            model.ubg.push_back(0);
          }
          Uk_prev = Uk[k];

          cout << "Uk_prev = " << Uk_prev << endl;
        }

        /// NAC
        // Robust horizon = 1
        //model.g.push_back(Uk[0] - dummy_u);
        model.g.push_back(Uk[0] - Uk0[0]);

        for (int iu = 0; iu < nu; ++iu) {
          model.lbg.push_back(0);
          model.ubg.push_back(0);
        }

      }
    }




    //cout << "checkpoint s7" << endl;

    model.Cost += Cost_sens;
    // model.Cost = model.Cost / ns;

    cout << "print total cost function = " << model.Cost << endl;
    //cout << "print total variables  = "  << model.w << endl;
    //cout << "print total constraint  = " << model.g << endl;



    model.p_uncertain = p_xinit;

    return model;
  }


}