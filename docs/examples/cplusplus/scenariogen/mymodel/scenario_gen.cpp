//
// Created by Zhou Yu on 9/30/19.
//

#include <casadi/casadi.hpp>
#include <casadi/core/sensitivity.hpp>
#include <casadi/core/timing.hpp>
#include "cstr_model.hpp"
#include "basic_nlp_helper.cpp"


namespace casadi {


  nlp_setup scenario_gen(double time_horizon, int horizon_length, MX p_xinit, MX p, vector<MX> param,
                                 vector<double> xinit0, const vector<vector<double>> &p_c,
                                 int nx, int nu, int np, int d, int ns, int index_k) {

    /// step 1 - nominal scenario - parameterize in uncertainty_p
    cout << "checkpoint -1" << endl;

    /// change the base_index to decide the base scenario
    int base_index = 0;
    nlp_setup nominal = nmpc_nominal_p(time_horizon, horizon_length, xinit0, p, p_c[base_index], index_k);

    //cout << " print out nlp = " << nominal.nlp << endl;
    //cout << " print out arg = " << nominal.arg << endl;

    cout << "checkpoint 0" << endl;
    // solve for nominal solution
    auto res_nom = nominal.solver(nominal.arg);

    //cout << "res_nom = " << res_nom.at("x") << endl;

    int N_tot = res_nom.at("x").size1();
    //int g_tot = res_nom.at("g").size1();

    cout << "checkpoint 1" << endl;
    cout << "N_tot = " << N_tot << endl;


    MXDict nlp_gen = nominal.nlp;
    //Function solver_gen = nominal.solver;
    //std::map<std::string, DM> arg = nominal.arg;

    /// Compute the sign of dg for inequality constraints
    vector<double> param_num;
    param_num.push_back(double(param[0](0)));
    param_num.push_back(double(param[0](1)));
    //param_num.push_back(double(param[1](0)));
    //param_num.push_back(double(param[1](1)));

    //DM dg_p0 = getDg(res_nom, nlp_gen, param_num);

    DM dg_p0 = getDg(res_nom, nlp_gen, p_c[0]);
    cout << "dg_p0 = " << dg_p0 << endl;

    cout << "checkpoint 2" << endl;

    ///****************************************************
    /// Step 2
    /// scenario generation with Schur-complement
    int nr = 1;

    vector<int> NAC_Ctrl_index;
    for (int i = 0; i < nr; ++i) {
      NAC_Ctrl_index.push_back(nx * (i + 1) + nx * d * i + nu * i);
    };


    /// NAC_Ctrl_index already considers for all the u's
    cout << "NAC index list = " << NAC_Ctrl_index << endl;

    /// scenario construction
    int n, m;
    //n = nominal.arg["lbx"].size1();
    //m = nominal.arg["lbw"].size1();
    n = N_tot;
    m = res_nom.at("g").size1();
    int length = n + m;

    cout << "total_length = " << length << endl;


    cout << "checkpoint 3" << endl;

    // TODO
    // generate a function that computes mNAC from nr (robust horizon length)
    int mNAC = 2;
    vector<DM> N(ns);
    N[0] = DM(length, mNAC * nu);
    N[1] = DM(length, mNAC * nu);
    N[2] = DM(length, mNAC * nu);
    for (int i:NAC_Ctrl_index) {
      for (int j = 0; j < nu; ++j) {
        //for (int s = 0; s < ns; ++s) {
        // scenario 1
        // need NACs for NAC constraint index 0 and 1
        // index 0 = NAC constraint 0, index 1 = NAC constraint 1
        N[0](i + j, 0 + j * mNAC) = 1;
        N[0](i + j, 1 + j * mNAC) = 1;
        // scenario 2 only for NAC index 0
        N[1](i + j, 0 + j * mNAC) = -1;
        // scenario 3 only for NAC index 1
        N[2](i + j, 1 + j * mNAC) = -1;
        // }
      }
    }

    cout << "all N blocks = " << N << endl;


    /// construct the multiplier gamma vector
    DM gamma(mNAC * nu, 1);

    // fetch the KKT matrix and RHS for each scenario
    vector<vector<DM>> KR(ns);
    vector<DM> K(ns), R(ns);
    for (int is = 0; is < ns; ++is) {
      KR[is] = getKKTaRHS(res_nom, nlp_gen, p_c[base_index], p_c[is]);
      K[is] = KR[is][0];
      R[is] = KR[is][1];
    }


    //cout << "check on K matrix = " << K << endl;
    //cout << "check on R matrix = " << R << endl;


    ///start counting time
    FStats NACtime;
    NACtime.tic();


    /*
    /// try assembling the KKT matrix before solve

    Linsol linear_solver = Linsol("linear_solver", "ma27", K[0].sparsity());

    vector<Linsol> linearsolver(ns);
    for (int is = 0; is < ns; ++is) {
      linearsolver[is] = Linsol("linear_solver", "ma27", K[is].sparsity());
      //linearsolver[is].nfact(K[is]);
    }
    */




    // LHS
    vector<DM> KiN(ns);
    DM totLHS(mNAC * nu, mNAC * nu);
    for (int is = 0; is < ns; ++is) {
      //KiN[is] = linearsolver[is].solve(K[is], N[is]);
      KiN[is] = solve(K[is], N[is], "ma27");
      totLHS += mtimes(N[is].T(), KiN[is]);
    }


    //auto linear_solver = Linsol("linear_solver", "ma27", K1.sparsity());
    //linear_solver.sfact(K1);

    // RHS
    vector<DM> Kir(ns);
    DM totRHS(mNAC * nu, 1);
    for (int is = 0; is < ns; ++is) {
      // Kir[is] = linearsolver[is].solve(K[is], R[is]);
      Kir[is] = solve(K[is], R[is], "ma27");
      totRHS += mtimes(N[is].T(), Kir[is]);
    }

    gamma = solve(totLHS, -totRHS, "ma27");

    cout << "gamma = " << gamma << endl;

    vector<DM> schurR(ns);
    vector<DM> ds(ns), s_c(ns), s_pert;
    for (int is = 0; is < ns; ++is) {
      schurR[is] = R[is] + mtimes(N[is], gamma);
      // ds[is] = linearsolver[is].solve(K[is], -schurR[is]);
      ds[is] = solve(K[is], -schurR[is], "ma27");
    }


    /// end time
    NACtime.toc();
    cout << "total t_wall time = " << NACtime.t_wall << endl;
    cout << "total t_proc time = " << NACtime.t_proc << endl;


    DM s = DM::vertcat({res_nom.at("x"), res_nom.at("lam_g")});
    //cout << "nominal scenario optimal solution s = " << s << endl;

    for (int is = 0; is < ns; ++is) {
      s_c[is] = s + ds[is];
      s_pert.push_back(s_c[is]);
    }


    vector<DM> CA_pert(ns), CB_pert(ns), TR_pert(ns), TK_pert(ns), F_pert(ns), QK_pert(ns);
    vector<DM> CA_ds(ns), CB_ds(ns), TR_ds(ns), TK_ds(ns), F_ds(ns), QK_ds(ns);

    for (int is = 0; is < ns; ++is) {

      CA_pert[is] = s_pert[is](Slice(0, N_tot, nu + nx + nx * d));
      CB_pert[is] = s_pert[is](Slice(1, N_tot, nu + nx + nx * d));
      TR_pert[is] = s_pert[is](Slice(2, N_tot, nu + nx + nx * d));
      TK_pert[is] = s_pert[is](Slice(3, N_tot, nu + nx + nx * d));
      F_pert[is] = s_pert[is](Slice(4, N_tot, nu + nx + nx * d));
      QK_pert[is] = s_pert[is](Slice(5, N_tot, nu + nx + nx * d));


      CA_ds[is] = ds[is](Slice(0, N_tot, nu + nx + nx * d));
      CB_ds[is] = ds[is](Slice(1, N_tot, nu + nx + nx * d));
      TR_ds[is] = ds[is](Slice(2, N_tot, nu + nx + nx * d));
      TK_ds[is] = ds[is](Slice(3, N_tot, nu + nx + nx * d));
      F_ds[is] = ds[is](Slice(4, N_tot, nu + nx + nx * d));
      QK_ds[is] = ds[is](Slice(5, N_tot, nu + nx + nx * d));

      cout << setw(30) << " For scenario s = " << is << endl;
      cout << setw(30) << "ds(CA): " << CA_ds[is] << endl;
      cout << setw(30) << "ds(CB): " << CB_ds[is] << endl;
      cout << setw(30) << "ds(TR): " << TR_ds[is] << endl;
      cout << setw(30) << "ds(TK): " << TK_ds[is] << endl;
      cout << setw(30) << "ds(F):  " << F_ds[is] << endl;
      cout << setw(30) << "ds(QK): " << QK_ds[is] << endl;


      cout << setw(30) << "Perturbed solution (CA): " << CA_pert[is] << endl;
      cout << setw(30) << "Perturbed solution (CB): " << CB_pert[is] << endl;
      cout << setw(30) << "Perturbed solution (TR): " << TR_pert[is] << endl;
      cout << setw(30) << "Perturbed solution (TK): " << TK_pert[is] << endl;
      cout << setw(30) << "Perturbed solution (F):  " << F_pert[is] << endl;
      cout << setw(30) << "Perturbed solution (QK): " << QK_pert[is] << endl;

    }

    /// step 2.5
    /// have the index list for worse case scenarios
    vector<int> worst_case{1};
    //if (index_k >= 2) {
    //  worst_case = {0};
    //}


    /// Step 3
    /// Calculate the approximate multistage solutions based on sensitivity
    cout << "checkpoint 4" << endl;

    nlp_setup step3 = gen_step3(time_horizon, horizon_length, p_xinit, param, xinit0, ns, worst_case, ds, index_k);
    return step3;




  }





  nlp_setup scenario_gen_9c(double time_horizon, int horizon_length, MX p_xinit, MX p, vector<MX> param,
                         vector<double> xinit0, const vector<vector<double>> &p_c,
                         int nx, int nu, int np, int d, int ns, int index_k) {

    /// step 1 - nominal scenario - parameterize in uncertainty_p
    cout << "checkpoint -1" << endl;

    /// change the base_index to decide the base scenario
    int base_index = 0;
    nlp_setup nominal = nmpc_nominal_p(time_horizon, horizon_length, xinit0, p, p_c[base_index], index_k);

    //cout << " print out nlp = " << nominal.nlp << endl;
    //cout << " print out arg = " << nominal.arg << endl;

    cout << "checkpoint 0" << endl;
    // solve for nominal solution
    auto res_nom = nominal.solver(nominal.arg);

    //cout << "res_nom = " << res_nom.at("x") << endl;

    int N_tot = res_nom.at("x").size1();
    //int g_tot = res_nom.at("g").size1();

    cout << "checkpoint 1" << endl;
    cout << "N_tot = " << N_tot << endl;


    MXDict nlp_gen = nominal.nlp;
    //Function solver_gen = nominal.solver;
    //std::map<std::string, DM> arg = nominal.arg;

    /// Compute the sign of dg for inequality constraints
    vector<double> param_num;
    param_num.push_back(double(param[0](0)));
    param_num.push_back(double(param[0](1)));
    //param_num.push_back(double(param[1](0)));
    //param_num.push_back(double(param[1](1)));

    //DM dg_p0 = getDg(res_nom, nlp_gen, param_num);

    DM dg_p0 = getDg(res_nom, nlp_gen, p_c[0]);
    cout << "dg_p0 = " << dg_p0 << endl;

    cout << "checkpoint 2" << endl;

    ///****************************************************
    /// Step 2
    /// scenario generation with Schur-complement
    int nr = 1;

    vector<int> NAC_Ctrl_index;
    for (int i = 0; i < nr; ++i) {
      NAC_Ctrl_index.push_back(nx * (i + 1) + nx * d * i + nu * i);
    };


    /// NAC_Ctrl_index already considers for all the u's
    cout << "NAC index list = " << NAC_Ctrl_index << endl;

    /// scenario construction
    int n, m;
    //n = nominal.arg["lbx"].size1();
    //m = nominal.arg["lbw"].size1();
    n = N_tot;
    m = res_nom.at("g").size1();
    int length = n + m;

    cout << "total_length = " << length << endl;


    cout << "checkpoint 3" << endl;

    // TODO
    // generate a function that computes mNAC from nr (robust horizon length)
    int mNAC = 8;
    vector<DM> N(ns);

    for (int is = 0; is < ns; ++is) {
      N[is] = DM(length,mNAC*nu);
    }

    /// the following is different for different number of scenarios and robust horizons
    for (int i:NAC_Ctrl_index) {
      for (int j = 0; j < nu; ++j) {
        //for (int s = 0; s < ns; ++s) {
        // scenario 1
        // need NACs for NAC constraint index 0 and 1
        // index 0 = NAC constraint 0, index 1 = NAC constraint 1
        N[0](i + j, 0 + j*mNAC) = 1;
        N[0](i + j, 1 + j*mNAC) = 1;
        N[0](i + j, 2 + j*mNAC) = 1;
        N[0](i + j, 3 + j*mNAC) = 1;
        N[0](i + j, 4 + j*mNAC) = 1;
        N[0](i + j, 5 + j*mNAC) = 1;
        N[0](i + j, 6 + j*mNAC) = 1;
        N[0](i + j, 7 + j*mNAC) = 1;



        // scenario 2 only for NAC index 0
        N[1](i + j, 0 + j*mNAC) = -1;
        // scenario 3 only for NAC index 1
        N[2](i + j, 1 + j*mNAC) = -1;

        N[3](i + j, 2 + j*mNAC) = -1;
        N[4](i + j, 3 + j*mNAC) = -1;
        N[5](i + j, 4 + j*mNAC) = -1;
        N[6](i + j, 5 + j*mNAC) = -1;
        N[7](i + j, 6 + j*mNAC) = -1;
        N[8](i + j, 7 + j*mNAC) = -1;




        // }

      }


    }

    cout << "all N blocks = " << N << endl;


    /// construct the multiplier gamma vector
    DM gamma(mNAC * nu, 1);

    // fetch the KKT matrix and RHS for each scenario
    vector<vector<DM>> KR(ns);
    vector<DM> K(ns), R(ns);
    for (int is = 0; is < ns; ++is) {
      KR[is] = getKKTaRHS(res_nom, nlp_gen, p_c[base_index], p_c[is]);
      K[is] = KR[is][0];
      R[is] = KR[is][1];
    }


    //cout << "check on K matrix = " << K << endl;
    //cout << "check on R matrix = " << R << endl;


    ///start counting time
    FStats NACtime;
    NACtime.tic();


    /*
    /// try assembling the KKT matrix before solve

    Linsol linear_solver = Linsol("linear_solver", "ma27", K[0].sparsity());

    vector<Linsol> linearsolver(ns);
    for (int is = 0; is < ns; ++is) {
      linearsolver[is] = Linsol("linear_solver", "ma27", K[is].sparsity());
      //linearsolver[is].nfact(K[is]);
    }
    */




    // LHS
    vector<DM> KiN(ns);
    DM totLHS(mNAC * nu, mNAC * nu);
    for (int is = 0; is < ns; ++is) {
      //KiN[is] = linearsolver[is].solve(K[is], N[is]);
      KiN[is] = solve(K[is], N[is], "ma27");
      totLHS += mtimes(N[is].T(), KiN[is]);
    }


    //auto linear_solver = Linsol("linear_solver", "ma27", K1.sparsity());
    //linear_solver.sfact(K1);

    // RHS
    vector<DM> Kir(ns);
    DM totRHS(mNAC * nu, 1);
    for (int is = 0; is < ns; ++is) {
      // Kir[is] = linearsolver[is].solve(K[is], R[is]);
      Kir[is] = solve(K[is], R[is], "ma27");
      totRHS += mtimes(N[is].T(), Kir[is]);
    }

    gamma = solve(totLHS, -totRHS, "ma27");

    cout << "gamma = " << gamma << endl;

    vector<DM> schurR(ns);
    vector<DM> ds(ns), s_c(ns), s_pert;
    for (int is = 0; is < ns; ++is) {
      schurR[is] = R[is] + mtimes(N[is], gamma);
      // ds[is] = linearsolver[is].solve(K[is], -schurR[is]);
      ds[is] = solve(K[is], -schurR[is], "ma27");
    }


    /// end time
    NACtime.toc();
    cout << "total t_wall time = " << NACtime.t_wall << endl;
    cout << "total t_proc time = " << NACtime.t_proc << endl;


    DM s = DM::vertcat({res_nom.at("x"), res_nom.at("lam_g")});
    //cout << "nominal scenario optimal solution s = " << s << endl;

    for (int is = 0; is < ns; ++is) {
      s_c[is] = s + ds[is];
      s_pert.push_back(s_c[is]);
    }


    vector<DM> CA_pert(ns), CB_pert(ns), TR_pert(ns), TK_pert(ns), F_pert(ns), QK_pert(ns);
    vector<DM> CA_ds(ns), CB_ds(ns), TR_ds(ns), TK_ds(ns), F_ds(ns), QK_ds(ns);

    for (int is = 0; is < ns; ++is) {

      CA_pert[is] = s_pert[is](Slice(0, N_tot, nu + nx + nx * d));
      CB_pert[is] = s_pert[is](Slice(1, N_tot, nu + nx + nx * d));
      TR_pert[is] = s_pert[is](Slice(2, N_tot, nu + nx + nx * d));
      TK_pert[is] = s_pert[is](Slice(3, N_tot, nu + nx + nx * d));
      F_pert[is] = s_pert[is](Slice(4, N_tot, nu + nx + nx * d));
      QK_pert[is] = s_pert[is](Slice(5, N_tot, nu + nx + nx * d));


      CA_ds[is] = ds[is](Slice(0, N_tot, nu + nx + nx * d));
      CB_ds[is] = ds[is](Slice(1, N_tot, nu + nx + nx * d));
      TR_ds[is] = ds[is](Slice(2, N_tot, nu + nx + nx * d));
      TK_ds[is] = ds[is](Slice(3, N_tot, nu + nx + nx * d));
      F_ds[is] = ds[is](Slice(4, N_tot, nu + nx + nx * d));
      QK_ds[is] = ds[is](Slice(5, N_tot, nu + nx + nx * d));

      cout << setw(30) << " For scenario s = " << is << endl;
      cout << setw(30) << "ds(CA): " << CA_ds[is] << endl;
      cout << setw(30) << "ds(CB): " << CB_ds[is] << endl;
      cout << setw(30) << "ds(TR): " << TR_ds[is] << endl;
      cout << setw(30) << "ds(TK): " << TK_ds[is] << endl;
      cout << setw(30) << "ds(F):  " << F_ds[is] << endl;
      cout << setw(30) << "ds(QK): " << QK_ds[is] << endl;


      cout << setw(30) << "Perturbed solution (CA): " << CA_pert[is] << endl;
      cout << setw(30) << "Perturbed solution (CB): " << CB_pert[is] << endl;
      cout << setw(30) << "Perturbed solution (TR): " << TR_pert[is] << endl;
      cout << setw(30) << "Perturbed solution (TK): " << TK_pert[is] << endl;
      cout << setw(30) << "Perturbed solution (F):  " << F_pert[is] << endl;
      cout << setw(30) << "Perturbed solution (QK): " << QK_pert[is] << endl;

    }

    /// step 2.5
    /// have the index list for worse case scenarios
    vector<int> worst_case{ 6, 7};
    //if (index_k >= 2) {
    //  worst_case = {0};
    //}


    /// Step 3
    /// Calculate the approximate multistage solutions based on sensitivity
    cout << "checkpoint 4" << endl;

    nlp_setup step3 = gen_step3(time_horizon, horizon_length, p_xinit, param, xinit0, ns, worst_case, ds, index_k);
    return step3;




  }


}  // casadi