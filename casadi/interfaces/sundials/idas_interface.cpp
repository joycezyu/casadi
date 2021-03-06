/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#include "idas_interface.hpp"
#include "casadi/core/casadi_misc.hpp"

// Macro for error handling
#define THROWING(fcn, ...) \
idas_error(CASADI_STR(fcn), fcn(__VA_ARGS__))

using namespace std;
namespace casadi {

  extern "C"
  int CASADI_INTEGRATOR_IDAS_EXPORT
      casadi_register_integrator_idas(Integrator::Plugin* plugin) {
    plugin->creator = IdasInterface::creator;
    plugin->name = "idas";
    plugin->doc = IdasInterface::meta_doc.c_str();
    plugin->version = CASADI_VERSION;
    plugin->options = &IdasInterface::options_;
    return 0;
  }

  extern "C"
  void CASADI_INTEGRATOR_IDAS_EXPORT casadi_load_integrator_idas() {
    Integrator::registerPlugin(casadi_register_integrator_idas);
  }

  IdasInterface::IdasInterface(const std::string& name, const Function& dae)
    : SundialsInterface(name, dae) {
  }

  IdasInterface::~IdasInterface() {
    clear_mem();
  }

  Options IdasInterface::options_
  = {{&SundialsInterface::options_},
     {{"suppress_algebraic",
       {OT_BOOL,
        "Suppress algebraic variables in the error testing"}},
      {"calc_ic",
       {OT_BOOL,
        "Use IDACalcIC to get consistent initial conditions."}},
      {"calc_icB",
       {OT_BOOL,
        "Use IDACalcIC to get consistent initial conditions for "
        "backwards system [default: equal to calc_ic]."}},
      {"abstolv",
       {OT_DOUBLEVECTOR,
        "Absolute tolerarance for each component"}},
      {"max_step_size",
       {OT_DOUBLE,
        "Maximim step size"}},
      {"first_time",
       {OT_DOUBLE,
        "First requested time as a fraction of the time interval"}},
      {"cj_scaling",
       {OT_BOOL,
        "IDAS scaling on cj for the user-defined linear solver module"}},
      {"init_xdot",
       {OT_DOUBLEVECTOR,
        "Initial values for the state derivatives"}}
     }
  };

  void IdasInterface::init(const Dict& opts) {
    if (verbose_) casadi_message(name_ + "::init");

    // Call the base class init
    SundialsInterface::init(opts);

    // Default options
    cj_scaling_ = true;
    calc_ic_ = true;
    suppress_algebraic_ = false;
    max_step_size_ = 0;

    // Read options
    for (auto&& op : opts) {
      if (op.first=="init_xdot") {
        init_xdot_ = op.second;
      } else if (op.first=="cj_scaling") {
        cj_scaling_ = op.second;
      } else if (op.first=="calc_ic") {
        calc_ic_ = op.second;
      } else if (op.first=="suppress_algebraic") {
        suppress_algebraic_ = op.second;
      } else if (op.first=="max_step_size") {
        max_step_size_ = op.second;
      } else if (op.first=="abstolv") {
        abstolv_ = op.second;
      }
    }

    // Default dependent options
    calc_icB_ = calc_ic_;
    first_time_ = grid_.back();

    // Read dependent options
    for (auto&& op : opts) {
      if (op.first=="calc_icB") {
        calc_icB_ = op.second;
      } else if (op.first=="first_time") {
        first_time_ = op.second;
      }
    }

    create_function("daeF", {"x", "z", "p", "t"}, {"ode", "alg"});
    create_function("quadF", {"x", "z", "p", "t"}, {"quad"});
    create_function("daeB", {"rx", "rz", "rp", "x", "z", "p", "t"}, {"rode", "ralg"});
    create_function("quadB", {"rx", "rz", "rp", "x", "z", "p", "t"}, {"rquad"});

    // Get initial conditions for the state derivatives
    if (init_xdot_.empty()) {
      init_xdot_.resize(nx_, 0);
    } else {
      casadi_assert(
        init_xdot_.size()==nx_,
        "Option \"init_xdot\" has incorrect length. Expecting " + str(nx_) + ", "
        "but got " + str(init_xdot_.size()) + ". "
        "Note that this message may actually be generated by the augmented integrator. "
        "In that case, make use of the 'augmented_options' options "
        "to correct 'init_xdot' for the augmented integrator.");
    }

    // Attach functions for jacobian information
    if (newton_scheme_!=SD_DIRECT || (ns_>0 && second_order_correction_)) {
      create_function("jtimesF",
        {"t", "x", "z", "p", "fwd:x", "fwd:z"},
        {"fwd:ode", "fwd:alg"});
      if (nrx_>0) {
        create_function("jtimesB",
          {"t", "x", "z", "p", "rx", "rz", "rp", "fwd:rx", "fwd:rz"},
          {"fwd:rode", "fwd:ralg"});
      }
    }
  }

  int IdasInterface::res(double t, N_Vector xz, N_Vector xzdot,
                                N_Vector rr, void *user_data) {
    try {
      auto m = to_mem(user_data);
      auto& s = m->self;
      m->arg[0] = NV_DATA_S(xz);
      m->arg[1] = NV_DATA_S(xz)+s.nx_;
      m->arg[2] = m->p;
      m->arg[3] = &t;
      m->res[0] = NV_DATA_S(rr);
      m->res[1] = NV_DATA_S(rr)+s.nx_;
      s.calc_function(m, "daeF");

      // Subtract state derivative to get residual
      casadi_axpy(s.nx_, -1., NV_DATA_S(xzdot), NV_DATA_S(rr));
      return 0;
    } catch(int flag) { // recoverable error
      return flag;
    } catch(exception& e) { // non-recoverable error
      uerr() << "res failed: " << e.what() << endl;
      return -1;
    }
  }

  void IdasInterface::ehfun(int error_code, const char *module, const char *function,
                                   char *msg, void *eh_data) {
    try {
      //auto m = to_mem(eh_data);
      //auto& s = m->self;
      uerr() << msg << endl;
    } catch(exception& e) {
      uerr() << "ehfun failed: " << e.what() << endl;
    }
  }

  int IdasInterface::jtimes(double t, N_Vector xz, N_Vector xzdot, N_Vector rr, N_Vector v,
                                   N_Vector Jv, double cj, void *user_data,
                                   N_Vector tmp1, N_Vector tmp2) {
    try {
      auto m = to_mem(user_data);
      auto& s = m->self;
      m->arg[0] = &t;
      m->arg[1] = NV_DATA_S(xz);
      m->arg[2] = NV_DATA_S(xz)+s.nx_;
      m->arg[3] = m->p;
      m->arg[4] = NV_DATA_S(v);
      m->arg[5] = NV_DATA_S(v)+s.nx_;
      m->res[0] = NV_DATA_S(Jv);
      m->res[1] = NV_DATA_S(Jv)+s.nx_;
      s.calc_function(m, "jtimesF");

      // Subtract state derivative to get residual
      casadi_axpy(s.nx_, -cj, NV_DATA_S(v), NV_DATA_S(Jv));

      return 0;
    } catch(int flag) { // recoverable error
      return flag;
    } catch(exception& e) { // non-recoverable error
      uerr() << "jtimes failed: " << e.what() << endl;
      return -1;
    }
  }

  int IdasInterface::jtimesB(double t, N_Vector xz, N_Vector xzdot, N_Vector xzB,
                                    N_Vector xzdotB, N_Vector resvalB, N_Vector vB, N_Vector JvB,
                                    double cjB, void *user_data,
                                    N_Vector tmp1B, N_Vector tmp2B) {
    try {
      auto m = to_mem(user_data);
      auto& s = m->self;
      m->arg[0] = &t;
      m->arg[1] = NV_DATA_S(xz);
      m->arg[2] = NV_DATA_S(xz)+s.nx_;
      m->arg[3] = m->p;
      m->arg[4] = NV_DATA_S(xzB);
      m->arg[5] = NV_DATA_S(xzB)+s.nrx_;
      m->arg[6] = m->rp;
      m->arg[7] = NV_DATA_S(vB);
      m->arg[8] = NV_DATA_S(vB)+s.nrx_;
      m->res[0] = NV_DATA_S(JvB);
      m->res[1] = NV_DATA_S(JvB) + s.nrx_;
      s.calc_function(m, "jtimesB");

      // Subtract state derivative to get residual
      casadi_axpy(s.nrx_, cjB, NV_DATA_S(vB), NV_DATA_S(JvB));

      return 0;
    } catch(int flag) { // recoverable error
      return flag;
    } catch(exception& e) { // non-recoverable error
      uerr() << "jtimesB failed: " << e.what() << endl;
      return -1;
    }
  }

  int IdasInterface::init_mem(void* mem) const {
    if (SundialsInterface::init_mem(mem)) return 1;
    auto m = to_mem(mem);

    // Create IDAS memory block
    m->mem = IDACreate();
    casadi_assert(m->mem!=nullptr, "IDACreate: Creation failed");

    // Set error handler function
    THROWING(IDASetErrHandlerFn, m->mem, ehfun, m);

    // Set user data
    THROWING(IDASetUserData, m->mem, m);

    // Allocate n-vectors for ivp
    m->xzdot = N_VNew_Serial(nx_+nz_);

    // Initialize Idas
    double t0 = 0;
    N_VConst(0.0, m->xz);
    N_VConst(0.0, m->xzdot);
    IDAInit(m->mem, res, t0, m->xz, m->xzdot);
    if (verbose_) casadi_message("IDA initialized");

    // Include algebraic variables in error testing
    THROWING(IDASetSuppressAlg, m->mem, suppress_algebraic_);

    // Maxinum order for the multistep method
    THROWING(IDASetMaxOrd, m->mem, max_multistep_order_);

    // Set maximum step size
    THROWING(IDASetMaxStep, m->mem, max_step_size_);

    // Initial step size
    if (step0_) THROWING(IDASetInitStep, m->mem, step0_);

    // Maximum order of method
    if (max_order_) THROWING(IDASetMaxOrd, m->mem, max_order_);

    // Coeff. in the nonlinear convergence test
    if (nonlin_conv_coeff_) THROWING(IDASetNonlinConvCoef, m->mem, nonlin_conv_coeff_);

    if (!abstolv_.empty()) {
      // Vector absolute tolerances
      N_Vector nv_abstol = N_VNew_Serial(abstolv_.size());
      copy(abstolv_.begin(), abstolv_.end(), NV_DATA_S(nv_abstol));
      THROWING(IDASVtolerances, m->mem, reltol_, nv_abstol);
      N_VDestroy_Serial(nv_abstol);
    } else {
      // Scalar absolute tolerances
      THROWING(IDASStolerances, m->mem, reltol_, abstol_);
    }

    // Maximum number of steps
    THROWING(IDASetMaxNumSteps, m->mem, max_num_steps_);

    // Set algebraic components
    N_Vector id = N_VNew_Serial(nx_+nz_);
    fill_n(NV_DATA_S(id), nx_, 1);
    fill_n(NV_DATA_S(id)+nx_, nz_, 0);

    // Pass this information to IDAS
    THROWING(IDASetId, m->mem, id);

    // Delete the allocated memory
    N_VDestroy_Serial(id);

    // attach a linear solver
    if (newton_scheme_==SD_DIRECT) {
      // Direct scheme
      IDAMem IDA_mem = IDAMem(m->mem);
      IDA_mem->ida_lmem   = m;
      IDA_mem->ida_lsetup = lsetup;
      IDA_mem->ida_lsolve = lsolve;
      IDA_mem->ida_setupNonNull = TRUE;
    } else {
      // Iterative scheme
      switch (newton_scheme_) {
      case SD_DIRECT: casadi_assert_dev(0);
      case SD_GMRES: THROWING(IDASpgmr, m->mem, max_krylov_); break;
      case SD_BCGSTAB: THROWING(IDASpbcg, m->mem, max_krylov_); break;
      case SD_TFQMR: THROWING(IDASptfqmr, m->mem, max_krylov_); break;
      }
      THROWING(IDASpilsSetJacTimesVecFn, m->mem, jtimes);
      if (use_precon_) THROWING(IDASpilsSetPreconditioner, m->mem, psetup, psolve);
    }

    // Quadrature equations
    if (nq_>0) {

      // Initialize quadratures in IDAS
      THROWING(IDAQuadInit, m->mem, rhsQ, m->q);

      // Should the quadrature errors be used for step size control?
      if (quad_err_con_) {
        THROWING(IDASetQuadErrCon, m->mem, true);

        // Quadrature error tolerances
        // TODO(Joel): vector absolute tolerances
        THROWING(IDAQuadSStolerances, m->mem, reltol_, abstol_);
      }
    }

    if (verbose_) casadi_message("Attached linear solver");

    // Adjoint sensitivity problem
    if (nrx_>0) {
      m->rxzdot = N_VNew_Serial(nrx_+nrz_);
      N_VConst(0.0, m->rxz);
      N_VConst(0.0, m->rxzdot);
    }
    if (verbose_) casadi_message("Initialized adjoint sensitivities");

    // Initialize adjoint sensitivities
    if (nrx_>0) {
      int interpType = interp_==SD_HERMITE ? IDA_HERMITE : IDA_POLYNOMIAL;
      THROWING(IDAAdjInit, m->mem, steps_per_checkpoint_, interpType);
    }

    m->first_callB = true;
    return 0;
  }

  void IdasInterface::reset(IntegratorMemory* mem, double t, const double* _x,
                            const double* _z, const double* _p) const {
    if (verbose_) casadi_message(name_ + "::reset");
    auto m = to_mem(mem);

    // Reset the base classes
    SundialsInterface::reset(mem, t, _x, _z, _p);

    // Re-initialize
    copy(init_xdot_.begin(), init_xdot_.end(), NV_DATA_S(m->xzdot));
    THROWING(IDAReInit, m->mem, grid_.front(), m->xz, m->xzdot);

    // Re-initialize quadratures
    if (nq_>0) THROWING(IDAQuadReInit, m->mem, m->q);

    // Correct initial conditions, if necessary
    if (calc_ic_) {
      THROWING(IDACalcIC, m->mem, IDA_YA_YDP_INIT , first_time_);
      THROWING(IDAGetConsistentIC, m->mem, m->xz, m->xzdot);
    }

    // Re-initialize backward integration
    if (nrx_>0) THROWING(IDAAdjReInit, m->mem);

    // Set the stop time of the integration -- don't integrate past this point
    if (stop_at_end_) setStopTime(m, grid_.back());
  }

  void IdasInterface::
  advance(IntegratorMemory* mem, double t, double* x, double* z, double* q) const {
    auto m = to_mem(mem);

    casadi_assert(t>=grid_.front(),
      "IdasInterface::integrate(" + str(t) + "): "
      "Cannot integrate to a time earlier than t0 (" + str(grid_.front()) + ")");
    casadi_assert(t<=grid_.back() || !stop_at_end_,
      "IdasInterface::integrate(" + str(t) + "): "
      "Cannot integrate past a time later than tf (" + str(grid_.back()) + ") "
      "unless stop_at_end is set to False.");

    // Integrate, unless already at desired time
    double ttol = 1e-9;   // tolerance
    if (fabs(m->t-t)>=ttol) {
      // Integrate forward ...
      if (nrx_>0) { // ... with taping
        THROWING(IDASolveF, m->mem, t, &m->t, m->xz, m->xzdot, IDA_NORMAL, &m->ncheck);
      } else { // ... without taping
        THROWING(IDASolve, m->mem, t, &m->t, m->xz, m->xzdot, IDA_NORMAL);
      }

      // Get quadratures
      if (nq_>0) {
        double tret;
        THROWING(IDAGetQuad, m->mem, &tret, m->q);
      }
    }

    // Set function outputs
    casadi_copy(NV_DATA_S(m->xz), nx_, x);
    casadi_copy(NV_DATA_S(m->xz)+nx_, nz_, z);
    casadi_copy(NV_DATA_S(m->q), nq_, q);

    // Get stats
    THROWING(IDAGetIntegratorStats, m->mem, &m->nsteps, &m->nfevals, &m->nlinsetups,
             &m->netfails, &m->qlast, &m->qcur, &m->hinused,
             &m->hlast, &m->hcur, &m->tcur);
    THROWING(IDAGetNonlinSolvStats, m->mem, &m->nniters, &m->nncfails);

  }

  void IdasInterface::resetB(IntegratorMemory* mem, double t, const double* rx,
                             const double* rz, const double* rp) const {
    if (verbose_) casadi_message(name_ + "::resetB");
    auto m = to_mem(mem);

    // Reset the base classes
    SundialsInterface::resetB(mem, t, rx, rz, rp);

    if (m->first_callB) {
      // Create backward problem
      THROWING(IDACreateB, m->mem, &m->whichB);
      THROWING(IDAInitB, m->mem, m->whichB, resB, grid_.back(), m->rxz, m->rxzdot);
      THROWING(IDASStolerancesB, m->mem, m->whichB, reltol_, abstol_);
      THROWING(IDASetUserDataB, m->mem, m->whichB, m);
      THROWING(IDASetMaxNumStepsB, m->mem, m->whichB, max_num_steps_);

      // Set algebraic components
      N_Vector id = N_VNew_Serial(nrx_+nrz_);
      fill_n(NV_DATA_S(id), nrx_, 1);
      fill_n(NV_DATA_S(id)+nrx_, nrz_, 0);
      THROWING(IDASetIdB, m->mem, m->whichB, id);
      N_VDestroy_Serial(id);

      // attach linear solver
      if (newton_scheme_==SD_DIRECT) {
        // Direct scheme
        IDAMem IDA_mem = IDAMem(m->mem);
        IDAadjMem IDAADJ_mem = IDA_mem->ida_adj_mem;
        IDABMem IDAB_mem = IDAADJ_mem->IDAB_mem;
        IDAB_mem->ida_lmem   = m;
        IDAB_mem->IDA_mem->ida_lmem = m;
        IDAB_mem->IDA_mem->ida_lsetup = lsetupB;
        IDAB_mem->IDA_mem->ida_lsolve = lsolveB;
        IDAB_mem->IDA_mem->ida_setupNonNull = TRUE;
      } else {
        // Iterative scheme
        switch (newton_scheme_) {
        case SD_DIRECT: casadi_assert_dev(0);
        case SD_GMRES: THROWING(IDASpgmrB, m->mem, m->whichB, max_krylov_); break;
        case SD_BCGSTAB: THROWING(IDASpbcgB, m->mem, m->whichB, max_krylov_); break;
        case SD_TFQMR: THROWING(IDASptfqmrB, m->mem, m->whichB, max_krylov_); break;
        }
        THROWING(IDASpilsSetJacTimesVecFnB, m->mem, m->whichB, jtimesB);
        if (use_precon_) THROWING(IDASpilsSetPreconditionerB, m->mem, m->whichB, psetupB, psolveB);
      }

      // Quadratures for the adjoint problem
      THROWING(IDAQuadInitB, m->mem, m->whichB, rhsQB, m->rq);
      if (quad_err_con_) {
        THROWING(IDASetQuadErrConB, m->mem, m->whichB, true);
        THROWING(IDAQuadSStolerancesB, m->mem, m->whichB, reltol_, abstol_);
      }

      // Mark initialized
      m->first_callB = false;
    } else {
      // Re-initialize
      THROWING(IDAReInitB, m->mem, m->whichB, grid_.back(), m->rxz, m->rxzdot);
      if (nrq_>0) {
        // Workaround (bug in SUNDIALS)
        // THROWING(IDAQuadReInitB, m->mem, m->whichB[dir], m->rq[dir]);
        void* memB = IDAGetAdjIDABmem(m->mem, m->whichB);
        THROWING(IDAQuadReInit, memB, m->rq);
      }
    }

    // Correct initial values for the integration if necessary
    if (calc_icB_) {
      THROWING(IDACalcICB, m->mem, m->whichB, grid_.front(), m->xz, m->xzdot);
      THROWING(IDAGetConsistentICB, m->mem, m->whichB, m->rxz, m->rxzdot);
    }
  }

  void IdasInterface::retreat(IntegratorMemory* mem, double t, double* rx,
                              double* rz, double* rq) const {
    auto m = to_mem(mem);

    // Integrate, unless already at desired time
    if (t<m->t) {
      THROWING(IDASolveB, m->mem, t, IDA_NORMAL);
      THROWING(IDAGetB, m->mem, m->whichB, &m->t, m->rxz, m->rxzdot);
      if (nrq_>0) {
        THROWING(IDAGetQuadB, m->mem, m->whichB, &m->t, m->rq);
      }
    }

    // Save outputs
    casadi_copy(NV_DATA_S(m->rxz), nrx_, rx);
    casadi_copy(NV_DATA_S(m->rxz)+nrx_, nrz_, rz);
    casadi_copy(NV_DATA_S(m->rq), nrq_, rq);

    // Get stats
    IDAMem IDA_mem = IDAMem(m->mem);
    IDAadjMem IDAADJ_mem = IDA_mem->ida_adj_mem;
    IDABMem IDAB_mem = IDAADJ_mem->IDAB_mem;
    THROWING(IDAGetIntegratorStats, IDAB_mem->IDA_mem, &m->nstepsB, &m->nfevalsB,
             &m->nlinsetupsB, &m->netfailsB, &m->qlastB, &m->qcurB, &m->hinusedB,
             &m->hlastB, &m->hcurB, &m->tcurB);
    THROWING(IDAGetNonlinSolvStats, IDAB_mem->IDA_mem, &m->nnitersB, &m->nncfailsB);
  }

  void IdasInterface::idas_error(const char* module, int flag) {
    // Successfull return or warning
    if (flag>=IDA_SUCCESS) return;
    // Construct error message
    char* flagname = IDAGetReturnFlagName(flag);
    stringstream ss;
    ss << module << " returned \"" << flagname << "\". Consult IDAS documentation.";
    free(flagname);
    casadi_error(ss.str());
  }

  int IdasInterface::rhsQ(double t, N_Vector xz, N_Vector xzdot, N_Vector rhsQ,
                                 void *user_data) {
    try {
      auto m = to_mem(user_data);
      auto& s = m->self;
      m->arg[0] = NV_DATA_S(xz);
      m->arg[1] = NV_DATA_S(xz)+s.nx_;
      m->arg[2] = m->p;
      m->arg[3] = &t;
      m->res[0] = NV_DATA_S(rhsQ);
      s.calc_function(m, "quadF");

      return 0;
    } catch(int flag) { // recoverable error
      return flag;
    } catch(exception& e) { // non-recoverable error
      uerr() << "rhsQ failed: " << e.what() << endl;
      return -1;
    }
  }

  int IdasInterface::resB(double t, N_Vector xz, N_Vector xzdot, N_Vector rxz,
                                 N_Vector rxzdot, N_Vector rr, void *user_data) {
    try {
      auto m = to_mem(user_data);
      auto& s = m->self;
      m->arg[0] = NV_DATA_S(rxz);
      m->arg[1] = NV_DATA_S(rxz)+s.nrx_;
      m->arg[2] = m->rp;
      m->arg[3] = NV_DATA_S(xz);
      m->arg[4] = NV_DATA_S(xz)+s.nx_;
      m->arg[5] = m->p;
      m->arg[6] = &t;
      m->res[0] = NV_DATA_S(rr);
      m->res[1] = NV_DATA_S(rr)+s.nrx_;
      s.calc_function(m, "daeB");

      // Subtract state derivative to get residual
      casadi_axpy(s.nrx_, 1., NV_DATA_S(rxzdot), NV_DATA_S(rr));

      return 0;
    } catch(int flag) { // recoverable error
      return flag;
    } catch(exception& e) { // non-recoverable error
      uerr() << "resB failed: " << e.what() << endl;
      return -1;
    }
  }

  int IdasInterface::rhsQB(double t, N_Vector xz, N_Vector xzdot, N_Vector rxz,
                                  N_Vector rxzdot, N_Vector rqdot, void *user_data) {
    try {
      auto m = to_mem(user_data);
      auto& s = m->self;
      m->arg[0] = NV_DATA_S(rxz);
      m->arg[1] = NV_DATA_S(rxz)+s.nrx_;
      m->arg[2] = m->rp;
      m->arg[3] = NV_DATA_S(xz);
      m->arg[4] = NV_DATA_S(xz)+s.nx_;
      m->arg[5] = m->p;
      m->arg[6] = &t;
      m->res[0] = NV_DATA_S(rqdot);
      s.calc_function(m, "quadB");

      // Negate (note definition of g)
      casadi_scal(s.nrq_, -1., NV_DATA_S(rqdot));

      return 0;
    } catch(int flag) { // recoverable error
      return flag;
    } catch(exception& e) { // non-recoverable error
      uerr() << "resQB failed: " << e.what() << endl;
      return -1;
    }
  }

  void IdasInterface::setStopTime(IntegratorMemory* mem, double tf) const {
    // Set the stop time of the integration -- don't integrate past this point
    auto m = to_mem(mem);
    //auto& s = m->self;
    THROWING(IDASetStopTime, m->mem, tf);
  }

  int IdasInterface::psolve(double t, N_Vector xz, N_Vector xzdot, N_Vector rr,
                                    N_Vector rvec, N_Vector zvec, double cj, double delta,
                                    void *user_data, N_Vector tmp) {
    try {
      auto m = to_mem(user_data);
      auto& s = m->self;

      // Get right-hand sides in m->v1, ordered by sensitivity directions
      double* vx = NV_DATA_S(rvec);
      double* vz = vx + s.nx_;
      double* v_it = m->v1;
      for (int d=0; d<=s.ns_; ++d) {
        casadi_copy(vx + d*s.nx1_, s.nx1_, v_it);
        v_it += s.nx1_;
        casadi_copy(vz + d*s.nz1_, s.nz1_, v_it);
        v_it += s.nz1_;
      }

      // Solve for undifferentiated right-hand-side, save to output
      if (s.linsolF_.solve(m->jac, m->v1, 1, false, m->mem_linsolF))
        casadi_error("'jac' solve failed");
      vx = NV_DATA_S(zvec); // possibly different from rvec
      vz = vx + s.nx_;
      casadi_copy(m->v1, s.nx1_, vx);
      casadi_copy(m->v1 + s.nx1_, s.nz1_, vz);

      // Sensitivity equations
      if (s.ns_>0) {
        // Second order correction
        if (s.second_order_correction_) {
          // The outputs will double as seeds for jtimesF
          casadi_fill(vx + s.nx1_, s.nx_ - s.nx1_, 0.);
          casadi_fill(vz + s.nz1_, s.nz_ - s.nz1_, 0.);
          m->arg[0] = &t; // t
          m->arg[1] = NV_DATA_S(xz); // x
          m->arg[2] = NV_DATA_S(xz)+s.nx_; // z
          m->arg[3] = m->p; // p
          m->arg[4] = vx; // fwd:x
          m->arg[5] = vz; // fwd:z
          m->res[0] = m->v2; // fwd:ode
          m->res[1] = m->v2 + s.nx_; // fwd:alg
          s.calc_function(m, "jtimesF");

          // Subtract m->v2 (reordered) from m->v1
          v_it = m->v1 + s.nx1_ + s.nz1_;
          for (int d=1; d<=s.ns_; ++d) {
            casadi_axpy(s.nx1_, -1., m->v2 + d*s.nx1_, v_it);
            v_it += s.nx1_;
            casadi_axpy(s.nz1_, -1., m->v2 + s.nx_ + d*s.nz1_, v_it);
            v_it += s.nz1_;
          }
        }

        // Solve for sensitivity right-hand-sides
        if (s.linsolF_.solve(m->jac, m->v1 + s.nx1_ + s.nz1_, s.ns_, false, m->mem_linsolF)) {
          casadi_error("'jac' solve failed");
        }

        // Save to output, reordered
        v_it = m->v1 + s.nx1_ + s.nz1_;
        for (int d=1; d<=s.ns_; ++d) {
          casadi_copy(v_it, s.nx1_, vx + d*s.nx1_);
          v_it += s.nx1_;
          casadi_copy(v_it, s.nz1_, vz + d*s.nz1_);
          v_it += s.nz1_;
        }
      }

      return 0;
    } catch(int flag) { // recoverable error
      return flag;
    } catch(exception& e) { // non-recoverable error
      uerr() << "psolve failed: " << e.what() << endl;
      return -1;
    }
  }

  int IdasInterface::psolveB(double t, N_Vector xz, N_Vector xzdot, N_Vector xzB,
                                    N_Vector xzdotB, N_Vector resvalB, N_Vector rvecB,
                                    N_Vector zvecB, double cjB, double deltaB,
                                    void *user_data, N_Vector tmpB) {
    try {
      auto m = to_mem(user_data);
      auto& s = m->self;

      // Get right-hand sides in m->v1, ordered by sensitivity directions
      double* vx = NV_DATA_S(rvecB);
      double* vz = vx + s.nrx_;
      double* v_it = m->v1;
      for (int d=0; d<=s.ns_; ++d) {
        casadi_copy(vx + d*s.nrx1_, s.nrx1_, v_it);
        v_it += s.nrx1_;
        casadi_copy(vz + d*s.nrz1_, s.nrz1_, v_it);
        v_it += s.nrz1_;
      }

      // Solve for undifferentiated right-hand-side, save to output
      if (s.linsolB_.solve(m->jacB, m->v1, 1, false, m->mem_linsolB))
        casadi_error("'jacB' solve failed");
      vx = NV_DATA_S(zvecB); // possibly different from rvecB
      vz = vx + s.nrx_;
      casadi_copy(m->v1, s.nrx1_, vx);
      casadi_copy(m->v1 + s.nrx1_, s.nrz1_, vz);

      // Sensitivity equations
      if (s.ns_>0) {
        // Second order correction
        if (s.second_order_correction_) {
          // The outputs will double as seeds for jtimesB
          casadi_fill(vx + s.nrx1_, s.nrx_ - s.nrx1_, 0.);
          casadi_fill(vz + s.nrz1_, s.nrz_ - s.nrz1_, 0.);

          // Get second-order-correction, save to m->v2
          m->arg[0] = &t; // t
          m->arg[1] = NV_DATA_S(xz); // x
          m->arg[2] = NV_DATA_S(xz)+s.nx_; // z
          m->arg[3] = m->p; // p
          m->arg[4] = NV_DATA_S(xzB); // rx
          m->arg[5] = NV_DATA_S(xzB)+s.nrx_; // rz
          m->arg[6] = m->rp; // rp
          m->arg[7] = vx; // fwd:rx
          m->arg[8] = vz; // fwd:rz
          m->res[0] = m->v2; // fwd:rode
          m->res[1] = m->v2 + s.nrx_; // fwd:ralg
          s.calc_function(m, "jtimesB");

          // Subtract m->v2 (reordered) from m->v1
          v_it = m->v1 + s.nrx1_ + s.nrz1_;
          for (int d=1; d<=s.ns_; ++d) {
            casadi_axpy(s.nrx1_, -1., m->v2 + d*s.nrx1_, v_it);
            v_it += s.nrx1_;
            casadi_axpy(s.nrz1_, -1., m->v2 + s.nrx_ + d*s.nrz1_, v_it);
            v_it += s.nrz1_;
          }
        }

        // Solve for sensitivity right-hand-sides
        if (s.linsolB_.solve(m->jacB, m->v1 + s.nrx1_ + s.nrz1_, s.ns_, false, m->mem_linsolB)) {
          casadi_error("'jacB' solve failed");
        }

        // Save to output, reordered
        v_it = m->v1 + s.nrx1_ + s.nrz1_;
        for (int d=1; d<=s.ns_; ++d) {
          casadi_copy(v_it, s.nrx1_, vx + d*s.nrx1_);
          v_it += s.nrx1_;
          casadi_copy(v_it, s.nrz1_, vz + d*s.nrz1_);
          v_it += s.nrz1_;
        }
      }

      return 0;
    } catch(int flag) { // recoverable error
      return flag;
    } catch(exception& e) { // non-recoverable error
      uerr() << "psolveB failed: " << e.what() << endl;
      return -1;
    }
  }

  int IdasInterface::psetup(double t, N_Vector xz, N_Vector xzdot, N_Vector rr,
                                   double cj, void* user_data,
                                   N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    try {
      auto m = to_mem(user_data);
      auto& s = m->self;
      m->arg[0] = &t;
      m->arg[1] = NV_DATA_S(xz);
      m->arg[2] = NV_DATA_S(xz)+s.nx_;
      m->arg[3] = m->p;
      m->arg[4] = &cj;
      m->res[0] = m->jac;
      if (s.calc_function(m, "jacF")) casadi_error("Calculating Jacobian failed");

      // Factorize the linear system
      if (s.linsolF_.nfact(m->jac, m->mem_linsolF)) casadi_error("Linear solve failed");

      return 0;
    } catch(int flag) { // recoverable error
      return flag;
    } catch(exception& e) { // non-recoverable error
      uerr() << "psetup failed: " << e.what() << endl;
      return -1;
    }
  }

  int IdasInterface::psetupB(double t, N_Vector xz, N_Vector xzdot,
                                    N_Vector rxz, N_Vector rxzdot,
                                    N_Vector rresval, double cj, void *user_data,
                                    N_Vector tmp1B, N_Vector tmp2B, N_Vector tmp3B) {
    try {
      auto m = to_mem(user_data);
      auto& s = m->self;
      m->arg[0] = &t;
      m->arg[1] = NV_DATA_S(rxz);
      m->arg[2] = NV_DATA_S(rxz)+s.nrx_;
      m->arg[3] = m->rp;
      m->arg[4] = NV_DATA_S(xz);
      m->arg[5] = NV_DATA_S(xz)+s.nx_;
      m->arg[6] = m->p;
      m->arg[7] = &cj;
      m->res[0] = m->jacB;
      if (s.calc_function(m, "jacB")) casadi_error("'jacB' calculation failed");

      // Factorize the linear system
      if (s.linsolB_.nfact(m->jacB, m->mem_linsolB)) casadi_error("'jacB' factorization failed");

      return 0;
    } catch(int flag) { // recoverable error
      return flag;
    } catch(exception& e) { // non-recoverable error
      uerr() << "psetupB failed: " << e.what() << endl;
      return -1;
    }
  }

  int IdasInterface::lsetup(IDAMem IDA_mem, N_Vector xz, N_Vector xzdot, N_Vector resp,
                                    N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3) {
    // Current time
    double t = IDA_mem->ida_tn;

    // Multiple of df_dydot to be added to the matrix
    double cj = IDA_mem->ida_cj;

    // Call the preconditioner setup function (which sets up the linear solver)
    if (psetup(t, xz, xzdot, nullptr, cj, IDA_mem->ida_lmem,
      vtemp1, vtemp1, vtemp3)) return 1;

    return 0;
  }

  int IdasInterface::lsetupB(IDAMem IDA_mem, N_Vector xzB, N_Vector xzdotB, N_Vector respB,
                                     N_Vector vtemp1B, N_Vector vtemp2B, N_Vector vtemp3B) {
    try {
      auto m = to_mem(IDA_mem->ida_lmem);
      //auto& s = m->self;
      IDAadjMem IDAADJ_mem;
      //IDABMem IDAB_mem;

      // Current time
      double t = IDA_mem->ida_tn; // TODO(Joel): is this correct?
      // Multiple of df_dydot to be added to the matrix
      double cj = IDA_mem->ida_cj;

      IDA_mem = static_cast<IDAMem>(IDA_mem->ida_user_data);

      IDAADJ_mem = IDA_mem->ida_adj_mem;
      //IDAB_mem = IDAADJ_mem->ia_bckpbCrt;

      // Get FORWARD solution from interpolation.
      if (IDAADJ_mem->ia_noInterp==FALSE) {
        int flag = IDAADJ_mem->ia_getY(IDA_mem, t, IDAADJ_mem->ia_yyTmp, IDAADJ_mem->ia_ypTmp,
                                   nullptr, nullptr);
        if (flag != IDA_SUCCESS) casadi_error("Could not interpolate forward states");
      }
      // Call the preconditioner setup function (which sets up the linear solver)
      if (psetupB(t, IDAADJ_mem->ia_yyTmp, IDAADJ_mem->ia_ypTmp,
        xzB, xzdotB, nullptr, cj, static_cast<void*>(m), vtemp1B, vtemp1B, vtemp3B)) return 1;

      return 0;
    } catch(int flag) { // recoverable error
      return flag;
    } catch(exception& e) { // non-recoverable error
      uerr() << "lsetupB failed: " << e.what() << endl;
      return -1;
    }
  }

  int IdasInterface::lsolve(IDAMem IDA_mem, N_Vector b, N_Vector weight, N_Vector xz,
                                   N_Vector xzdot, N_Vector rr) {
    try {
      auto m = to_mem(IDA_mem->ida_lmem);
      auto& s = m->self;

      // Current time
      double t = IDA_mem->ida_tn;

      // Multiple of df_dydot to be added to the matrix
      double cj = IDA_mem->ida_cj;

      // Accuracy
      double delta = 0.0;

      // Call the preconditioner solve function (which solves the linear system)
      if (psolve(t, xz, xzdot, rr, b, b, cj,
        delta, static_cast<void*>(m), nullptr)) return 1;

      // Scale the correction to account for change in cj
      if (s.cj_scaling_) {
        double cjratio = IDA_mem->ida_cjratio;
        if (cjratio != 1.0) N_VScale(2.0/(1.0 + cjratio), b, b);
      }

      return 0;
    } catch(int flag) { // recoverable error
      return flag;
    } catch(exception& e) { // non-recoverable error
      uerr() << "lsolve failed: " << e.what() << endl;
      return -1;
    }
  }

  int IdasInterface::lsolveB(IDAMem IDA_mem, N_Vector b, N_Vector weight, N_Vector xzB,
                                    N_Vector xzdotB, N_Vector rrB) {
    try {
      auto m = to_mem(IDA_mem->ida_lmem);
      auto& s = m->self;
      IDAadjMem IDAADJ_mem;
      //IDABMem IDAB_mem;
      int flag;

      // Current time
      double t = IDA_mem->ida_tn; // TODO(Joel): is this correct?
      // Multiple of df_dydot to be added to the matrix
      double cj = IDA_mem->ida_cj;
      double cjratio = IDA_mem->ida_cjratio;

      IDA_mem = (IDAMem) IDA_mem->ida_user_data;

      IDAADJ_mem = IDA_mem->ida_adj_mem;
      //IDAB_mem = IDAADJ_mem->ia_bckpbCrt;

      // Get FORWARD solution from interpolation.
      if (IDAADJ_mem->ia_noInterp==FALSE) {
        flag = IDAADJ_mem->ia_getY(IDA_mem, t, IDAADJ_mem->ia_yyTmp, IDAADJ_mem->ia_ypTmp,
                                   nullptr, nullptr);
        if (flag != IDA_SUCCESS) casadi_error("Could not interpolate forward states");
      }

      // Accuracy
      double delta = 0.0;

      // Call the preconditioner solve function (which solves the linear system)
      if (psolveB(t, IDAADJ_mem->ia_yyTmp, IDAADJ_mem->ia_ypTmp, xzB, xzdotB,
        rrB, b, b, cj, delta, static_cast<void*>(m), nullptr)) return 1;

      // Scale the correction to account for change in cj
      if (s.cj_scaling_) {
        if (cjratio != 1.0) N_VScale(2.0/(1.0 + cjratio), b, b);
      }
      return 0;
    } catch(int flag) { // recoverable error
      return flag;
    } catch(exception& e) { // non-recoverable error
      uerr() << "lsolveB failed: " << e.what() << endl;
      return -1;
    }
  }

  Function IdasInterface::getJ(bool backward) const {
    return oracle_.is_a("SXFunction") ? getJ<SX>(backward) : getJ<MX>(backward);
  }

  template<typename MatType>
  Function IdasInterface::getJ(bool backward) const {
    vector<MatType> a = MatType::get_input(oracle_);
    vector<MatType> r = const_cast<Function&>(oracle_)(a);
    MatType cj = MatType::sym("cj");

    // Get the Jacobian in the Newton iteration
    if (backward) {
      MatType jac = MatType::jacobian(r[DE_RODE], a[DE_RX]) + cj*MatType::eye(nrx_);
      if (nrz_>0) {
        jac = horzcat(vertcat(jac,
                              MatType::jacobian(r[DE_RALG], a[DE_RX])),
                      vertcat(MatType::jacobian(r[DE_RODE], a[DE_RZ]),
                              MatType::jacobian(r[DE_RALG], a[DE_RZ])));
      }
      return Function("jacB", {a[DE_T], a[DE_RX], a[DE_RZ], a[DE_RP],
                               a[DE_X], a[DE_Z], a[DE_P], cj}, {jac});
     } else {
      MatType jac = MatType::jacobian(r[DE_ODE], a[DE_X]) - cj*MatType::eye(nx_);
      if (nz_>0) {
        jac = horzcat(vertcat(jac,
                              MatType::jacobian(r[DE_ALG], a[DE_X])),
                      vertcat(MatType::jacobian(r[DE_ODE], a[DE_Z]),
                              MatType::jacobian(r[DE_ALG], a[DE_Z])));
      }
      return Function("jacF", {a[DE_T], a[DE_X], a[DE_Z], a[DE_P], cj}, {jac});
    }
  }

  IdasMemory::IdasMemory(const IdasInterface& s) : self(s) {
    this->mem = nullptr;
    this->xzdot = nullptr;
    this->rxzdot = nullptr;

    // Reset checkpoints counter
    this->ncheck = 0;
  }

  IdasMemory::~IdasMemory() {
    if (this->mem) IDAFree(&this->mem);
    if (this->xzdot) N_VDestroy_Serial(this->xzdot);
    if (this->rxzdot) N_VDestroy_Serial(this->rxzdot);
  }

} // namespace casadi
