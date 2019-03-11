
#ifndef CASADI_LINSOL_LSQR_EXPORT_H
#define CASADI_LINSOL_LSQR_EXPORT_H

#ifdef CASADI_LINSOL_LSQR_STATIC_DEFINE
#  define CASADI_LINSOL_LSQR_EXPORT
#  define CASADI_LINSOL_LSQR_NO_EXPORT
#else
#  ifndef CASADI_LINSOL_LSQR_EXPORT
#    ifdef casadi_linsol_lsqr_EXPORTS
        /* We are building this library */
#      define CASADI_LINSOL_LSQR_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_LINSOL_LSQR_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_LINSOL_LSQR_NO_EXPORT
#    define CASADI_LINSOL_LSQR_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_LINSOL_LSQR_DEPRECATED
#  define CASADI_LINSOL_LSQR_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef CASADI_LINSOL_LSQR_DEPRECATED_EXPORT
#  define CASADI_LINSOL_LSQR_DEPRECATED_EXPORT CASADI_LINSOL_LSQR_EXPORT CASADI_LINSOL_LSQR_DEPRECATED
#endif

#ifndef CASADI_LINSOL_LSQR_DEPRECATED_NO_EXPORT
#  define CASADI_LINSOL_LSQR_DEPRECATED_NO_EXPORT CASADI_LINSOL_LSQR_NO_EXPORT CASADI_LINSOL_LSQR_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef CASADI_LINSOL_LSQR_NO_DEPRECATED
#    define CASADI_LINSOL_LSQR_NO_DEPRECATED
#  endif
#endif

#endif
