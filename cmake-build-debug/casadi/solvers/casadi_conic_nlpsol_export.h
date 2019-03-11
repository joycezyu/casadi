
#ifndef CASADI_CONIC_NLPSOL_EXPORT_H
#define CASADI_CONIC_NLPSOL_EXPORT_H

#ifdef CASADI_CONIC_NLPSOL_STATIC_DEFINE
#  define CASADI_CONIC_NLPSOL_EXPORT
#  define CASADI_CONIC_NLPSOL_NO_EXPORT
#else
#  ifndef CASADI_CONIC_NLPSOL_EXPORT
#    ifdef casadi_conic_nlpsol_EXPORTS
        /* We are building this library */
#      define CASADI_CONIC_NLPSOL_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_CONIC_NLPSOL_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_CONIC_NLPSOL_NO_EXPORT
#    define CASADI_CONIC_NLPSOL_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_CONIC_NLPSOL_DEPRECATED
#  define CASADI_CONIC_NLPSOL_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef CASADI_CONIC_NLPSOL_DEPRECATED_EXPORT
#  define CASADI_CONIC_NLPSOL_DEPRECATED_EXPORT CASADI_CONIC_NLPSOL_EXPORT CASADI_CONIC_NLPSOL_DEPRECATED
#endif

#ifndef CASADI_CONIC_NLPSOL_DEPRECATED_NO_EXPORT
#  define CASADI_CONIC_NLPSOL_DEPRECATED_NO_EXPORT CASADI_CONIC_NLPSOL_NO_EXPORT CASADI_CONIC_NLPSOL_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef CASADI_CONIC_NLPSOL_NO_DEPRECATED
#    define CASADI_CONIC_NLPSOL_NO_DEPRECATED
#  endif
#endif

#endif
