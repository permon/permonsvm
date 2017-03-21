#if !defined(__SVMIMPL_H)
#define	__SVMIMPL_H

#include <private/qpsimpl.h>
#include <permonsvm.h>

struct _p_PermonSVM {
    PETSCHEADER(int);
    PetscBool autoPostSolve;
    PetscBool setupcalled;
    PetscBool setfromoptionscalled;
    
    PetscReal C, C_min, C_max, C_step;
    PetscInt nfolds;
    Mat Xt;
    Vec y;
    
    Vec w;
    PetscScalar b;
    
    QPS qps;
};

#endif 

