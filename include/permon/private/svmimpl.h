#if !defined(__SVMIMPL_H)
#define	__SVMIMPL_H

#include <permon/private/qpsimpl.h>
#include <permonsvm.h>

typedef struct _SVMOps *SVMOps;

struct _SVMOps {
  PetscErrorCode (*reset)(SVM);
  PetscErrorCode (*destroy)(SVM);
  PetscErrorCode (*setfromoptions)(PetscOptionItems *,SVM);
  PetscErrorCode (*setup)(SVM);
  PetscErrorCode (*train)(SVM);
  PetscErrorCode (*posttrain)(SVM);
  PetscErrorCode (*predict)(SVM,Mat,Vec *);
  PetscErrorCode (*view)(SVM,PetscViewer);
};

struct _p_SVM {
  PETSCHEADER(struct _SVMOps);

  PetscBool autoposttrain;
  PetscBool posttraincalled;
  PetscBool setupcalled;
  PetscBool setfromoptionscalled;
    
  PetscReal C, LogCMin, LogCMax, LogCBase;
  PetscInt nfolds;
  SVMLossType loss_type;

  PetscBool warm_start;

  Mat Xt;
  Vec y;
  Vec y_inner;
  PetscScalar y_map[2];
  Mat D;
    
  Vec w;
  PetscScalar b;
    
  QPS qps;

  void *data;
};

#endif 

