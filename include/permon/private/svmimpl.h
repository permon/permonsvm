
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
  PetscErrorCode (*test)(SVM,Mat,Vec,PetscInt *,PetscInt *);
  PetscErrorCode (*gridsearch)(SVM);
  PetscErrorCode (*crossvalidation)(SVM,PetscReal [],PetscInt,PetscReal []);
  PetscErrorCode (*view)(SVM,PetscViewer);
};

struct _p_SVM {
  PETSCHEADER(struct _SVMOps);

  PetscReal C,C_old;
  PetscReal LogCMin,LogCMax,LogCBase;
  PetscInt  nfolds;

  SVMLossType loss_type;

  PetscBool   warm_start;

  PetscBool autoposttrain;
  PetscBool posttraincalled;
  PetscBool setupcalled;
  PetscBool setfromoptionscalled;

  void *data;
};
#endif

