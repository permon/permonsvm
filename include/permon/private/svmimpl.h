
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
  PetscErrorCode (*reconstructhyperplane)(SVM);
  PetscErrorCode (*predict)(SVM,Mat,Vec *);
  PetscErrorCode (*test)(SVM);
  PetscErrorCode (*gridsearch)(SVM);
  PetscErrorCode (*crossvalidation)(SVM,PetscReal [],PetscInt,PetscReal []);
  PetscErrorCode (*view)(SVM,PetscViewer);
  PetscErrorCode (*viewscore)(SVM,PetscViewer);
  PetscErrorCode (*computemodelscores)(SVM,Vec,Vec);
  PetscErrorCode (*computehingeloss)(SVM);
  PetscErrorCode (*computemodelparams)(SVM);
};

struct _p_SVM {
  PETSCHEADER(struct _SVMOps);

  Mat                 Xt_test;
  Vec                 y_test;

  ModelScore          cv_model_score;
  CrossValidationType cv_type;
  PetscReal           C,C_old;
  PetscReal           Cp,Cp_old;
  PetscReal           Cn,Cn_old;
  PetscReal           LogCMin,LogCMax,LogCBase;
  PetscInt            nfolds;

  SVMLossType         loss_type;

  PetscBool           warm_start;

  PetscBool           autoposttrain;
  PetscBool           posttraincalled;
  PetscBool           setupcalled;
  PetscBool           setfromoptionscalled;

  void *data;
};
#endif

