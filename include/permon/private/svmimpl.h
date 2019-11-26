
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
  PetscErrorCode (*loadtrainingdataset)(SVM,PetscViewer);
  PetscErrorCode (*viewtrainingdataset)(SVM,PetscViewer);
};

struct _p_SVM {
  PETSCHEADER(struct _SVMOps);

  char training_dataset_file[PETSC_MAX_PATH_LEN];
  char test_dataset_file[PETSC_MAX_PATH_LEN];

  Mat                 Xt_test;
  Vec                 y_test;

  ModelScore          hopt_score_types[7];
  PetscInt            hopt_nscore_types;
  CrossValidationType cv_type;
  PetscInt            penalty_type;
  PetscReal           C,C_old;
  PetscReal           Cp,Cp_old;
  PetscReal           Cn,Cn_old;
  PetscReal           logC_base,logC_start,logC_end,logC_step;
  PetscReal           logCp_base,logCp_start,logCp_end,logCp_step;
  PetscReal           logCn_base,logCn_start,logCn_end;
  PetscInt            nfolds;

  SVMLossType         loss_type;
  PetscInt            svm_mod;

  PetscBool           warm_start;

  PetscBool           hyperoptset;
  PetscBool           autoposttrain;
  PetscBool           posttraincalled;
  PetscBool           setupcalled;
  PetscBool           setfromoptionscalled;

  void *data;
};

FLLOP_EXTERN PetscLogEvent SVM_LoadDataset;
#endif

