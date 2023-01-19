
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
  PetscErrorCode (*convergedsetup)(SVM);
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
  PetscErrorCode (*loadgramian)(SVM,PetscViewer);
  PetscErrorCode (*viewgramian)(SVM,PetscViewer);
  PetscErrorCode (*viewtrainingpredictions)(SVM,PetscViewer);
  PetscErrorCode (*viewtestpredictions)(SVM,PetscViewer);
};

struct _p_SVM {
  PETSCHEADER(struct _SVMOps);

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
  PetscReal           logCn_base,logCn_start,logCn_end,logCn_step;
  PetscInt            nfolds;

  SVMLossType         loss_type;
  PetscInt            svm_mod;
  PetscReal           user_bias;

  PetscBool           warm_start;

  PetscBool           hyperoptset;
  PetscBool           autoposttrain;
  PetscBool           posttraincalled;
  PetscBool           setupcalled;
  PetscBool           setfromoptionscalled;

  void *data;
};

FLLOP_EXTERN PetscLogEvent SVM_LoadDataset;
FLLOP_EXTERN PetscLogEvent SVM_LoadGramian;

#endif

