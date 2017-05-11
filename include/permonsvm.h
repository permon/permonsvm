#if !defined(__PERMONSVM_H)
#define __PERMONSVM_H

#include <permonqps.h>

/*S
  PermonSVM - PERMON class for Support Vector Machines (SVM) classification on top of PermonQP solvers (QPS)

  Level: beginner

  Concepts: SVM classification

.seealso:  PermonSVMCreate(), PermonSVMDestroy(), QP, QPS
 S*/
typedef struct _p_PermonSVM* PermonSVM;

FLLOP_EXTERN PetscClassId SVM_CLASSID;
#define SVM_CLASS_NAME  "svm"

/*E
  PermonSVMLossType - Determines the loss function for soft-margin SVM (non-separable samples).

  Level: beginner

.seealso: PermonSVMSetLossType, PermonSVMGetLossType()
E*/
typedef enum {PERMON_SVM_L1, PERMON_SVM_L2} PermonSVMLossType;
FLLOP_EXTERN const char *const PermonSVMLossTypes[];

/*MC
     PERMON_SVM_L1 - L1 type of loss function, \xi = max(0, 1 - y_i * w' * x_i)

   Level: beginner

.seealso: PermonSVMLossType, PERMON_SVM_L2, PermonSVMGetLossType(), PermonSVMSetLossType()
M*/

/*MC
     PERMON_SVM_L2 - L2 type of loss function, \xi = max(0, 1 - y_i * w' * x_i)^2

   Level: beginner

.seealso: PermonSVMLossType, PERMON_SVM_L1, PermonSVMGetLossType(), PermonSVMSetLossType()
M*/

FLLOP_EXTERN PetscErrorCode PermonSVMInitializePackage();
FLLOP_EXTERN PetscErrorCode PermonSVMCreate(MPI_Comm comm, PermonSVM *svm_out);
FLLOP_EXTERN PetscErrorCode PermonSVMReset(PermonSVM svm);
FLLOP_EXTERN PetscErrorCode PermonSVMDestroy(PermonSVM *svm);
FLLOP_EXTERN PetscErrorCode PermonSVMView(PermonSVM svm, PetscViewer v);
FLLOP_EXTERN PetscErrorCode PermonSVMSetC(PermonSVM svm, PetscReal _C);
FLLOP_EXTERN PetscErrorCode PermonSVMGetC(PermonSVM svm, PetscReal *C);
FLLOP_EXTERN PetscErrorCode PermonSVMSetLogCMin(PermonSVM svm, PetscReal LogCMin);
FLLOP_EXTERN PetscErrorCode PermonSVMGetLogCMin(PermonSVM svm, PetscReal *LogCMin);
FLLOP_EXTERN PetscErrorCode PermonSVMSetLogCMax(PermonSVM svm, PetscReal LogCMax);
FLLOP_EXTERN PetscErrorCode PermonSVMGetLogCMax(PermonSVM svm, PetscReal *LogCMax);
FLLOP_EXTERN PetscErrorCode PermonSVMSetLogCBase(PermonSVM svm, PetscReal LogCBase);
FLLOP_EXTERN PetscErrorCode PermonSVMGetLogCBase(PermonSVM svm, PetscReal *LogCBase);
FLLOP_EXTERN PetscErrorCode PermonSVMSetTrainingSamples(PermonSVM svm, Mat Xt, Vec y);
FLLOP_EXTERN PetscErrorCode PermonSVMGetTrainingSamples(PermonSVM svm, Mat *Xt, Vec *y);
FLLOP_EXTERN PetscErrorCode PermonSVMSetLossType(PermonSVM svm, PermonSVMLossType type);
FLLOP_EXTERN PetscErrorCode PermonSVMGetLossType(PermonSVM svm, PermonSVMLossType *type);
FLLOP_EXTERN PetscErrorCode PermonSVMSetNfolds(PermonSVM svm, PetscInt nfolds);
FLLOP_EXTERN PetscErrorCode PermonSVMGetNfolds(PermonSVM svm, PetscInt *nfolds);
FLLOP_EXTERN PetscErrorCode PermonSVMSetQPS(PermonSVM svm, QPS qps);
FLLOP_EXTERN PetscErrorCode PermonSVMGetQPS(PermonSVM svm, QPS *qps);
FLLOP_EXTERN PetscErrorCode PermonSVMSetOptionsPrefix(PermonSVM svm,const char prefix[]);
FLLOP_EXTERN PetscErrorCode PermonSVMAppendOptionsPrefix(PermonSVM svm,const char prefix[]);
FLLOP_EXTERN PetscErrorCode PermonSVMGetOptionsPrefix(PermonSVM svm,const char *prefix[]);
FLLOP_EXTERN PetscErrorCode PermonSVMSetUp(PermonSVM svm);
FLLOP_EXTERN PetscErrorCode PermonSVMSetAutoPostTrain(PermonSVM svm, PetscBool flg); 
FLLOP_EXTERN PetscErrorCode PermonSVMSetFromOptions(PermonSVM svm);
FLLOP_EXTERN PetscErrorCode PermonSVMTrain(PermonSVM svm);
FLLOP_EXTERN PetscErrorCode PermonSVMPostTrain(PermonSVM svm);
FLLOP_EXTERN PetscErrorCode PermonSVMGetSeparatingHyperplane(PermonSVM svm, Vec *w, PetscReal *b);
FLLOP_EXTERN PetscErrorCode PermonSVMClassify(PermonSVM svm, Mat Xt_test, Vec *y_out);
FLLOP_EXTERN PetscErrorCode PermonSVMTest(PermonSVM svm, Mat Xt_test, Vec y_known, PetscInt *N_all, PetscInt *N_eq);
FLLOP_EXTERN PetscErrorCode PermonSVMCrossValidate(PermonSVM svm);

#endif

