#if !defined(__PERMONSVM_H)
#define __PERMONSVM_H

#include <permonqps.h>

/*S
  SVM - PERMON class for Support Vector Machines (SVM) classification on top of PermonQP solvers (QPS)

  Level: beginner

  Concepts: SVM classification

.seealso:  SVMCreate(), SVMDestroy(), QP, QPS
 S*/
typedef struct _p_SVM* SVM;

FLLOP_EXTERN PetscClassId SVM_CLASSID;
#define SVM_CLASS_NAME  "svm"

#define SVMType       char*
#define SVM_BINARY    "binary"

/*E
  SVMLossType - Determines the loss function for soft-margin SVM (non-separable samples).

  Level: beginner

.seealso: SVMSetLossType, SVMGetLossType()
E*/
typedef enum {SVM_L1, SVM_L2} SVMLossType;
FLLOP_EXTERN const char *const SVMLossTypes[];

/*MC
     SVM_L1 - L1 type of loss function, \xi = max(0, 1 - y_i * w' * x_i)

   Level: beginner

.seealso: SVMLossType, SVM_L2, SVMGetLossType(), SVMSetLossType()
M*/

/*MC
     SVM_L2 - L2 type of loss function, \xi = max(0, 1 - y_i * w' * x_i)^2

   Level: beginner

.seealso: SVMLossType, SVM_L1, SVMGetLossType(), SVMSetLossType()
M*/

FLLOP_EXTERN PetscErrorCode SVMInitializePackage();
FLLOP_EXTERN PetscErrorCode SVMFinalizePackage();

FLLOP_EXTERN PetscFunctionList SVMList;
FLLOP_EXTERN PetscBool SVMRegisterAllCalled;
FLLOP_EXTERN PetscErrorCode SVMRegisterAll();
FLLOP_EXTERN PetscErrorCode SVMRegister(const char [],PetscErrorCode (*function)(SVM));

FLLOP_EXTERN PetscErrorCode SVMCreate(MPI_Comm comm, SVM *svm_out);
FLLOP_EXTERN PetscErrorCode SVMReset(SVM svm);
FLLOP_EXTERN PetscErrorCode SVMDestroy(SVM *svm);
FLLOP_EXTERN PetscErrorCode SVMView(SVM svm, PetscViewer v);
FLLOP_EXTERN PetscErrorCode SVMSetC(SVM svm, PetscReal _C);
FLLOP_EXTERN PetscErrorCode SVMGetC(SVM svm, PetscReal *C);
FLLOP_EXTERN PetscErrorCode SVMSetLogCMin(SVM svm, PetscReal LogCMin);
FLLOP_EXTERN PetscErrorCode SVMGetLogCMin(SVM svm, PetscReal *LogCMin);
FLLOP_EXTERN PetscErrorCode SVMSetLogCMax(SVM svm, PetscReal LogCMax);
FLLOP_EXTERN PetscErrorCode SVMGetLogCMax(SVM svm, PetscReal *LogCMax);
FLLOP_EXTERN PetscErrorCode SVMSetLogCBase(SVM svm, PetscReal LogCBase);
FLLOP_EXTERN PetscErrorCode SVMGetLogCBase(SVM svm, PetscReal *LogCBase);
FLLOP_EXTERN PetscErrorCode SVMSetTrainingSamples(SVM svm, Mat Xt, Vec y);
FLLOP_EXTERN PetscErrorCode SVMGetTrainingSamples(SVM svm, Mat *Xt, Vec *y);
FLLOP_EXTERN PetscErrorCode SVMSetLossType(SVM svm, SVMLossType type);
FLLOP_EXTERN PetscErrorCode SVMGetLossType(SVM svm, SVMLossType *type);
FLLOP_EXTERN PetscErrorCode SVMSetNfolds(SVM svm, PetscInt nfolds);
FLLOP_EXTERN PetscErrorCode SVMGetNfolds(SVM svm, PetscInt *nfolds);
FLLOP_EXTERN PetscErrorCode SVMSetWarmStart(SVM svm, PetscBool flg);
FLLOP_EXTERN PetscErrorCode SVMSetQPS(SVM svm, QPS qps);
FLLOP_EXTERN PetscErrorCode SVMGetQPS(SVM svm, QPS *qps);
FLLOP_EXTERN PetscErrorCode SVMSetOptionsPrefix(SVM svm,const char prefix[]);
FLLOP_EXTERN PetscErrorCode SVMAppendOptionsPrefix(SVM svm,const char prefix[]);
FLLOP_EXTERN PetscErrorCode SVMGetOptionsPrefix(SVM svm,const char *prefix[]);
FLLOP_EXTERN PetscErrorCode SVMSetUp(SVM svm);
FLLOP_EXTERN PetscErrorCode SVMSetAutoPostTrain(SVM svm, PetscBool flg);
FLLOP_EXTERN PetscErrorCode SVMSetFromOptions(SVM svm);
FLLOP_EXTERN PetscErrorCode SVMTrain(SVM svm);
FLLOP_EXTERN PetscErrorCode SVMPostTrain(SVM svm);
FLLOP_EXTERN PetscErrorCode SVMGetSeparatingHyperplane(SVM svm, Vec *w, PetscReal *b);
FLLOP_EXTERN PetscErrorCode SVMClassify(SVM svm, Mat Xt_test, Vec *y_out);
FLLOP_EXTERN PetscErrorCode SVMTest(SVM svm, Mat Xt_test, Vec y_known, PetscInt *N_all, PetscInt *N_eq);
FLLOP_EXTERN PetscErrorCode SVMCrossValidate(SVM svm);

#endif

