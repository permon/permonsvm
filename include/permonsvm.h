#if !defined(__PERMONSVM_H)
#define __PERMONSVM_H

typedef struct _p_PermonSVM* PermonSVM;

FLLOP_EXTERN PetscClassId SVM_CLASSID;
#define SVM_CLASS_NAME  "svm"

FLLOP_EXTERN PetscErrorCode PermonSVMInitializePackage();
FLLOP_EXTERN PetscErrorCode PermonSVMCreate(MPI_Comm comm, PermonSVM *svm_out);
FLLOP_EXTERN PetscErrorCode PermonSVMReset(PermonSVM svm);
FLLOP_EXTERN PetscErrorCode PermonSVMDestroy(PermonSVM *svm);
FLLOP_EXTERN PetscErrorCode PermonSVMView(PermonSVM svm, PetscViewer v);
FLLOP_EXTERN PetscErrorCode PermonSVMSetPenalty(PermonSVM svm, PetscReal _C);
FLLOP_EXTERN PetscErrorCode PermonSVMGetPenalty(PermonSVM svm, PetscReal *C);
FLLOP_EXTERN PetscErrorCode PermonSVMSetPenaltyMin(PermonSVM svm, PetscReal C_min);
FLLOP_EXTERN PetscErrorCode PermonSVMGetPenaltyMin(PermonSVM svm, PetscReal *C_min);
FLLOP_EXTERN PetscErrorCode PermonSVMSetPenaltyMax(PermonSVM svm, PetscReal C_max);
FLLOP_EXTERN PetscErrorCode PermonSVMGetPenaltyMax(PermonSVM svm, PetscReal *C_max);
FLLOP_EXTERN PetscErrorCode PermonSVMSetPenaltyStep(PermonSVM svm, PetscReal C_step);
FLLOP_EXTERN PetscErrorCode PermonSVMGetPenaltyStep(PermonSVM svm, PetscReal *C_step);
FLLOP_EXTERN PetscErrorCode PermonSVMSetTrainingSamples(PermonSVM svm, Mat Xt, Vec y);
FLLOP_EXTERN PetscErrorCode PermonSVMGetTrainingSamples(PermonSVM svm, Mat *Xt, Vec *y);
FLLOP_EXTERN PetscErrorCode PermonSVMSetNfolds(PermonSVM svm, PetscInt nfolds);
FLLOP_EXTERN PetscErrorCode PermonSVMGetNfolds(PermonSVM svm, PetscInt *nfolds);
FLLOP_EXTERN PetscErrorCode PermonSVMSetQPS(PermonSVM svm, QPS qps);
FLLOP_EXTERN PetscErrorCode PermonSVMGetQPS(PermonSVM svm, QPS *qps);
FLLOP_EXTERN PetscErrorCode PermonSVMSetUp(PermonSVM svm);
FLLOP_EXTERN PetscErrorCode PermonSVMSetAutoPostTrain(PermonSVM svm, PetscBool flg); 
FLLOP_EXTERN PetscErrorCode PermonSVMSetFromOptions(PermonSVM svm);
FLLOP_EXTERN PetscErrorCode PermonSVMTrain(PermonSVM svm);
FLLOP_EXTERN PetscErrorCode PermonSVMPostTrain(PermonSVM svm);
FLLOP_EXTERN PetscErrorCode PermonSVMGetSepparatingHyperplane(PermonSVM svm, Vec *w, PetscReal *b);
FLLOP_EXTERN PetscErrorCode PermonSVMClassify(PermonSVM svm, Mat Xt_test, Vec *y_out);
FLLOP_EXTERN PetscErrorCode PermonSVMTest(PermonSVM svm, Mat Xt_test, Vec y_known, PetscInt *N);
FLLOP_EXTERN PetscErrorCode PermonSVMCrossValidate(PermonSVM svm);

#endif

