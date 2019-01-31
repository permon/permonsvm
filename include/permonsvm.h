
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

/*E
  ModelScore - type of model performance score.

  Level: beginner

.seealso SVMGetModelScore()
E*/
typedef enum {MODEL_ACCURACY, MODEL_PRECISION, MODEL_SENSITIVITY, MODEL_F1, MODEL_MCC, MODEL_AUC_ROC, MODEL_G1} ModelScore;
FLLOP_EXTERN const char *const ModelScores[];

/*MC
  MODEL_ACCURACY - a ratio of correctly predicted samples to the total samples.

  Level: beginner

.seealso ModelScore, MODEL_PRECISION, MODEL_SENSTIVITY, MODEL_F1, MODEL_MCC, MODEL_AUC_ROC, MODEL_G1 SVMGetModelScore()
M*/

/*MC
  MODEL_PRECISION - a ratio of correctly predicted positive samples to the total predicted positive samples.

  Level: beginner

.seealso ModelScore, MODEL_ACCURACY, MODEL_SENSTIVITY, MODEL_F1, MODEL_MCC, MODEL_AUC_ROC, MODEL_G1 SVMGetModelScore()
M*/

/*MC
  MODEL_SENSTIVITY - a ratio of correctly predicted positive samples to the all samples in actual class.

  Level: beginner

.seealso ModelScore, MODEL_ACCURACY, MODEL_PRECISION, MODEL_F1, MODEL_MCC, MODEL_AUC_ROC, MODEL_G1 SVMGetModelScore()
M*/

/*MC
  MODEL_F1 - a harmonic average of precision and sensitivity.

  Level: beginner

.seealso ModelScore, MODEL_ACCURACY, MODEL_PRECISION, MODEL_SENSTIVITY, MODEL_MCC, MODEL_AUC_ROC, MODEL_G1 SVMGetModelScore()
M*/

/*MC
  MODEL_MCC - Matthews correlation coefficient.

  Level: beginner

.seealso ModelScore, MODEL_ACCURACY, MODEL_PRECISION, MODEL_SENSTIVITY, MODEL_F1, MODEL_AUC_ROC, MODEL_G1 SVMGetModelScore()
M*/

/*MC
  MODEL_AUC_ROC - Area Under Curve (AUC) Receiver Operating Characteristics (ROC)

  Level: beginner

.seealso ModelScore, MODEL_ACCURACY, MODEL_PRECISION, MODEL_SENSTIVITY, MODEL_F1, MODEL_MCC, MODEL_G1 SVMGetModelScore()
M*/

/*MC
  MODEL_G1 - Gini coefficient

  Level: beginner

.seealso ModelScore, MODEL_ACCURACY, MODEL_PRECISION, MODEL_SENSTIVITY, MODEL_F1, MODEL_MCC, MODEL_AUC_ROC SVMGetModelScore()
M*/

/*E
  CrossValidationType - type of cross validation.

  Level: beginner

.seealso: CROSS_VALIDATION_KFOLD, CROSS_VALIDATION_STRATIFIED_KFOLD
E*/
typedef enum {CROSS_VALIDATION_KFOLD,CROSS_VALIDATION_STRATIFIED_KFOLD} CrossValidationType;
FLLOP_EXTERN const char *const CrossValidationTypes[];

/*MC
  CROSS_VALIDATION_KFOLD - k-fold cross validation.

  Level: intermediate

.seealso CrossValidationType, CROSS_VALIDATION_STRATIFIED_KFOLD
M*/

/*MC
  CROSS_VALIDATION_STRATIFIED_KFOLD - stratified k-fold cross validation.

  Level: intermediate

.seealso CrossValidationType, CROSS_VALIDATION_KFOLD
M*/

FLLOP_EXTERN PetscErrorCode SVMInitializePackage();
FLLOP_EXTERN PetscErrorCode SVMFinalizePackage();

FLLOP_EXTERN PetscFunctionList SVMList;
FLLOP_EXTERN PetscBool SVMRegisterAllCalled;
FLLOP_EXTERN PetscErrorCode SVMRegisterAll();
FLLOP_EXTERN PetscErrorCode SVMRegister(const char [],PetscErrorCode (*function)(SVM));

FLLOP_EXTERN PetscErrorCode SVMCreate(MPI_Comm,SVM *);
FLLOP_EXTERN PetscErrorCode SVMSetType(SVM,const SVMType);
FLLOP_EXTERN PetscErrorCode SVMReset(SVM);
FLLOP_EXTERN PetscErrorCode SVMDestroy(SVM *);
FLLOP_EXTERN PetscErrorCode SVMDestroyDefault(SVM);
FLLOP_EXTERN PetscErrorCode SVMSetFromOptions(SVM);
FLLOP_EXTERN PetscErrorCode SVMSetUp(SVM);
FLLOP_EXTERN PetscErrorCode SVMView(SVM,PetscViewer);
FLLOP_EXTERN PetscErrorCode SVMViewScore(SVM,PetscViewer);

FLLOP_EXTERN PetscErrorCode SVMSetTrainingDataset(SVM,Mat,Vec);
FLLOP_EXTERN PetscErrorCode SVMGetTrainingDataset(SVM,Mat *,Vec *);
FLLOP_EXTERN PetscErrorCode SVMSetTestDataset(SVM,Mat,Vec);
FLLOP_EXTERN PetscErrorCode SVMGetTestDataset(SVM,Mat *,Vec *);

FLLOP_EXTERN PetscErrorCode SVMSetMod(SVM,PetscInt);
FLLOP_EXTERN PetscErrorCode SVMGetMod(SVM,PetscInt *);
FLLOP_EXTERN PetscErrorCode SVMSetLossType(SVM,SVMLossType);
FLLOP_EXTERN PetscErrorCode SVMGetLossType(SVM,SVMLossType *);
FLLOP_EXTERN PetscErrorCode SVMSetPenaltyType(SVM,PetscInt);
FLLOP_EXTERN PetscErrorCode SVMGetPenaltyType(SVM,PetscInt *);
FLLOP_EXTERN PetscErrorCode SVMSetPenalty(SVM,PetscInt,PetscReal []);
FLLOP_EXTERN PetscErrorCode SVMSetC(SVM,PetscReal);
FLLOP_EXTERN PetscErrorCode SVMGetC(SVM,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMSetCp(SVM,PetscReal);
FLLOP_EXTERN PetscErrorCode SVMGetCp(SVM,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMSetCn(SVM,PetscReal);
FLLOP_EXTERN PetscErrorCode SVMGetCn(SVM,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMSetNfolds(SVM,PetscInt);
FLLOP_EXTERN PetscErrorCode SVMGetNfolds(SVM,PetscInt *);
FLLOP_EXTERN PetscErrorCode SVMSetSeparatingHyperplane(SVM,Vec,PetscReal);
FLLOP_EXTERN PetscErrorCode SVMGetSeparatingHyperplane(SVM,Vec *,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMSetBias(SVM,PetscReal);
FLLOP_EXTERN PetscErrorCode SVMGetBias(SVM,PetscReal *);

FLLOP_EXTERN PetscErrorCode SVMTrain(SVM);
FLLOP_EXTERN PetscErrorCode SVMPostTrain(SVM);
FLLOP_EXTERN PetscErrorCode SVMReconstructHyperplane(SVM);
FLLOP_EXTERN PetscErrorCode SVMSetAutoPostTrain(SVM,PetscBool);
FLLOP_EXTERN PetscErrorCode SVMPredict(SVM,Mat,Vec *);
FLLOP_EXTERN PetscErrorCode SVMTest(SVM);

FLLOP_EXTERN PetscErrorCode SVMComputeModelScores(SVM,Vec,Vec);
FLLOP_EXTERN PetscErrorCode SVMGetModelScore(SVM,ModelScore,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMComputeHingeLoss(SVM svm);
FLLOP_EXTERN PetscErrorCode SVMComputeModelParams(SVM svm);

FLLOP_EXTERN PetscErrorCode SVMSetHyperOpt(SVM,PetscBool);
FLLOP_EXTERN PetscErrorCode SVMSetHyperOptScoreTypes(SVM,PetscInt,ModelScore []);
FLLOP_EXTERN PetscErrorCode SVMGetHyperOptNScoreTypes(SVM,PetscInt *);
FLLOP_EXTERN PetscErrorCode SVMGetCrossValidationScoreType(SVM,ModelScore *);

FLLOP_EXTERN PetscErrorCode SVMGridSearch(SVM);
FLLOP_EXTERN PetscErrorCode SVMSetLogCMin(SVM,PetscReal);
FLLOP_EXTERN PetscErrorCode SVMGetLogCMin(SVM,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMSetLogCMax(SVM,PetscReal);
FLLOP_EXTERN PetscErrorCode SVMGetLogCMax(SVM,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMSetLogCBase(SVM,PetscReal);
FLLOP_EXTERN PetscErrorCode SVMGetLogCBase(SVM,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMSetLogCpMin(SVM,PetscReal);
FLLOP_EXTERN PetscErrorCode SVMGetLogCpMin(SVM,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMSetLogCpMax(SVM,PetscReal);
FLLOP_EXTERN PetscErrorCode SVMGetLogCpMax(SVM,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMSetLogCpBase(SVM,PetscReal);
FLLOP_EXTERN PetscErrorCode SVMGetLogCpBase(SVM,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMSetLogCnMin(SVM,PetscReal);
FLLOP_EXTERN PetscErrorCode SVMGetLogCnMin(SVM,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMSetLogCnMax(SVM,PetscReal);
FLLOP_EXTERN PetscErrorCode SVMGetLogCnMax(SVM,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMSetLogCnBase(SVM,PetscReal);
FLLOP_EXTERN PetscErrorCode SVMGetLogCnBase(SVM,PetscReal *);

FLLOP_EXTERN PetscErrorCode SVMSetCrossValidationType(SVM,CrossValidationType);
FLLOP_EXTERN PetscErrorCode SVMGetCrossValidationType(SVM,CrossValidationType *);
FLLOP_EXTERN PetscErrorCode SVMCrossValidation(SVM,PetscReal [],PetscInt,PetscReal []);
FLLOP_EXTERN PetscErrorCode SVMKFoldCrossValidation(SVM,PetscReal [],PetscInt,PetscReal []);
FLLOP_EXTERN PetscErrorCode SVMStratifiedKFoldCrossValidation(SVM,PetscReal [],PetscInt,PetscReal []);

FLLOP_EXTERN PetscErrorCode SVMSetQPS(SVM,QPS);
FLLOP_EXTERN PetscErrorCode SVMGetQPS(SVM,QPS *);
FLLOP_EXTERN PetscErrorCode SVMSetWarmStart(SVM,PetscBool);

FLLOP_EXTERN PetscErrorCode SVMSetOptionsPrefix(SVM svm,const char []);
FLLOP_EXTERN PetscErrorCode SVMAppendOptionsPrefix(SVM svm,const char []);
FLLOP_EXTERN PetscErrorCode SVMGetOptionsPrefix(SVM svm,const char *[]);

FLLOP_EXTERN PetscErrorCode MatCreate_Biased(Mat,PetscReal,Mat *);
#endif
