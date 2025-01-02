#pragma once

#include <permonqps.h>
#include <petscviewerhdf5.h>

/*S
  SVM - PERMON class for Support Vector Machines (SVM) classification on top of PermonQP solvers (QPS)

  Level: beginner

  Concepts: SVM classification

.seealso:  SVMCreate(), SVMDestroy(), QP, QPS
 S*/
typedef struct _p_SVM* SVM;

PERMON_EXTERN PetscClassId SVM_CLASSID;
#define SVM_CLASS_NAME  "svm"

#define SVMType  char*
#define SVM_BINARY      "binary"
#define SVM_PROBABILITY "probability"

/*E
  SVMLossType - Determines the loss function for soft-margin SVM (non-separable samples).

  Level: beginner

.seealso: SVMSetLossType, SVMGetLossType()
E*/
typedef enum {SVM_L1, SVM_L2} SVMLossType;
PERMON_EXTERN const char *const SVMLossTypes[];

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
PERMON_EXTERN const char *const ModelScores[];

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
PERMON_EXTERN const char *const CrossValidationTypes[];

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

PERMON_EXTERN PetscErrorCode SVMInitializePackage();
PERMON_EXTERN PetscErrorCode SVMFinalizePackage();

PERMON_EXTERN PetscFunctionList SVMList;
PERMON_EXTERN PetscBool SVMRegisterAllCalled;
PERMON_EXTERN PetscErrorCode SVMRegisterAll();
PERMON_EXTERN PetscErrorCode SVMRegister(const char [],PetscErrorCode (*function)(SVM));

PERMON_EXTERN PetscErrorCode SVMCreate(MPI_Comm,SVM *);
PERMON_EXTERN PetscErrorCode SVMSetType(SVM,const SVMType);
PERMON_EXTERN PetscErrorCode SVMReset(SVM);
PERMON_EXTERN PetscErrorCode SVMDestroy(SVM *);
PERMON_EXTERN PetscErrorCode SVMDestroyDefault(SVM);
PERMON_EXTERN PetscErrorCode SVMSetFromOptions(SVM);
PERMON_EXTERN PetscErrorCode SVMSetUp(SVM);
PERMON_EXTERN PetscErrorCode SVMView(SVM,PetscViewer);
PERMON_EXTERN PetscErrorCode SVMViewScore(SVM,PetscViewer);

PERMON_EXTERN PetscErrorCode SVMSetGramian(SVM,Mat);
PERMON_EXTERN PetscErrorCode SVMGetGramian(SVM,Mat *);
PERMON_EXTERN PetscErrorCode SVMSetOperator(SVM,Mat);
PERMON_EXTERN PetscErrorCode SVMGetOperator(SVM,Mat *);
PERMON_EXTERN PetscErrorCode SVMComputeOperator(SVM,Mat *);

PERMON_EXTERN PetscErrorCode SVMSetTrainingDataset(SVM,Mat,Vec);
PERMON_EXTERN PetscErrorCode SVMGetTrainingDataset(SVM,Mat *,Vec *);
PERMON_EXTERN PetscErrorCode SVMSetCalibrationDataset(SVM,Mat,Vec);
PERMON_EXTERN PetscErrorCode SVMGetCalibrationDataset(SVM,Mat *,Vec *);
PERMON_EXTERN PetscErrorCode SVMSetTestDataset(SVM,Mat,Vec);
PERMON_EXTERN PetscErrorCode SVMGetTestDataset(SVM,Mat *,Vec *);

PERMON_EXTERN PetscErrorCode SVMGetLabels(SVM,const PetscReal *[]);

PERMON_EXTERN PetscErrorCode SVMSetMod(SVM,PetscInt);
PERMON_EXTERN PetscErrorCode SVMGetMod(SVM,PetscInt *);
PERMON_EXTERN PetscErrorCode SVMSetLossType(SVM,SVMLossType);
PERMON_EXTERN PetscErrorCode SVMGetLossType(SVM,SVMLossType *);
PERMON_EXTERN PetscErrorCode SVMSetPenaltyType(SVM,PetscInt);
PERMON_EXTERN PetscErrorCode SVMGetPenaltyType(SVM,PetscInt *);
PERMON_EXTERN PetscErrorCode SVMSetPenalty(SVM,PetscInt,PetscReal []);
PERMON_EXTERN PetscErrorCode SVMSetC(SVM,PetscReal);
PERMON_EXTERN PetscErrorCode SVMGetC(SVM,PetscReal *);
PERMON_EXTERN PetscErrorCode SVMSetCp(SVM,PetscReal);
PERMON_EXTERN PetscErrorCode SVMGetCp(SVM,PetscReal *);
PERMON_EXTERN PetscErrorCode SVMSetCn(SVM,PetscReal);
PERMON_EXTERN PetscErrorCode SVMGetCn(SVM,PetscReal *);
PERMON_EXTERN PetscErrorCode SVMSetNfolds(SVM,PetscInt);
PERMON_EXTERN PetscErrorCode SVMGetNfolds(SVM,PetscInt *);
PERMON_EXTERN PetscErrorCode SVMSetSeparatingHyperplane(SVM,Vec,PetscReal);
PERMON_EXTERN PetscErrorCode SVMGetSeparatingHyperplane(SVM,Vec *,PetscReal *);
PERMON_EXTERN PetscErrorCode SVMSetBias(SVM,PetscReal);
PERMON_EXTERN PetscErrorCode SVMGetBias(SVM,PetscReal *);
PERMON_EXTERN PetscErrorCode SVMSetUserBias(SVM,PetscReal);
PERMON_EXTERN PetscErrorCode SVMGetUserBias(SVM,PetscReal *);

PERMON_EXTERN PetscErrorCode SVMTrain(SVM);
PERMON_EXTERN PetscErrorCode SVMPostTrain(SVM);
PERMON_EXTERN PetscErrorCode SVMReconstructHyperplane(SVM);
PERMON_EXTERN PetscErrorCode SVMSetAutoPostTrain(SVM,PetscBool);
PERMON_EXTERN PetscErrorCode SVMGetAutoPostTrain(SVM,PetscBool *);
PERMON_EXTERN PetscErrorCode SVMGetDistancesFromHyperplane(SVM,Mat,Vec *);
PERMON_EXTERN PetscErrorCode SVMPredict(SVM,Mat,Vec *);
PERMON_EXTERN PetscErrorCode SVMTest(SVM);

PERMON_EXTERN PetscErrorCode SVMConvergedSetUp(SVM);
PERMON_EXTERN PetscErrorCode SVMDefaultConvergedCreate(SVM, void **);
PERMON_EXTERN PetscErrorCode SVMDefaultConvergedDestroy(void *);

PERMON_EXTERN PetscErrorCode SVMComputeModelScores(SVM,Vec,Vec);
PERMON_EXTERN PetscErrorCode SVMGetModelScore(SVM,ModelScore,PetscReal *);
PERMON_EXTERN PetscErrorCode SVMComputeHingeLoss(SVM svm);
PERMON_EXTERN PetscErrorCode SVMComputeModelParams(SVM svm);

PERMON_EXTERN PetscErrorCode SVMSetHyperOpt(SVM,PetscBool);
PERMON_EXTERN PetscErrorCode SVMSetHyperOptScoreTypes(SVM,PetscInt,ModelScore []);
PERMON_EXTERN PetscErrorCode SVMGetHyperOptNScoreTypes(SVM,PetscInt *);
PERMON_EXTERN PetscErrorCode SVMGetHyperOptScoreTypes(SVM svm,const ModelScore *types[]);

PERMON_EXTERN PetscErrorCode SVMGridSearch(SVM);
/* Penalty type 1 */
PERMON_EXTERN PetscErrorCode SVMGridSearchSetBaseLogC(SVM,PetscReal);
PERMON_EXTERN PetscErrorCode SVMGridSearchGetBaseLogC(SVM,PetscReal *);
PERMON_EXTERN PetscErrorCode SVMGridSearchSetStrideLogC(SVM,PetscReal,PetscReal,PetscReal);
PERMON_EXTERN PetscErrorCode SVMGridSearchGetStrideLogC(SVM,PetscReal *,PetscReal *,PetscReal *);
/* Penalty type 2 */
PERMON_EXTERN PetscErrorCode SVMGridSearchSetPositiveBaseLogC(SVM,PetscReal);
PERMON_EXTERN PetscErrorCode SVMGridSearchGetPositiveBaseLogC(SVM,PetscReal *);
PERMON_EXTERN PetscErrorCode SVMGridSearchSetPositiveStrideLogC(SVM,PetscReal,PetscReal,PetscReal);
PERMON_EXTERN PetscErrorCode SVMGridSearchGetPositiveStrideLogC(SVM,PetscReal *,PetscReal *,PetscReal *);
PERMON_EXTERN PetscErrorCode SVMGridSearchSetNegativeBaseLogC(SVM,PetscReal);
PERMON_EXTERN PetscErrorCode SVMGridSearchGetNegativeBaseLogC(SVM,PetscReal *);
PERMON_EXTERN PetscErrorCode SVMGridSearchSetNegativeStrideLogC(SVM,PetscReal,PetscReal,PetscReal);
PERMON_EXTERN PetscErrorCode SVMGridSearchGetNegativeStrideLogC(SVM,PetscReal *,PetscReal *,PetscReal *);

PERMON_EXTERN PetscErrorCode SVMSetCrossValidationType(SVM,CrossValidationType);
PERMON_EXTERN PetscErrorCode SVMGetCrossValidationType(SVM,CrossValidationType *);
PERMON_EXTERN PetscErrorCode SVMCrossValidation(SVM,PetscReal [],PetscInt,PetscReal []);
PERMON_EXTERN PetscErrorCode SVMKFoldCrossValidation(SVM,PetscReal [],PetscInt,PetscReal []);
PERMON_EXTERN PetscErrorCode SVMStratifiedKFoldCrossValidation(SVM,PetscReal [],PetscInt,PetscReal []);

PERMON_EXTERN PetscErrorCode SVMSetQPS(SVM,QPS);
PERMON_EXTERN PetscErrorCode SVMGetQPS(SVM,QPS *);
PERMON_EXTERN PetscErrorCode SVMGetQP(SVM,QP *);
PERMON_EXTERN PetscErrorCode SVMSetWarmStart(SVM,PetscBool);

PERMON_EXTERN PetscErrorCode SVMSetOptionsPrefix(SVM svm,const char []);
PERMON_EXTERN PetscErrorCode SVMAppendOptionsPrefix(SVM svm,const char []);
PERMON_EXTERN PetscErrorCode SVMGetOptionsPrefix(SVM svm,const char *[]);

PERMON_EXTERN PetscErrorCode SVMGetTao(SVM,Tao *);
PERMON_EXTERN PetscErrorCode SVMGetInnerSVM(SVM,SVM *);

/* SVM probability */
PERMON_EXTERN PetscErrorCode SVMProbabilitySetConvertLabelsToTargetProbability(SVM,PetscBool);
PERMON_EXTERN PetscErrorCode SVMProbabilityGetConvertLabelsToTargetProbability(SVM,PetscBool *);
PERMON_EXTERN PetscErrorCode SVMProbabilityGetSigmoidParams(SVM,PetscReal *,PetscReal *);
PERMON_EXTERN PetscErrorCode SVMProbabilitySetThreshold(SVM,PetscReal);
PERMON_EXTERN PetscErrorCode SVMProbabilityGetThreshold(SVM,PetscReal *);
PERMON_EXTERN PetscErrorCode SVMProbabilityConvertProbabilityToLabels(SVM,Vec);

/* Input/Output functions */
PERMON_EXTERN PetscErrorCode PetscViewerLoadSVMDataset(Mat,Vec,PetscViewer);
PERMON_EXTERN PetscErrorCode SVMViewDataset(SVM,Mat,Vec,PetscViewer);

PERMON_EXTERN PetscErrorCode SVMLoadGramian(SVM,PetscViewer);
PERMON_EXTERN PetscErrorCode SVMViewGramian(SVM,PetscViewer);
PERMON_EXTERN PetscErrorCode SVMLoadDataset(SVM,PetscViewer,Mat,Vec);
PERMON_EXTERN PetscErrorCode SVMDatasetInfo(SVM,Mat,Vec,PetscViewer);
PERMON_EXTERN PetscErrorCode SVMLoadTrainingDataset(SVM,PetscViewer);
PERMON_EXTERN PetscErrorCode SVMViewTrainingDataset(SVM,PetscViewer);
PERMON_EXTERN PetscErrorCode SVMLoadTestDataset(SVM,PetscViewer);
PERMON_EXTERN PetscErrorCode SVMViewTestDataset(SVM,PetscViewer);
PERMON_EXTERN PetscErrorCode SVMLoadCalibrationDataset(SVM,PetscViewer);
PERMON_EXTERN PetscErrorCode SVMViewCalibrationDataset(SVM,PetscViewer);

PERMON_EXTERN PetscErrorCode SVMViewTrainingPredictions(SVM,PetscViewer);
PERMON_EXTERN PetscErrorCode SVMViewTestPredictions(SVM,PetscViewer);

PERMON_EXTERN PetscErrorCode PetscViewerSVMLightOpen(MPI_Comm,const char [],PetscViewer *);

/* Functions related to a biased matrix */
PERMON_EXTERN PetscErrorCode MatBiasedCreate(Mat,PetscReal,Mat *);
PERMON_EXTERN PetscErrorCode MatBiasedGetInnerMat(Mat,Mat *);
PERMON_EXTERN PetscErrorCode MatBiasedGetBias(Mat,PetscReal *);
