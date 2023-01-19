
#include "probimpl.h"

PetscErrorCode SVMReset_Probability(SVM svm)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&svm_prob->Xt_training));
  PetscCall(VecDestroy(&svm_prob->y_training));
  PetscCall(MatDestroy(&svm_prob->Xt_calib));
  PetscCall(VecDestroy(&svm_prob->y_calib));
  PetscFunctionReturn(0);
}

PetscErrorCode SVMDestroy_Probability(SVM svm)
{

  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetTrainingDataset_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetTrainingDataset_C",NULL));

  PetscCall(SVMDestroyDefault(svm));
  PetscFunctionReturn(0);
}

PetscErrorCode SVMView_Probability(SVM svm,PetscViewer v)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode SVMViewScore_Probability(SVM svm,PetscViewer v)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode SVMSetTrainingDataset_Probability(SVM svm,Mat Xt_training,Vec y_training)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  Mat      Xt_calib;
  PetscInt k,l,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(Xt_training,MAT_CLASSID,2);
  PetscCheckSameComm(svm,1,Xt_training,2);
  PetscValidHeaderSpecific(y_training,VEC_CLASSID,3);
  PetscCheckSameComm(svm,1,y_training,3);

  PetscCall(MatGetSize(Xt_training,&k,NULL));
  PetscCall(VecGetSize(y_training,&l));
  if (k != l) {
    SETERRQ(PetscObjectComm((PetscObject) Xt_training),PETSC_ERR_ARG_SIZ,"Dimensions are incompatible, X_training(%" PetscInt_FMT
                                                                         ",) != y_training(%" PetscInt_FMT ")",k,l);
  }

  PetscCall(SVMGetCalibrationDataset(svm,&Xt_calib,NULL));
  if (Xt_calib != NULL) {
    PetscCall(MatGetSize(Xt_training,NULL,&l));
    PetscCall(MatGetSize(Xt_calib,NULL,&n));

    if (l != n) {
      SETERRQ(PetscObjectComm((PetscObject) Xt_calib),PETSC_ERR_ARG_SIZ,"Dimensions are incompatible, "
                                                                        "X_training(,%" PetscInt_FMT ") !="
                                                                        "X_calib(,%" PetscInt_FMT ")",l,n);
    }
  }

  PetscCall(MatDestroy(&svm_prob->Xt_training));
  svm_prob->Xt_training = Xt_training;
  PetscCall(PetscObjectReference((PetscObject) Xt_training));

  PetscCall(VecDestroy(&svm_prob->y_training));
  svm_prob->y_training = y_training;
  PetscCall(PetscObjectReference((PetscObject) y_training));
  PetscFunctionReturn(0);
}

PetscErrorCode SVMGetTrainingDataset_Probability(SVM svm,Mat *Xt_training,Vec *y_training)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscFunctionBegin;
  if (Xt_training) {
    PetscValidPointer(Xt_training,2);
    *Xt_training = svm_prob->Xt_training;
  }
  if (y_training) {
    PetscValidPointer(y_training,3);
    *y_training = svm_prob->y_training;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVMSetCalibrationDataset_Probability(SVM svm,Mat Xt_calib,Vec y_calib)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  Mat      Xt_training;

  PetscInt k,l;
  PetscInt n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(Xt_calib,MAT_CLASSID,2);
  PetscCheckSameComm(svm,1,Xt_calib,2);
  PetscValidHeaderSpecific(y_calib,VEC_CLASSID,3);
  PetscCheckSameComm(svm,1,y_calib,3);

  PetscCall(MatGetSize(Xt_calib,&k,NULL));
  PetscCall(VecGetSize(y_calib,&l));
  if (k != l) {
    SETERRQ(PetscObjectComm((PetscObject) Xt_calib),PETSC_ERR_ARG_SIZ,"Dimensions are incompatible, X_calib(%" PetscInt_FMT
                                                                      ",) != y_calib(%" PetscInt_FMT ")",k,l);
  }

  PetscCall(SVMGetTrainingDataset(svm,&Xt_training,NULL));
  if (Xt_training != NULL) {
    PetscCall(MatGetSize(Xt_calib,NULL,&l));
    PetscCall(MatGetSize(Xt_training,NULL,&n));

    if (l != n) {
      SETERRQ(PetscObjectComm((PetscObject) Xt_calib),PETSC_ERR_ARG_SIZ,"Dimensions are incompatible, "
                                                                        "X_calib(,%" PetscInt_FMT ") !="
                                                                        "X_training(,%" PetscInt_FMT ")",l,n);
    }
  }

  PetscCall(MatDestroy(&svm_prob->Xt_calib));
  svm_prob->Xt_calib = Xt_calib;
  PetscCall(PetscObjectReference((PetscObject) Xt_calib));

  PetscCall(VecDestroy(&svm_prob->y_calib));
  svm_prob->y_calib = y_calib;
  PetscCall(PetscObjectReference((PetscObject) y_calib));
  PetscFunctionReturn(0);
}

PetscErrorCode SVMGetCalibrationDataset_Probability(SVM svm,Mat *Xt_calib,Vec *y_calib)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscFunctionBegin;
  if (Xt_calib) {
    PetscValidPointer(Xt_calib,2);
    *Xt_calib = svm_prob->Xt_calib;
  }
  if (y_calib) {
    PetscValidPointer(y_calib,3);
    *y_calib = svm_prob->y_calib;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVMSetUp_Probability(SVM svm)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscFunctionBegin;
  if (svm->setupcalled) PetscFunctionReturn(0);

  PetscFunctionReturn(0);
}

PetscErrorCode SVMSetOptionsPrefix_Probability(PetscOptionItems *PetscOptionsObject,SVM svm)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode SVMAppendOptionsPrefix_Probability(SVM svm)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode SVMGetOptionsPrefix_Probability(SVM svm)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode SVMSetFromOptions_Probability(SVM svm)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode SVMTrain_Probability(SVM svm)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode SVMPostTrain_Probability(SVM svm)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode SVMPredict_Probability(SVM svm,Mat Xt_pred,Vec *y_out)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode SVMComputeModelScores_Probability(SVM svm,Vec y_pred,Vec y_known)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode SVMTest_Probability(SVM svm)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode SVMLoadCalibrationDataset_Probability(SVM svm,PetscViewer v)
{
  MPI_Comm  comm;

  Mat       Xt_calib,Xt_biased;
  Vec       y_calib;

  PetscReal user_bias;
  PetscInt  mod;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) svm,&comm));
  PetscCall(MatCreate(comm,&Xt_calib));
  /* Create matrix of calibration samples */
  PetscCall(PetscObjectSetName((PetscObject) Xt_calib,"Xt_calib"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) Xt_calib,"Xt_calib_"));
  PetscCall(MatSetFromOptions(Xt_calib));
  /* Create label vector of calibration samples */
  PetscCall(VecCreate(comm,&y_calib));
  PetscCall(PetscObjectSetName((PetscObject) y_calib,"Xt_calib_"));
  PetscCall(VecSetFromOptions(y_calib));

  PetscCall(PetscLogEventBegin(SVM_LoadDataset,svm,0,0,0));
  PetscCall(PetscViewerLoadSVMDataset(Xt_calib,y_calib,v));
  PetscCall(PetscLogEventEnd(SVM_LoadDataset,svm,0,0,0));

  PetscCall(SVMGetMod(svm,&mod));
  if (mod == 2) {
    PetscCall(SVMGetUserBias(svm,&user_bias));
    PetscCall(MatBiasedCreate(Xt_calib,user_bias,&Xt_biased));
    Xt_calib = Xt_biased;
  }
  PetscCall(SVMSetCalibrationDataset(svm,Xt_calib,y_calib));

  /* Clean up */
  PetscCall(MatDestroy(&Xt_calib));
  PetscCall(VecDestroy(&y_calib));
  PetscFunctionReturn(0);
}

PetscErrorCode SVMViewCalibrationDataset_Probability(SVM svm,PetscViewer v)
{
  MPI_Comm   comm;

  Mat        Xt;
  Vec        y;

  PetscBool  isascii;
  const char *type_name = NULL;

  PetscFunctionBegin;
  PetscCall(SVMGetCalibrationDataset(svm,&Xt,&y));
  if (!Xt || !y) {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    SETERRQ(comm,PETSC_ERR_ARG_NULL,"Calibration dataset is not set");
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    /* Print info related to svm type */
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject) svm,v));

    PetscCall(PetscViewerASCIIPushTab(v));
    PetscCall(SVMViewDataset(svm,Xt,y,v));
    PetscCall(PetscViewerASCIIPopTab(v));
  } else {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    PetscCall(PetscObjectGetType((PetscObject) v,&type_name));

    SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewCalibrationDataset",type_name);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVMCreate_Probability(SVM svm)
{
  SVM_Probability *svm_prob;

  PetscFunctionBegin;
  PetscCall(PetscNew(&svm_prob));
  svm->data = (void *) svm_prob;

  svm_prob->Xt_training = NULL;
  svm_prob->y_training  = NULL;
  svm_prob->Xt_calib    = NULL;
  svm_prob->y_calib     = NULL;

  svm->ops->setup              = SVMSetUp_Probability;
  svm->ops->reset              = SVMReset_Probability;
  svm->ops->destroy            = SVMDestroy_Probability;
  svm->ops->setfromoptions     = SVMSetOptionsPrefix_Probability;
  svm->ops->train              = SVMTrain_Probability;
  svm->ops->posttrain          = SVMPostTrain_Probability;
  svm->ops->predict            = SVMPredict_Probability;
  svm->ops->test               = SVMTest_Probability;
  svm->ops->view               = SVMView_Probability;
  svm->ops->viewscore          = SVMViewScore_Probability;
  svm->ops->computemodelscores = SVMComputeModelScores_Probability;

  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetTrainingDataset_C",SVMSetTrainingDataset_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetTrainingDataset_C",SVMGetTrainingDataset_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetCalibrationDataset_C",SVMSetCalibrationDataset_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetCalibrationDataset_C",SVMGetCalibrationDataset_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMLoadCalibrationDataset_C",SVMLoadCalibrationDataset_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMViewCalibrationDataset_C",SVMViewCalibrationDataset_Probability));

  PetscFunctionReturn(0);
}
