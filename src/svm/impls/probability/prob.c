
#include "probimpl.h"

PetscErrorCode SVMReset_Probability(SVM svm)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode SVMDestroy_Probability(SVM svm)
{

  PetscFunctionBegin;
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

PetscErrorCode SVMSetUp_Probability(SVM svm)
{

  PetscFunctionBegin;
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

PetscErrorCode SVMCreate_Probability(SVM svm)
{
  SVM_Probability *svm_prob;

  PetscFunctionBegin;
  PetscCall(PetscNew(&svm_prob));
  svm->data = (void *) svm_prob;

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

  PetscFunctionReturn(0);
}
