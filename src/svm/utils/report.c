
#include "report.h"

PetscErrorCode ConfusionMatrix(SVM svm,Vec y_pred,Vec y_known,PetscInt confusion_mat[])
{
  PetscInt TP,FP,TN,FN; // confusion matrix values
  PetscInt N;

  Vec      y_known_sub;
  Vec      y_pred_sub;

  Vec      vec_label;
  IS       is_label;

  IS       is_eq;

  const PetscReal *labels = NULL;

  PetscFunctionBegin;

  PetscCall(SVMGetLabels(svm,&labels));
  PetscCall(VecDuplicate(y_pred,&vec_label));

  /* TN and FN samples */
  PetscCall(VecSet(vec_label,labels[0]));

  PetscCall(VecWhichEqual(y_known,vec_label,&is_label));
  PetscCall(VecGetSubVector(y_known,is_label,&y_known_sub));
  PetscCall(VecGetSubVector(y_pred,is_label,&y_pred_sub));
  PetscCall(VecWhichEqual(y_known_sub,y_pred_sub,&is_eq));

  PetscCall(ISGetSize(is_eq,&TN));
  PetscCall(VecGetSize(y_known_sub,&N));
  FN = N - TN;

  PetscCall(VecRestoreSubVector(y_known,is_label,&y_known_sub));
  PetscCall(VecRestoreSubVector(y_pred,is_label,&y_pred_sub));
  PetscCall(ISDestroy(&is_label));
  PetscCall(ISDestroy(&is_eq));

  /* TP and FP samples */
  PetscCall(VecSet(vec_label,labels[1]));

  PetscCall(VecWhichEqual(y_known,vec_label,&is_label));
  PetscCall(VecGetSubVector(y_known,is_label,&y_known_sub));
  PetscCall(VecGetSubVector(y_pred,is_label,&y_pred_sub));
  PetscCall(VecWhichEqual(y_known_sub,y_pred_sub,&is_eq));

  PetscCall(ISGetSize(is_eq,&TP));
  PetscCall(VecGetSize(y_known_sub,&N));
  FP = N - TP;

  PetscCall(VecRestoreSubVector(y_known,is_label,&y_known_sub));
  PetscCall(VecRestoreSubVector(y_pred,is_label,&y_pred_sub));
  PetscCall(ISDestroy(&is_label));
  PetscCall(ISDestroy(&is_eq));

  confusion_mat[0] = TP;
  confusion_mat[1] = FP;
  confusion_mat[2] = FN;
  confusion_mat[3] = TN;

  /* Clen up */
  PetscCall(VecDestroy(&vec_label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ClassificationReport(SVM svm,Vec y_pred,Vec y_known)
{
  PetscInt confusion_mat[4];

  PetscFunctionBegin;
  PetscCall(ConfusionMatrix(svm,y_pred,y_known,confusion_mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}
