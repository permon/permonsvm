
#include "report.h"

PetscErrorCode BinaryConfusionMatrix(SVM svm,Vec y_pred,Vec y_known,PetscInt confusion_mat[])
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

  /* Clean up */
  PetscCall(VecDestroy(&vec_label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMGetBinaryClassificationReport(SVM svm,Vec y_pred,Vec y_known,PetscInt *cmat, PetscReal *scores)
{
  PetscInt  confusion_mat[4];
  PetscInt  TP,FP,TN,FN;

  PetscReal accuracy[2];
  PetscReal auc_roc;
  PetscReal precision[3];
  PetscReal sensitivity[3];
  PetscReal F1[3];
  PetscReal jaccard_index[3];

  PetscReal specifity;
  PetscReal TPR,FPR;
  PetscReal x[3],y[3],dx;

  PetscInt  i;

  PetscFunctionBegin;
  PetscCall(BinaryConfusionMatrix(svm,y_pred,y_known,confusion_mat));

  TP = confusion_mat[0];
  FP = confusion_mat[1];
  FN = confusion_mat[2];
  TN = confusion_mat[3];

  if (cmat != NULL) {
    PetscCall(PetscMemcpy(cmat,confusion_mat,4 * sizeof(PetscInt)));
  }

  if (scores == NULL) {
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  precision[0] = (PetscReal) TP / (PetscReal) (TP + FP); /* precision (positive class) */
  precision[1] = (PetscReal) TN / (PetscReal) (TN + FN); /* precision (negative class) */
  precision[2] = (precision[0] + precision[1]) / 2.;     /* precision (average)        */

  sensitivity[0] = (PetscReal) TP / (PetscReal) (TP + FN); /* sensitivity (positive class) */
  sensitivity[1] = (PetscReal) TN / (PetscReal) (TN + FP); /* sensitivity (negative class) */
  sensitivity[2] = (sensitivity[0] + sensitivity[1]) / 2.; /* sensitivity (average)        */

  F1[0] = 2. * (precision[0] * sensitivity[0]) / (precision[0] + sensitivity[0]); /* F1 (positive class) */
  F1[1] = 2. * (precision[1] * sensitivity[1]) / (precision[1] + sensitivity[1]); /* F1 (negative class) */
  F1[2] = (F1[0] + F1[1]) / 2.;                                                   /* F1 (average) */

  jaccard_index[0] = (PetscReal) TP / (PetscReal) (TP + FP + FN); /* Jaccard index (positive class) */
  jaccard_index[1] = (PetscReal) TN / (PetscReal) (TN + FN + FP); /* Jaccard index (negative class) */
  jaccard_index[2] = (jaccard_index[0] + jaccard_index[1]) / 2.;  /* Jaccard index (average) */

  /* Area under curve (trapezoidal rule) */
  specifity = (PetscReal) TN / (PetscReal) (TN + FP);

  FPR = 1 - specifity;
  TPR = sensitivity[0];

  x[0] = 0; x[1] = FPR; x[2] = 1;
  y[0] = 0; y[1] = TPR; y[2] = 1;

  auc_roc = 0.;
  for (i = 0; i < 2; ++i) {
    dx = x[i+1] - x[i];
    auc_roc += ((y[i] + y[i+1]) / 2.) * dx;
  }

  /* Accuracy */
  accuracy[0] = (PetscReal) (TP + TN) / (PetscReal) (TP + TN + FP + FN);

  /* Set scores */
  scores[0] = accuracy[0];

  scores[1] = precision[0];
  scores[2] = precision[1];
  scores[3] = precision[2];

  scores[4] = sensitivity[0];
  scores[5] = sensitivity[1];
  scores[6] = sensitivity[2];

  scores[7] = F1[0];
  scores[8] = F1[1];
  scores[9] = F1[2];

  scores[10] = jaccard_index[0];
  scores[11] = jaccard_index[1];
  scores[12] = jaccard_index[2];

  scores[13] = auc_roc;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMPrintBinaryClassificationReport(SVM svm,PetscInt *cmat,PetscReal *scores,PetscViewer v)
{
  const PetscReal *labels;

  PetscFunctionBegin;
  PetscCall(SVMGetLabels(svm,&labels));

  /* Print confusion matrix */
  PetscCall(PetscViewerASCIIPrintf(v,"Confusion matrix:\n"));
  PetscCall(PetscViewerASCIIPushTab(v));
  PetscCall(PetscViewerASCIIPrintf(v,"TP = %4" PetscInt_FMT ""  ,cmat[0]));
  PetscCall(PetscViewerASCIIPrintf(v,"FP = %4" PetscInt_FMT "\n",cmat[1]));
  PetscCall(PetscViewerASCIIPrintf(v,"FN = %4" PetscInt_FMT ""  ,cmat[2]));
  PetscCall(PetscViewerASCIIPrintf(v,"TN = %4" PetscInt_FMT "\n",cmat[3]));
  PetscCall(PetscViewerASCIIPopTab(v));

  /* Print classification report */
  PetscCall(PetscViewerASCIIPrintf(v,"Classification report:\n"));
  /* Header */
  PetscCall(PetscViewerASCIIPushTab(v));
  PetscCall(PetscViewerASCIIPrintf(v,"label\tprecision\trecall\tF1\tJaccard index\n"));
  PetscCall(PetscViewerASCIIPrintf(v,"%.1f  \t%.2f\t\t%.2f\t%.2f\t%.2f\n",labels[0],scores[2],scores[5],scores[8],scores[11]));
  PetscCall(PetscViewerASCIIPrintf(v,"%.1f  \t%.2f\t\t%.2f\t%.2f\t%.2f\n",labels[1],scores[1],scores[4],scores[7],scores[10]));
  PetscCall(PetscViewerASCIIPrintf(v,"mean  \t%.2f\t\t%.2f\t%.2f\t%.2f\n"          ,scores[3],scores[6],scores[9],scores[12]));
  PetscCall(PetscViewerASCIIPrintf(v,"accuracy = %.2f%%\n",(double) scores[0] * 100));
  PetscCall(PetscViewerASCIIPrintf(v,"auc_roc  = %.2f%%\n",(double) scores[13] * 100));
  PetscCall(PetscViewerASCIIPopTab(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}
