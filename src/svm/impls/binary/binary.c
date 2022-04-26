
#include "binaryimpl.h"

const char *const SVMLossTypes[]={"L1","L2","SVMLossType","SVM_",0};

typedef struct {
  SVM svm_inner;
} SVM_Binary_mctx;

static PetscErrorCode SVMMonitorCreateCtx_Binary(void **,SVM);
static PetscErrorCode SVMMonitorDestroyCtx_Binary(void **);

static PetscErrorCode SVMMonitorDefault_Binary(QPS,PetscInt,PetscReal,void *);
static PetscErrorCode SVMMonitorObjFuncs_Binary(QPS,PetscInt,PetscReal,void *);
static PetscErrorCode SVMMonitorScores_Binary(QPS,PetscInt,PetscReal,void *);
static PetscErrorCode SVMMonitorTrainingScores_Binary(QPS,PetscInt,PetscReal,void *);

#undef __FUNCT__
#define __FUNCT__ "SVMReset_Binary"
PetscErrorCode SVMReset_Binary(SVM svm)
{
  SVM_Binary *svm_binary;

  PetscInt i;
  PetscFunctionBegin;
  svm_binary = (SVM_Binary *) svm->data;

  if (svm_binary->qps) {
    PetscCall(QPSDestroy(&svm_binary->qps));
    svm_binary->qps = NULL;
  }
  PetscCall(VecDestroy(&svm_binary->w));
  PetscCall(MatDestroy(&svm_binary->Xt_training));
  PetscCall(MatDestroy(&svm_binary->G));
  PetscCall(MatDestroy(&svm_binary->J));
  PetscCall(VecDestroy(&svm_binary->diag));
  PetscCall(VecDestroy(&svm_binary->y_training));
  PetscCall(VecDestroy(&svm_binary->y_inner));
  PetscCall(ISDestroy(&svm_binary->is_p));
  PetscCall(ISDestroy(&svm_binary->is_n));

  PetscCall(PetscMemzero(svm_binary->y_map,2 * sizeof(PetscScalar)));
  PetscCall(PetscMemzero(svm_binary->confusion_matrix,4 * sizeof(PetscInt)));
  PetscCall(PetscMemzero(svm_binary->model_scores,7 * sizeof(PetscReal)));

  svm_binary->w           = NULL;
  svm_binary->Xt_training = NULL;
  svm_binary->y_training  = NULL;
  svm_binary->y_inner     = NULL;
  svm_binary->is_p        = NULL;
  svm_binary->is_n        = NULL;
  svm_binary->G           = NULL;
  svm_binary->J           = NULL;
  svm_binary->diag        = NULL;

  svm_binary->nsv         = 0;
  PetscCall(ISDestroy(&svm_binary->is_sv));
  svm_binary->is_sv       = NULL;

  for (i = 0; i < 3; ++i) {
    PetscCall(VecDestroy(&svm_binary->work[i]));
    svm_binary->work[i] = NULL;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SVMDestroy_Binary"
PetscErrorCode SVMDestroy_Binary(SVM svm)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetGramian_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetGramian_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetOperator_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetOperator_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetTrainingDataset_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetTrainingDataset_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMComputeOperator_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetQPS_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetQPS_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetQP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetBias_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetBias_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetSeparatingHyperplane_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetSeparatingHyperplane_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetModelScore_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetOptionsPrefix_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetOptionsPrefix_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMAppendOptionsPrefix_C",NULL));

  PetscCall(QPSDestroy(&svm_binary->qps));
  PetscCall(SVMDestroyDefault(svm));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMView_Binary"
PetscErrorCode SVMView_Binary(SVM svm,PetscViewer v)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  MPI_Comm    comm;
  SVMLossType loss_type;
  PetscInt    p;
  PetscBool   isascii;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject) svm);

  if (!v) v = PETSC_VIEWER_STDOUT_(comm);

  PetscCall(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&isascii));

  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject) svm,v));

    PetscCall(PetscViewerASCIIPushTab(v));
    PetscCall(PetscViewerASCIIPrintf(v,"model parameters:\n"));
    PetscCall(PetscViewerASCIIPushTab(v));
    PetscCall(PetscViewerASCIIPrintf(v,"||w||=%.4f",svm_binary->norm_w));
    PetscCall(PetscViewerASCIIPrintf(v,"bias=%.4f",svm_binary->b));
    PetscCall(PetscViewerASCIIPrintf(v,"margin=%.4f",svm_binary->margin));
    PetscCall(PetscViewerASCIIPrintf(v,"NSV=%d\n",svm_binary->nsv));
    PetscCall(PetscViewerASCIIPopTab(v));

    PetscCall(SVMGetLossType(svm,&loss_type));
    PetscCall(SVMGetPenaltyType(svm,&p));
    PetscCall(PetscViewerASCIIPrintf(v,"%s hinge loss:\n",SVMLossTypes[loss_type]));
    PetscCall(PetscViewerASCIIPushTab(v));
    if (p == 1) {
      if (loss_type == SVM_L1) {
        PetscCall(PetscViewerASCIIPrintf(v,"sum(xi_i)=%.4f\n",svm_binary->hinge_loss));
      } else {
        PetscCall(PetscViewerASCIIPrintf(v,"sum(xi_i^2)=%.4f\n",svm_binary->hinge_loss));
      }
    } else {
      if (loss_type == SVM_L1) {
        PetscCall(PetscViewerASCIIPrintf(v,"sum(xi_i+)=%.4f",svm_binary->hinge_loss_p));
        PetscCall(PetscViewerASCIIPrintf(v,"sum(xi_i-)=%.4f\n",svm_binary->hinge_loss_n));
      } else {
        PetscCall(PetscViewerASCIIPrintf(v,"sum(xi_i+^2)=%.4f",svm_binary->hinge_loss_n));
        PetscCall(PetscViewerASCIIPrintf(v,"sum(xi_i-^2)=%.4f\n",svm_binary->hinge_loss_n));
      }
    }
    PetscCall(PetscViewerASCIIPopTab(v));

    PetscCall(PetscViewerASCIIPrintf(v,"objective functions:\n",SVMLossTypes[loss_type]));
    PetscCall(PetscViewerASCIIPushTab(v));
    PetscCall(PetscViewerASCIIPrintf(v,"primalObj=%.4f",svm_binary->primalObj));
    PetscCall(PetscViewerASCIIPrintf(v,"dualObj=%.4f",svm_binary->dualObj));
    PetscCall(PetscViewerASCIIPrintf(v,"gap=%.4f\n",svm_binary->primalObj - svm_binary->dualObj));
    PetscCall(PetscViewerASCIIPopTab(v));

    PetscCall(PetscViewerASCIIPopTab(v));
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewScore", ((PetscObject)v)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMViewScore_Binary"
PetscErrorCode SVMViewScore_Binary(SVM svm,PetscViewer v)
{
  MPI_Comm  comm;

  PetscReal   C,Cp,Cn;
  PetscInt    mod;
  SVMLossType loss_type;
  PetscInt    p; /* penalty type */

  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;
  PetscBool isascii;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject) svm);
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);

  PetscCall(PetscObjectTypeCompare((PetscObject) v,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(SVMGetPenaltyType(svm,&p));
    PetscCall(SVMGetMod(svm,&mod));
    PetscCall(SVMGetLossType(svm,&loss_type));

    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject) svm,v));

    PetscCall(PetscViewerASCIIPushTab(v));
    if (p == 1) {
      PetscCall(SVMGetC(svm,&C));
      PetscCall(PetscViewerASCIIPrintf(v,"model performance score with training parameters C=%.3f, mod=%d, loss=%s:\n",C,mod,SVMLossTypes[loss_type]));
    } else {
      PetscCall(SVMGetCp(svm,&Cp));
      PetscCall(SVMGetCn(svm,&Cn));
      PetscCall(PetscViewerASCIIPrintf(v,"model performance score with training parameters C+=%.3f, C-=%.3f, mod=%d, loss=%s:\n",Cp,Cn,mod,SVMLossTypes[loss_type]));
    }
    PetscCall(PetscViewerASCIIPushTab(v));

    PetscCall(PetscViewerASCIIPrintf(v,"Confusion matrix:\n"));
    PetscCall(PetscViewerASCIIPushTab(v));
    PetscCall(PetscViewerASCIIPrintf(v,"TP = %4d",svm_binary->confusion_matrix[0]));
    PetscCall(PetscViewerASCIIPrintf(v,"FP = %4d\n",svm_binary->confusion_matrix[1]));
    PetscCall(PetscViewerASCIIPrintf(v,"FN = %4d",svm_binary->confusion_matrix[2]));
    PetscCall(PetscViewerASCIIPrintf(v,"TN = %4d\n",svm_binary->confusion_matrix[3]));
    PetscCall(PetscViewerASCIIPopTab(v));

    PetscCall(PetscViewerASCIIPrintf(v,"accuracy=%.2f%%",svm_binary->model_scores[0] * 100.));
    PetscCall(PetscViewerASCIIPrintf(v,"precision=%.2f%%",svm_binary->model_scores[1] * 100.));
    PetscCall(PetscViewerASCIIPrintf(v,"sensitivity=%.2f%%\n",svm_binary->model_scores[2] * 100.));
    PetscCall(PetscViewerASCIIPrintf(v,"F1=%.2f",svm_binary->model_scores[3]));
    PetscCall(PetscViewerASCIIPrintf(v,"MCC=%.2f",svm_binary->model_scores[4]));
    PetscCall(PetscViewerASCIIPrintf(v,"AUC_ROC=%.2f",svm_binary->model_scores[5]));
    PetscCall(PetscViewerASCIIPrintf(v,"G1=%.2f\n",svm_binary->model_scores[6]));
    PetscCall(PetscViewerASCIIPopTab(v));

    PetscCall(PetscViewerASCIIPopTab(v));
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewScore", ((PetscObject)v)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetGramian_Binary"
PetscErrorCode SVMSetGramian_Binary(SVM svm,Mat G)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  Mat        Xt;
  PetscInt   m,n;

  PetscFunctionBegin;
  /* Checking that Gramian is square */
  PetscCall(MatGetSize(G,&m,&n));
  if (m != n) {
    SETERRQ(PetscObjectComm((PetscObject) G),PETSC_ERR_ARG_SIZ,"Gramian (kernel) matrix must be square, G(%D,%D)",m,n);
  }
  /* Checking dimension compatibility between training data matrix and Gramian */
  PetscCall(SVMGetTrainingDataset(svm,&Xt,NULL));
  if (Xt) {
    PetscCall(MatGetSize(Xt,&n,NULL));
    if (m != n) {
      SETERRQ(PetscObjectComm((PetscObject) G),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, G(%D,) != X_training(%D,)",m,n);
    }
  }
  PetscCall(MatDestroy(&svm_binary->G));
  PetscCall(PetscObjectReference((PetscObject) G));
  svm_binary->G = G;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetGramian_Binary"
PetscErrorCode SVMGetGramian_Binary(SVM svm,Mat *G)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  *G = svm_binary->G;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetOperator_Binary"
PetscErrorCode SVMSetOperator_Binary(SVM svm,Mat A)
{
  QP       qp;

  Mat      Xt;
  PetscInt m,n;

  PetscFunctionBegin;
  /* Checking that operator (Hessian) is square */
  PetscCall(MatGetSize(A,&m,&n));
  if (m != n) {
    SETERRQ(PetscObjectComm((PetscObject) A),PETSC_ERR_ARG_SIZ,"Hessian matrix must be square, G(%D,%D)",m,n);
  }
  /* Checking dimension compatibility between Hessian (operator) and training data matrices */
  PetscCall(SVMGetTrainingDataset(svm,&Xt,NULL));
  if (Xt) {
    PetscCall(MatGetSize(Xt,&n,NULL));
    if (m != n) {
      SETERRQ(PetscObjectComm((PetscObject) A),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, A(%D,) != X_training(%D,)",m,n);
    }
  }

  PetscCall(SVMGetQP(svm,&qp));
  PetscCall(QPSetOperator(qp,A));

  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetOperator_Binary"
PetscErrorCode SVMGetOperator_Binary(SVM svm,Mat *A)
{
  QP  qp;

  PetscFunctionBegin;
  PetscCall(SVMGetQP(svm,&qp));
  PetscCall(QPGetOperator(qp,A));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetTrainingDataset"
PetscErrorCode SVMSetTrainingDataset_Binary(SVM svm,Mat Xt_training,Vec y_training)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscReal  max;
  PetscInt   lo,hi;
  PetscInt   m,n;

  Mat        G;
  Vec        tmp;

  PetscFunctionBegin;
  PetscCall(MatGetSize(Xt_training,&m,NULL));
  PetscCall(VecGetSize(y_training,&n));
  if (m != n) {
    SETERRQ(PetscObjectComm((PetscObject) Xt_training),PETSC_ERR_ARG_SIZ,"Dimensions are incompatible, X_training(%D,) != y_training(%D)",m,n);
  }

  PetscCall(SVMGetGramian(svm,&G));
  if (G) {
    PetscCall(MatGetSize(G,&n,NULL));
    if (m != n) {
      SETERRQ(PetscObjectComm((PetscObject) G),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, X_training(%D,) != G(%D,)",m,n);
    }
  }

  PetscCall(MatDestroy(&svm_binary->Xt_training));
  svm_binary->Xt_training = Xt_training;
  PetscCall(PetscObjectReference((PetscObject) Xt_training));

  PetscCall(VecDestroy(&svm_binary->y_training));
  svm_binary->y_training = y_training;
  PetscCall(PetscObjectReference((PetscObject) y_training));

  /* Determine index sets of positive and negative samples */
  PetscCall(VecGetOwnershipRange(y_training,&lo,&hi));
  PetscCall(VecMax(y_training,NULL,&max));
  PetscCall(VecDuplicate(y_training,&tmp));
  PetscCall(VecSet(tmp,max));

  /* Index set for positive samples */
  PetscCall(ISDestroy(&svm_binary->is_p));
  PetscCall(VecWhichEqual(y_training,tmp,&svm_binary->is_p));

  /* Index set for negative samples */
  PetscCall(ISDestroy(&svm_binary->is_n));
  PetscCall(ISComplement(svm_binary->is_p,lo,hi,&svm_binary->is_n));

  /* Free memory */
  PetscCall(VecDestroy(&tmp));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetTrainingDataset_Binary"
PetscErrorCode SVMGetTrainingDataset_Binary(SVM svm,Mat *Xt_training,Vec *y_training)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  if (Xt_training) {
    PetscValidPointer(Xt_training,2);
    *Xt_training = svm_binary->Xt_training;
  }
  if (y_training) {
    PetscValidPointer(y_training,3);
    *y_training = svm_binary->y_training;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetUp_Remapy_Binary_Private"
static PetscErrorCode SVMSetUp_Remapy_Binary_Private(SVM svm)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  Vec         y;
  PetscInt    i,n;

  PetscScalar min,max;
  const PetscScalar *y_arr;
  PetscScalar *y_inner_arr;

  PetscFunctionBegin;
  PetscCall(SVMGetTrainingDataset(svm,NULL,&y));
  PetscCall(VecMin(y,NULL,&min));
  PetscCall(VecMax(y,NULL,&max));

  if (min == -1.0 && max == 1.0) {
    PetscCall(VecDestroy(&svm_binary->y_inner));
    svm_binary->y_inner = y;
    PetscCall(PetscObjectReference((PetscObject) y));
  } else {
    PetscCall(VecGetLocalSize(y,&n));
    PetscCall(VecDuplicate(y,&svm_binary->y_inner));
    PetscCall(VecGetArrayRead(y,&y_arr));
    PetscCall(VecGetArray(svm_binary->y_inner,&y_inner_arr));
    for (i = 0; i < n; ++i) {
      if (y_arr[i]==min) {
        y_inner_arr[i] = -1.0;
      } else if (y_arr[i] == max) {
        y_inner_arr[i] = 1.0;
      }
    }
    PetscCall(VecRestoreArrayRead(y,&y_arr));
    PetscCall(VecRestoreArray(svm_binary->y_inner,&y_inner_arr));
  }

  svm_binary->y_map[0] = min;
  svm_binary->y_map[1] = max;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMCreateQPS_Binary_Private"
PetscErrorCode SVMCreateQPS_Binary_Private(SVM svm,QPS *qps)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscReal rtol, divtol, max_eig_tol;
  PetscInt max_it, max_eig_it;

  QPS      qps_inner,qps_smalxe_inner;
  PetscInt svm_mod;

  PetscFunctionBegin;
  rtol = 1e-1;
  divtol = 1e100;
  max_it = 10000;
  max_eig_it = 100;
  max_eig_tol = 1e-5;

  PetscCall(SVMGetMod(svm,&svm_mod));

  PetscCall(QPSDestroy(&svm_binary->qps));
  PetscCall(QPSCreate(PetscObjectComm((PetscObject) svm),&qps_inner));

  if (svm_mod == 1) {
    PetscCall(QPSSetType(qps_inner,QPSSMALXE));
    PetscCall(QPSSMALXEGetInnerQPS(qps_inner,&qps_smalxe_inner));
    PetscCall(QPSSetType(qps_smalxe_inner,QPSMPGP));
    PetscCall(QPSSMALXESetOperatorMaxEigenvalueTolerance(qps_inner,max_eig_tol));
    PetscCall(QPSSMALXESetOperatorMaxEigenvalueIterations(qps_inner,max_eig_it));
    PetscCall(QPSSetTolerances(qps_smalxe_inner,rtol,PETSC_DEFAULT,divtol,max_it));
    PetscCall(QPSSMALXESetM1Initial(qps_inner,1.0,QPS_ARG_MULTIPLE));
    PetscCall(QPSSMALXESetRhoInitial(qps_inner,1.0,QPS_ARG_MULTIPLE));
  } else {
    PetscCall(QPSSetType(qps_inner,QPSMPGP));
    PetscCall(QPSSetTolerances(qps_inner,rtol,PETSC_DEFAULT,divtol,max_it));
    PetscCall(QPSMPGPSetOperatorMaxEigenvalueTolerance(qps_inner,max_eig_tol));
    PetscCall(QPSMPGPSetOperatorMaxEigenvalueIterations(qps_inner,max_eig_it));
  }
  *qps = qps_inner;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetQPS_Binary"
PetscErrorCode SVMGetQPS_Binary(SVM svm,QPS *qps)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;
  QPS        qps_inner;

  PetscFunctionBegin;
  if (!svm_binary->qps) {
    PetscCall(SVMCreateQPS_Binary_Private(svm,&qps_inner));
    svm_binary->qps = qps_inner;
  }

  *qps = svm_binary->qps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetQPS_Binary"
PetscErrorCode SVMSetQPS_Binary(SVM svm,QPS qps)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  PetscCheckSameComm(svm,1,qps,2);

  PetscCall(QPSDestroy(&svm_binary->qps));
  svm_binary->qps = qps;
  PetscCall(PetscObjectReference((PetscObject) qps));

  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetQP_Binary"
PetscErrorCode SVMGetQP_Binary(SVM svm,QP *qp)
{
  QPS qps;

  PetscFunctionBegin;
  PetscCall(SVMGetQPS(svm,&qps));
  PetscCall(QPSGetQP(qps,qp));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMUpdateOperator_Binary_Private"
PetscErrorCode SVMUpdateOperator_Binary_Private(SVM svm)
{
  SVM_Binary  *svm_binary = (SVM_Binary *) svm->data;

  Vec         diag_p,diag_n;

  PetscInt    p;
  SVMLossType loss_type;

  PetscReal   C,Cn,Cp;

  PetscFunctionBegin;
  PetscCall(SVMGetLossType(svm,&loss_type));
  if (loss_type == SVM_L1) PetscFunctionReturn(0);

  /* Update regularization of Hessian */
  PetscCall(SVMGetPenaltyType(svm,&p));
  if (p == 1) {
    PetscCall(SVMGetC(svm,&C));

    PetscCall(MatScale(svm_binary->J,0.));
    PetscCall(MatShift(svm_binary->J,1. / C));
  } else {
    PetscCall(SVMGetCp(svm,&Cp));
    PetscCall(SVMGetCn(svm,&Cn));

    PetscCall(VecGetSubVector(svm_binary->diag,svm_binary->is_p,&diag_p));
    PetscCall(VecSet(diag_p,1. / Cp));
    PetscCall(VecRestoreSubVector(svm_binary->diag,svm_binary->is_p,&diag_p));

    PetscCall(VecGetSubVector(svm_binary->diag,svm_binary->is_n,&diag_n));
    PetscCall(VecSet(diag_n,1. / Cn));
    PetscCall(VecRestoreSubVector(svm_binary->diag,svm_binary->is_n,&diag_n));
  }
  PetscFunctionReturn(0);
}

/* TODO implement SVMUpdateInitialVector_Binary_Private */

#undef __FUNCT__
#define __FUNCT__ "SVMUpdate_Binary_Private"
PetscErrorCode SVMUpdate_Binary_Private(SVM svm)
{
  SVM_Binary  *svm_binary = (SVM_Binary *) svm->data;

  QP          qp;

  Vec         x_init,x_init_p,x_init_n;
  Vec         ub = NULL,ub_p,ub_n;

  PetscInt    p;
  SVMLossType loss_type;

  PetscReal   C,Cn,Cp;

  PetscFunctionBegin;
  PetscCall(SVMUpdateOperator_Binary_Private(svm));

  PetscCall(SVMGetPenaltyType(svm,&p));
  if (p == 1) {
    PetscCall(SVMGetC(svm,&C));
  } else {
    PetscCall(SVMGetCp(svm,&Cp));
    PetscCall(SVMGetCn(svm,&Cn));
  }

  /* Update initial guess */
  PetscCall(SVMGetQP(svm,&qp));
  PetscCall(QPGetSolutionVector(qp,&x_init));

  /* TODO SVMUpdateInitialVector_Binary_Private */
  if (svm->warm_start) {
    if (p == 1) {
      PetscCall(VecScale(x_init,1. / svm->C_old));
      PetscCall(VecScale(x_init,C));
    } else {
      PetscCall(VecGetSubVector(x_init,svm_binary->is_p,&x_init_p));
      PetscCall(VecScale(x_init_p,1. / svm->Cp_old));
      PetscCall(VecScale(x_init_p,Cp));
      PetscCall(VecRestoreSubVector(x_init,svm_binary->is_p,&x_init_p));

      PetscCall(VecGetSubVector(x_init,svm_binary->is_n,&x_init_n));
      PetscCall(VecScale(x_init_n,1. / svm->Cn_old));
      PetscCall(VecScale(x_init_n,Cn));
      PetscCall(VecRestoreSubVector(x_init,svm_binary->is_n,&x_init_n));
    }
  } else {
    if (p == 1) {
      PetscCall(VecSet(x_init,C - 100 * PETSC_MACHINE_EPSILON));
    } else {
      PetscCall(VecGetSubVector(x_init,svm_binary->is_p,&x_init_p));
      PetscCall(VecSet(x_init_p,Cp - 100 * PETSC_MACHINE_EPSILON));
      PetscCall(VecRestoreSubVector(x_init,svm_binary->is_p,&x_init_p));

      PetscCall(VecGetSubVector(x_init,svm_binary->is_n,&x_init_n));
      PetscCall(VecSet(x_init_n,Cn - 100 * PETSC_MACHINE_EPSILON));
      PetscCall(VecRestoreSubVector(x_init,svm_binary->is_n,&x_init_n));
    }
  }

  /* Update upper bound vector */
  PetscCall(SVMGetLossType(svm,&loss_type));
  if (loss_type == SVM_L2) PetscFunctionReturn(0);

  PetscCall(QPGetBox(qp,NULL,NULL,&ub));
  if (p == 1) {
    PetscCall(VecSet(ub,C));
  } else {
    PetscCall(VecGetSubVector(ub,svm_binary->is_p,&ub_p));
    PetscCall(VecSet(ub_p,Cp));
    PetscCall(VecRestoreSubVector(ub,svm_binary->is_p,&ub_p));

    PetscCall(VecGetSubVector(ub,svm_binary->is_n,&ub_n));
    PetscCall(VecSet(ub_n,Cn));
    PetscCall(VecRestoreSubVector(ub,svm_binary->is_n,&ub_n));
  }

  svm->setupcalled     = PETSC_TRUE;
  svm->posttraincalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMComputeOperator_Binary"
PetscErrorCode SVMComputeOperator_Binary(SVM svm,Mat *A)
{
  MPI_Comm    comm;
  SVM_Binary  *svm_binary = (SVM_Binary *) svm->data;

  Mat         G,H,HpJ;
  Mat         mats[2];
  Mat         Xt,X = NULL;
  PetscInt    m,n,M,N;

  Vec         y;
  Vec         diag,diag_p,diag_n;

  PetscInt    p;         /* penalty type */
  SVMLossType loss_type;

  PetscReal   C,Cp,Cn;

  PetscFunctionBegin;
  /* Check if operator is set */
  PetscCall(SVMGetOperator(svm,&H));
  if (H) PetscFunctionReturn(0);

  PetscCall(SVMGetPenaltyType(svm,&p));
  PetscCall(SVMGetLossType(svm,&loss_type));

  PetscCall(SVMGetGramian(svm,&G));
  /* Create Gramian matrix (X^t * X) implicitly if it is not set by user, i.e. G == NULL */
  if (!G) {
    /* TODO add option for computing Gramian explicitly */
    PetscCall(SVMGetTrainingDataset(svm,&Xt,NULL));
    PetscCall(PermonMatTranspose(Xt,MAT_TRANSPOSE_CHEAPEST,&X));
    PetscCall(MatCreateNormal(X,&G));  /* G = X^t * X */
  } else {
    PetscCall(PetscObjectReference((PetscObject) G));
  }

  /* Remap y to -1,1 values if needed */
  PetscCall(SVMSetUp_Remapy_Binary_Private(svm));
  y = svm_binary->y_inner;

  /* Create Hessian matrix */
  H = G;
  PetscCall(MatDiagonalScale(H,y,y)); /* H = diag(y) * G * diag(y) */

  /* Regularize Hessian in case of l2-loss SVM */
  if (loss_type == SVM_L2) {
    /* 1 / 2t = C / 2 => t = 1 / C */
    /* H = H + t * I */
    /* https://link.springer.com/article/10.1134/S1054661812010129 */
    /* http://www.lib.kobe-u.ac.jp/repository/90000225.pdf */
    PetscCall(PetscObjectGetComm((PetscObject) H,&comm));
    PetscCall(MatDestroy(&svm_binary->J));

    if (p == 1) { /* Penalty type 1 */
      PetscCall(SVMGetC(svm,&C));

      PetscCall(MatGetLocalSize(H,&m,&n));
      PetscCall(MatGetSize(H,&M,&N));

      PetscCall(MatCreateConstantDiagonal(comm,m,n,M,N,1. / C,&svm_binary->J));
      PetscCall(MatAssemblyBegin(svm_binary->J,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(svm_binary->J,MAT_FINAL_ASSEMBLY));
    } else { /* Penalty type 2 */
      PetscCall(SVMGetCp(svm,&Cp));
      PetscCall(SVMGetCn(svm,&Cn));

      PetscCall(VecDestroy(&svm_binary->diag));
      PetscCall(VecDuplicate(y,&svm_binary->diag));
      diag = svm_binary->diag;

      PetscCall(VecGetSubVector(diag,svm_binary->is_p,&diag_p));
      PetscCall(VecSet(diag_p,1. / Cp));
      PetscCall(VecRestoreSubVector(diag,svm_binary->is_p,&diag_p));

      PetscCall(VecGetSubVector(diag,svm_binary->is_n,&diag_n));
      PetscCall(VecSet(diag_n,1. / Cn));
      PetscCall(VecRestoreSubVector(diag,svm_binary->is_n,&diag_n));

      PetscCall(MatCreateDiag(diag,&svm_binary->J));
    }

    mats[0] = svm_binary->J;
    mats[1] = H;
    PetscCall(MatCreateSum(comm,2,mats,&HpJ)); /* H = H + J */
    PetscCall(MatDestroy(&H));

    H  = HpJ;
  }

  *A = H;
  /* Decreasing reference counts */
  PetscCall(MatDestroy(&X));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetUp_Binary"
PetscErrorCode SVMSetUp_Binary(SVM svm)
{
  SVM_Binary  *svm_binary = (SVM_Binary *) svm->data;

  QPS         qps;
  QP          qp;

  Mat         Xt;
  Vec         y;

  Mat         H;
  Vec         e;
  Vec         x_init,x_init_p,x_init_n;
  Mat         Be = NULL;
  PetscReal   norm;
  Vec         lb;
  Vec         ub = NULL,ub_p,ub_n;

  PetscInt    p;        /* penalty type */
  PetscInt    svm_mod;
  SVMLossType loss_type;

  PetscReal   C,Cp,Cn;

  PetscInt    i;

  /* monitors */
  PetscBool   svm_monitor_set;
  void        *mctx;  /* monitor context */

  PetscFunctionBegin;
  if (svm->setupcalled) PetscFunctionReturn(0);

  if (svm->posttraincalled) {
    PetscCall(SVMUpdate_Binary_Private(svm));
    PetscFunctionReturn(0);
  }

  /* TODO generalize implementation of hyper parameter optimization */
  if (svm->hyperoptset) {
    PetscCall(SVMGridSearch(svm));
  }

  PetscCall(SVMComputeOperator(svm,&H)); /* compute Hessian of QP problem */
  PetscCall(SVMSetOperator(svm,H));

  PetscCall(SVMGetTrainingDataset(svm,&Xt,&y)); /* get samples and label vector for latter computing */

  PetscCall(SVMGetQPS(svm,&qps));
  PetscCall(QPSGetQP(qps,&qp));

  /* Set RHS */
  PetscCall(VecDuplicate(y,&e));  /* creating vector e same size and type as y_training */
  PetscCall(VecSet(e,1.));
  PetscCall(QPSetRhs(qp,e));      /* set linear term of QP problem */

  /* Set equality constraint for SVM mod 1 */
  PetscCall(SVMGetMod(svm,&svm_mod));
  if (svm_mod == 1) {
    PetscCall(MatCreateOneRow(y,&Be));   /* Be = y^t */
    PetscCall(VecNorm(y,NORM_2,&norm));
    PetscCall(MatScale(Be,1. / norm));    /* ||Be|| = 1 */
    PetscCall(QPSetEq(qp,Be,NULL));
  }

  /* Create box constraint */
  PetscCall(VecDuplicate(y,&lb));  /* create lower bound constraint */
  PetscCall(VecSet(lb,0.));

  PetscCall(SVMGetPenaltyType(svm,&p));
  if (p == 1) {
    PetscCall(SVMGetC(svm,&C));
  } else {
    PetscCall(SVMGetCp(svm,&Cp));
    PetscCall(SVMGetCn(svm,&Cn));
  }

  PetscCall(SVMGetLossType(svm,&loss_type));
  if (loss_type == SVM_L1) {
    PetscCall(VecDuplicate(lb,&ub));

    if (p == 1) {
      PetscCall(VecSet(ub,C));
    } else {
      /* Set upper bound constrain related to positive samples */
      PetscCall(VecGetSubVector(ub,svm_binary->is_p,&ub_p));
      PetscCall(VecSet(ub_p,Cp));
      PetscCall(VecRestoreSubVector(ub,svm_binary->is_p,&ub_p));

      /* Set upper bound constrain related to negative samples */
      PetscCall(VecGetSubVector(ub,svm_binary->is_n,&ub_n));
      PetscCall(VecSet(ub_n,Cn));
      PetscCall(VecRestoreSubVector(ub,svm_binary->is_n,&ub_n));
    }
  }

  PetscCall(QPSetBox(qp,NULL,lb,ub));

  /* TODO create public method for setting initial vector */
  /* Set initial guess */
  PetscCall(VecDuplicate(lb,&x_init));
  if (p == 1) {
    PetscCall(VecSet(x_init,C - 100 * PETSC_MACHINE_EPSILON));
  } else {
    PetscCall(VecGetSubVector(x_init,svm_binary->is_p,&x_init_p));
    PetscCall(VecSet(x_init_p,Cp - 100 * PETSC_MACHINE_EPSILON));
    PetscCall(VecRestoreSubVector(x_init,svm_binary->is_p,&x_init_p));

    PetscCall(VecGetSubVector(x_init,svm_binary->is_n,&x_init_n));
    PetscCall(VecSet(x_init_n,Cn - 100 * PETSC_MACHINE_EPSILON));
    PetscCall(VecRestoreSubVector(x_init,svm_binary->is_n,&x_init_n));
  }
  PetscCall(QPSetInitialVector(qp,x_init));
  PetscCall(VecDestroy(&x_init));

  /* TODO create public method for setting monitors */
  /* Set monitors */
  PetscCall(PetscOptionsHasName(NULL,((PetscObject) svm)->prefix,"-svm_monitor",&svm_monitor_set));
  if (svm_monitor_set) {
    PetscCall(SVMMonitorCreateCtx_Binary(&mctx,svm));
    if (svm_mod == 1) {
      QPS qps_inner;
      PetscCall(QPSSMALXEGetInnerQPS(qps,&qps_inner));
      PetscCall(QPSMonitorSet(qps_inner,SVMMonitorDefault_Binary,mctx,SVMMonitorDestroyCtx_Binary));
    } else {
      PetscCall(QPSMonitorSet(qps,SVMMonitorDefault_Binary,mctx,SVMMonitorDestroyCtx_Binary));
    }
  }
  PetscCall(PetscOptionsHasName(NULL,((PetscObject) svm)->prefix,"-svm_monitor_obj_funcs",&svm_monitor_set));
  if (svm_monitor_set) {
    PetscCall(SVMMonitorCreateCtx_Binary(&mctx,svm));
    if (svm_mod == 1) {
      QPS qps_inner;
      PetscCall(QPSSMALXEGetInnerQPS(qps,&qps_inner));
      PetscCall(QPSMonitorSet(qps_inner,SVMMonitorObjFuncs_Binary,mctx,SVMMonitorDestroyCtx_Binary));
    } else {
      PetscCall(QPSMonitorSet(qps,SVMMonitorObjFuncs_Binary,mctx,SVMMonitorDestroyCtx_Binary));
    }
  }
  PetscCall(PetscOptionsHasName(NULL,((PetscObject) svm)->prefix,"-svm_monitor_training_scores",&svm_monitor_set));
  if (svm_monitor_set) {
    PetscCall(SVMMonitorCreateCtx_Binary(&mctx,svm));
    if (svm_mod == 1) {
      QPS qps_inner;
      PetscCall(QPSSMALXEGetInnerQPS(qps,&qps_inner));
      PetscCall(QPSMonitorSet(qps_inner,SVMMonitorTrainingScores_Binary,mctx,SVMMonitorDestroyCtx_Binary));
    } else {
      PetscCall(QPSMonitorSet(qps,SVMMonitorTrainingScores_Binary,mctx,SVMMonitorDestroyCtx_Binary));
    }
  }
  PetscCall(PetscOptionsHasName(NULL,((PetscObject) svm)->prefix,"-svm_monitor_scores",&svm_monitor_set));
  if (svm_monitor_set) {
    Mat Xt_test;
    Vec y_test;

    PetscCall(SVMGetTestDataset(svm,&Xt_test,&y_test));
    if (!Xt_test && !y_test) {
      SETERRQ(((PetscObject) svm)->comm,PETSC_ERR_ARG_NULL,"Test dataset must be set for using -svm_monitor_scores.");
    }
    PetscCall(SVMMonitorCreateCtx_Binary(&mctx,svm));
    if (svm_mod == 1) {
      QPS qps_inner;
      PetscCall(QPSSMALXEGetInnerQPS(qps,&qps_inner));
      PetscCall(QPSMonitorSet(qps_inner,SVMMonitorScores_Binary,mctx,SVMMonitorDestroyCtx_Binary));
    } else {
      PetscCall(QPSMonitorSet(qps,SVMMonitorScores_Binary,mctx,SVMMonitorDestroyCtx_Binary));
    }
  }

  /* Set QPS */
  if (svm->setfromoptionscalled) {
    PetscCall(QPTFromOptions(qp));
    PetscCall(QPSSetFromOptions(qps));
  }
  PetscCall(QPSSetUp(qps));

  /* Create work vectors */
  for (i = 0; i < 3; ++i) {
    PetscCall(VecDestroy(&svm_binary->work[i]));
  }
  PetscCall(MatCreateVecs(Xt,NULL,&svm_binary->work[0])); /* TODO use duplicated vector y instead of creating vec? */
  PetscCall(VecDuplicate(svm_binary->work[0],&svm_binary->work[1]));
  PetscCall(VecDuplicate(svm_binary->work[0],&svm_binary->work[2]));

  /* Decreasing reference counts using destroy methods */
  PetscCall(MatDestroy(&H));
  PetscCall(MatDestroy(&Be));
  PetscCall(VecDestroy(&e));
  PetscCall(VecDestroy(&lb));
  PetscCall(VecDestroy(&ub));

  svm->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetOptionsPrefix_Binary"
PetscErrorCode SVMSetOptionsPrefix_Binary(SVM svm,const char prefix[])
{
  QPS qps;

  PetscFunctionBegin;
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) svm,prefix));
  PetscCall(SVMGetQPS(svm,&qps));
  PetscCall(QPSSetOptionsPrefix(qps,prefix));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMAppendOptionsPrefix_Binary"
PetscErrorCode SVMAppendOptionsPrefix_Binary(SVM svm,const char prefix[])
{
  QPS qps;

  PetscFunctionBegin;
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject) svm,prefix));
  PetscCall(SVMGetQPS(svm,&qps));
  PetscCall(QPSAppendOptionsPrefix(qps,prefix));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetOptionsPrefix_Binary"
PetscErrorCode SVMGetOptionsPrefix_Binary(SVM svm,const char *prefix[])
{

  PetscFunctionBegin;
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) svm,prefix));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMTrain_Binary"
PetscErrorCode SVMTrain_Binary(SVM svm)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  PetscCall(SVMSetUp(svm));
  PetscCall(QPSSetAutoPostSolve(svm_binary->qps,PETSC_FALSE));
  PetscCall(QPSSolve(svm_binary->qps));
  if (svm->autoposttrain) {
    PetscCall(SVMPostTrain(svm));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMReconstructHyperplane_Binary"
PetscErrorCode SVMReconstructHyperplane_Binary(SVM svm)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  QP        qp;
  Vec       lb,ub;

  Mat       Xt;
  Vec       x,y,yx,y_sv,t;
  Vec       Xtw_sv;

  Vec       w_inner;
  PetscReal b_inner;

  PetscInt  svm_mod;

  PetscFunctionBegin;
  PetscCall(SVMGetTrainingDataset(svm,&Xt,NULL));
  y = svm_binary->y_inner;

  /* Reconstruction of hyperplane normal */
  PetscCall(SVMGetQP(svm,&qp));
  PetscCall(QPGetSolutionVector(qp,&x));
  PetscCall(VecDuplicate(x,&yx));

  PetscCall(VecPointwiseMult(yx,y,x)); /* yx = y.*x */
  PetscCall(MatCreateVecs(Xt,&w_inner,NULL));
  PetscCall(MatMultTranspose(Xt,yx,w_inner)); /* w = (X^t)^t * yx = X * yx */

  /* Reconstruction of the hyperplane bias */
  PetscCall(SVMGetMod(svm,&svm_mod));
  if (svm_mod == 1) {
    PetscCall(QPGetBox(qp,NULL,&lb,&ub));

    PetscCall(ISDestroy(&svm_binary->is_sv));
    if (svm->loss_type == SVM_L1) {
      PetscCall(VecWhichBetween(lb,x,ub,&svm_binary->is_sv));
    } else {
      PetscCall(VecWhichGreaterThan(x,lb,&svm_binary->is_sv));
    }
    PetscCall(ISGetSize(svm_binary->is_sv,&svm_binary->nsv));

    PetscCall(MatMult(Xt,w_inner,svm_binary->work[2]));

    PetscCall(VecGetSubVector(y,svm_binary->is_sv,&y_sv));     /* y_sv = y(is_sv) */
    PetscCall(VecGetSubVector(svm_binary->work[2],svm_binary->is_sv,&Xtw_sv)); /* Xtw_sv = Xt(is_sv) */
    PetscCall(VecDuplicate(y_sv,&t));
    PetscCall(VecWAXPY(t,-1.,Xtw_sv,y_sv));
    PetscCall(VecRestoreSubVector(y,svm_binary->is_sv,&y_sv));
    PetscCall(VecRestoreSubVector(svm_binary->work[2],svm_binary->is_sv,&Xtw_sv));
    PetscCall(VecSum(t,&b_inner));

    b_inner /= svm_binary->nsv;

    PetscCall(VecDestroy(&t));
  } else {
    b_inner = 0.;
  }

  PetscCall(SVMSetSeparatingHyperplane(svm,w_inner,b_inner));

  PetscCall(VecDestroy(&w_inner));
  PetscCall(VecDestroy(&yx));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetSeparatingHyperplane_Binary"
PetscErrorCode SVMSetSeparatingHyperplane_Binary(SVM svm,Vec w,PetscReal b)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  PetscCheckSameComm(svm,1,w,2);

  PetscCall(VecDestroy(&svm_binary->w));
  svm_binary->w = w;
  svm_binary->b = b;
  PetscCall(PetscObjectReference((PetscObject) w));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetSeparatingHyperplane_Binary"
PetscErrorCode SVMGetSeparatingHyperplane_Binary(SVM svm,Vec *w,PetscReal *b)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  if (w) {
    PetscValidPointer(w,2);
    *w = svm_binary->w;
  }
  if (b) {
    PetscValidRealPointer(b,3);
    *b = svm_binary->b;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetBias_Binary"
PetscErrorCode SVMSetBias_Binary(SVM svm,PetscReal b)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveReal(svm,b,2);

  if (svm_binary->b != b) {
    svm_binary->b = b;
    svm->setupcalled = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetBias_Binary"
PetscErrorCode SVMGetBias_Binary(SVM svm,PetscReal *b)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  *b = svm_binary->b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMComputeModelParams_Binary"
PetscErrorCode SVMComputeModelParams_Binary(SVM svm)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  QP          qp;
  Vec         x,lb,ub;
  Vec         w;
  PetscReal   max;

  SVMLossType loss_type;
  PetscInt    svm_mod;

  PetscFunctionBegin;
  PetscCall(SVMGetQP(svm,&qp));
  PetscCall(SVMGetLossType(svm,&loss_type));
  PetscCall(SVMGetMod(svm,&svm_mod));

  PetscCall(SVMGetSeparatingHyperplane(svm,&w,NULL));
  if (!w) {
    PetscCall(SVMReconstructHyperplane(svm));
  }

  PetscCall(VecNorm(w,NORM_2,&svm_binary->norm_w));
  svm_binary->margin = 2. / svm_binary->norm_w;

  PetscCall(QPGetSolutionVector(qp,&x));
  PetscCall(QPGetBox(qp,NULL,&lb,&ub));

  PetscCall(VecMax(x,NULL,&max));
  PetscCall(VecChop(x,svm_binary->chop_tol*max));

  if (svm_mod == 2) {
    PetscCall(ISDestroy(&svm_binary->is_sv));
    if (loss_type == SVM_L1) {
      PetscCall(VecWhichBetween(lb, x, ub, &svm_binary->is_sv));
    } else {
      PetscCall(VecWhichGreaterThan(x, lb, &svm_binary->is_sv));
    }
    PetscCall(ISGetSize(svm_binary->is_sv, &svm_binary->nsv));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMComputeHingeLoss_Binary"
PetscErrorCode SVMComputeHingeLoss_Binary(SVM svm)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  Mat         Xt;
  Vec         y;

  Vec         w;
  PetscScalar b;

  PetscInt    p;        /* penalty type */
  PetscInt    svm_mod;
  Vec         work_p,work_n;
  SVMLossType loss_type;

  PetscFunctionBegin;
  PetscCall(SVMGetMod(svm,&svm_mod));
  PetscCall(SVMGetLossType(svm,&loss_type));
  PetscCall(SVMGetPenaltyType(svm,&p));
  PetscCall(SVMGetTrainingDataset(svm,&Xt,&y));

  PetscCall(SVMGetSeparatingHyperplane(svm,&w,NULL));
  if (!w) {
    PetscCall(SVMReconstructHyperplane(svm));
    PetscCall(SVMGetSeparatingHyperplane(svm,&w,NULL));
  }

  if (svm_mod == 1) {
    PetscCall(SVMGetSeparatingHyperplane(svm,NULL,&b));
    PetscCall(VecSet(svm_binary->work[1],-b));
    PetscCall(VecCopy(svm_binary->work[2],svm_binary->work[0]));
    PetscCall(VecAYPX(svm_binary->work[0],1.,svm_binary->work[1])); /* xi = Xtw - b */
  } else {
    PetscCall(MatMult(Xt,w,svm_binary->work[0])); /* xi = Xtw */
  }

  PetscCall(VecPointwiseMult(svm_binary->work[0],y,svm_binary->work[0])); /* xi = y .* xi */

  PetscCall(VecSet(svm_binary->work[1],1.));
  PetscCall(VecAYPX(svm_binary->work[0],-1.,svm_binary->work[1]));       /* xi = 1 - xi */
  PetscCall(VecSet(svm_binary->work[1],0.));
  PetscCall(VecPointwiseMax(svm_binary->work[0],svm_binary->work[1],svm_binary->work[0])); /* max(0,xi) */

  if (loss_type == SVM_L1) {
    if (p == 1) {
      PetscCall(VecSum(svm_binary->work[0],&svm_binary->hinge_loss)); /* hinge_loss = sum(xi) */
    } else {
      PetscCall(VecGetSubVector(svm_binary->work[0],svm_binary->is_p,&work_p));
      PetscCall(VecSum(work_p,&svm_binary->hinge_loss_p)); /* hinge_loss_p = sum(xi_p) */
      PetscCall(VecRestoreSubVector(svm_binary->work[0],svm_binary->is_p,&work_p));

      PetscCall(VecGetSubVector(svm_binary->work[0],svm_binary->is_n,&work_n));
      PetscCall(VecSum(work_n,&svm_binary->hinge_loss_n)); /* hinge_loss_n = sum(xi_n) */
      PetscCall(VecRestoreSubVector(svm_binary->work[0],svm_binary->is_n,&work_n));
    }
  } else {
    if (p == 1) {
      PetscCall(VecDot(svm_binary->work[0],svm_binary->work[0],&svm_binary->hinge_loss)); /* hinge_loss = sum(xi^2) */
    } else {
      PetscCall(VecGetSubVector(svm_binary->work[0],svm_binary->is_p,&work_p));
      PetscCall(VecDot(work_p,work_p,&svm_binary->hinge_loss_p)); /* hinge_loss_p = sum(xi_p^2) */
      PetscCall(VecRestoreSubVector(svm_binary->work[0],svm_binary->is_p,&work_p));

      PetscCall(VecGetSubVector(svm_binary->work[0],svm_binary->is_n,&work_n));
      PetscCall(VecDot(work_n,work_n,&svm_binary->hinge_loss_n)); /* hinge_loss_n = sum(xi_n^2) */
      PetscCall(VecRestoreSubVector(svm_binary->work[0],svm_binary->is_n,&work_n));
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMComputeObjFuncValues_Binary_Private"
PetscErrorCode SVMComputeObjFuncValues_Binary_Private(SVM svm)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscInt    p;        /* penalty type */
  PetscReal   C,Cp,Cn;

  Vec         w;
  SVMLossType loss_type;

  PetscReal   tmp;

  QP          qp;
  Vec         x;

  PetscFunctionBegin;
  PetscCall(SVMGetLossType(svm,&loss_type));
  PetscCall(SVMGetPenaltyType(svm,&p));

  PetscCall(SVMComputeHingeLoss(svm));

  /* Compute value of primal objective function */
  PetscCall(SVMGetSeparatingHyperplane(svm,&w,NULL));

  PetscCall(VecDot(w,w,&svm_binary->primalObj));
  svm_binary->primalObj *= 0.5;
  if (p == 1) {
    PetscCall(SVMGetC(svm,&C));

    tmp = C * svm_binary->hinge_loss;
  } else {
    PetscCall(SVMGetCp(svm,&Cp));
    PetscCall(SVMGetCn(svm,&Cn));

    tmp = Cp * svm_binary->hinge_loss_p;
    tmp += Cn * svm_binary->hinge_loss_n;
  }

  if (loss_type == SVM_L1) {
    svm_binary->primalObj += tmp;
  } else {
    svm_binary->primalObj += tmp / 2.;
  }

  /* Compute value of dual objective function */
  PetscCall(SVMGetQP(svm,&qp));
  PetscCall(QPGetSolutionVector(qp,&x));
  PetscCall(QPComputeObjective(qp,x,&svm_binary->dualObj));
  svm_binary->dualObj *= -1.;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMPostTrain_Binary"
PetscErrorCode SVMPostTrain_Binary(SVM svm)
{
  QPS       qps;

  PetscFunctionBegin;
  if (svm->posttraincalled) PetscFunctionReturn(0);

  PetscCall(SVMGetQPS(svm,&qps));
  PetscCall(QPSPostSolve(qps) );

  PetscCall(SVMReconstructHyperplane(svm));
  PetscCall(SVMComputeObjFuncValues_Binary_Private(svm));
  PetscCall(SVMComputeModelParams(svm));

  svm->posttraincalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetFromOptions_Binary"
PetscErrorCode SVMSetFromOptions_Binary(PetscOptionItems *PetscOptionsObject,SVM svm)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscReal b;
  PetscBool flg;

  PetscFunctionBegin;
  PetscObjectOptionsBegin((PetscObject) svm);
  PetscCall(PetscOptionsReal("-svm_bias","","SVMSetBias",svm_binary->b,&b,&flg));
  if (flg) {
    PetscCall(SVMSetBias(svm,b));
  }
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetHyperplaneSubNormal_Binary_Private"
PetscErrorCode SVMGetHyperplaneSubNormal_Binary_Private(SVM svm,Mat Xt_predict,IS *is_sub,Vec *w_sub)
{
  MPI_Comm  comm;

  Vec       w,w_sub_inner;
  IS        is,is_last,is_tmp;

  PetscInt  mod;

  PetscInt  n,lo,hi;
  PetscInt  N,N_predict;

  PetscFunctionBegin;
  PetscCall(SVMGetMod(svm,&mod));

  PetscCall(MatGetSize(Xt_predict,NULL,&N_predict));
  PetscCall(MatGetOwnershipRangeColumn(Xt_predict,&lo,&hi));

  PetscCall(SVMGetSeparatingHyperplane(svm,&w,NULL));
  PetscCall(PetscObjectGetComm((PetscObject) w,&comm));

  n = 0;
  N_predict += (1 - mod);
  if (hi < N_predict) {
    n = hi - lo;
  } else if (lo < N_predict) {
    n = N_predict - lo;
  }
  PetscCall(ISCreateStride(comm,n,lo,1,&is));

  if (mod == 2) {
    PetscCall(VecGetSize(w,&N));
    PetscCall(VecGetOwnershipRange(w,NULL,&hi));

    PetscCall(ISCreateStride(comm,(hi == N) ? 1 : 0,hi - 1,1,&is_last));
    /* Concatenate is and is_last */
    PetscCall(ISExpand(is,is_last,&is_tmp));
    /* Free memory */
    PetscCall(ISDestroy(&is));
    PetscCall(ISDestroy(&is_last));

    is = is_tmp;
  }

  PetscCall(VecGetSubVector(w,is,&w_sub_inner));

  *w_sub = w_sub_inner;
  *is_sub = is;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMCreateSubPredictDataset_Binary_Private"
PetscErrorCode SVMCreateSubPredictDataset_Binary_Private(SVM svm,Mat Xt_predict,Mat *Xt_out)
{
  MPI_Comm  comm;

  Vec       w;
  PetscInt  lo,hi;
  PetscInt  N,tmp;

  PetscInt  mod;

  Mat       Xt_sub;
  IS        is_cols,is_rows;

  PetscFunctionBegin;
  tmp = 0;

  PetscCall(SVMGetMod(svm,&mod));

  PetscCall(SVMGetSeparatingHyperplane(svm,&w,NULL));

  PetscCall(VecGetOwnershipRange(w,&lo,&hi));
  PetscCall(PetscObjectGetComm((PetscObject) Xt_predict,&comm));

  PetscCall(MatGetOwnershipIS(Xt_predict,&is_rows,NULL));
  if (mod == 2) {
    PetscCall(VecGetSize(w,&N));
    tmp = (hi == N) ? 1 : 0;
  }
  PetscCall(ISCreateStride(comm,hi - lo - tmp,lo,1,&is_cols));

  PetscCall(MatCreateSubMatrix(Xt_predict,is_rows,is_cols,MAT_INITIAL_MATRIX,&Xt_sub));

  *Xt_out = Xt_sub;

  /* Free memory */
  PetscCall(ISDestroy(&is_rows));
  PetscCall(ISDestroy(&is_cols));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMPredict_Binary"
PetscErrorCode SVMPredict_Binary(SVM svm,Mat Xt_pred,Vec *y_out)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  /* Hyperplane */
  Vec       w,w_tmp;
  PetscReal b;

  Mat       Xt_training,Xt_sub;
  PetscInt  N_training,N_predict;
  PetscInt  lo,hi;

  Vec       Xtw,y,sub_y;
  Vec       o;
  IS        is_p,is_n,is_w;

  PetscFunctionBegin;
  if (!svm->posttraincalled) {
    PetscCall(SVMPostTrain(svm));
  }
  PetscCall(SVMGetSeparatingHyperplane(svm,NULL,&b));
  PetscCall(SVMGetTrainingDataset(svm,&Xt_training,NULL));

  PetscCall(MatGetSize(Xt_training,NULL,&N_training));
  PetscCall(MatGetSize(Xt_pred,NULL,&N_predict));
  /* Check number of features */
  if (N_training > N_predict) {
    PetscCall(SVMGetHyperplaneSubNormal_Binary_Private(svm,Xt_pred,&is_w,&w));
  } else {
    PetscCall(SVMGetSeparatingHyperplane(svm,&w,NULL));
    if (N_training < N_predict) {
      PetscCall(SVMCreateSubPredictDataset_Binary_Private(svm,Xt_pred,&Xt_sub));
      Xt_pred = Xt_sub;
    }
  }

  /* Predict labels of unseen samples */
  PetscCall(MatCreateVecs(Xt_pred,NULL,&Xtw));
  PetscCall(VecGetOwnershipRange(Xtw,&lo,&hi));
  PetscCall(VecDuplicate(Xtw,&y));
  PetscCall(VecDuplicate(Xtw,&o));
  PetscCall(VecSet(o,0));

  PetscCall(MatMult(Xt_pred,w,Xtw));
  PetscCall(VecShift(Xtw,b)); /* shifting is not performed in case of b = 0 (inner implementation) */

  PetscCall(VecWhichGreaterThan(Xtw,o,&is_p));
  PetscCall(ISComplement(is_p,lo,hi,&is_n));

  PetscCall(VecGetSubVector(y,is_n,&sub_y));
  PetscCall(VecSet(sub_y,svm_binary->y_map[0]));
  PetscCall(VecRestoreSubVector(y,is_n,&sub_y));

  PetscCall(VecGetSubVector(y,is_p,&sub_y));
  PetscCall(VecSet(sub_y,svm_binary->y_map[1]));
  PetscCall(VecRestoreSubVector(y,is_p,&sub_y));

  *y_out = y;

  /* Free memory */
  PetscCall(VecDestroy(&Xtw));
  PetscCall(VecDestroy(&o));
  PetscCall(ISDestroy(&is_n));
  PetscCall(ISDestroy(&is_p));
  if (N_training > N_predict) {
    PetscCall(SVMGetSeparatingHyperplane(svm,&w_tmp,NULL));
    PetscCall(VecRestoreSubVector(w_tmp,is_w,&w));
    PetscCall(ISDestroy(&is_w));
  } else if (N_training < N_predict) {
    PetscCall(MatDestroy(&Xt_pred));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMComputeModelScores_Binary"
PetscErrorCode SVMComputeModelScores_Binary(SVM svm,Vec y_pred,Vec y_known)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  Vec       label,y_pred_sub,y_known_sub;
  IS        is_label,is_eq;

  PetscInt  TP,FP,TN,FN,N;
  /* AUC ROC */
  PetscReal specifity;
  PetscReal TPR,FPR;
  PetscReal x[3],y[3],dx;
  PetscInt  i;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(y_known,&label));

  /* TN and FN samples */
  PetscCall(VecSet(label,svm_binary->y_map[0]));
  PetscCall(VecWhichEqual(y_known,label,&is_label));
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
  PetscCall(VecSet(label,svm_binary->y_map[1]));
  PetscCall(VecWhichEqual(y_known,label,&is_label));
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

  /* confusion matrix */
  svm_binary->confusion_matrix[0] = TP;
  svm_binary->confusion_matrix[1] = FP;
  svm_binary->confusion_matrix[2] = FN;
  svm_binary->confusion_matrix[3] = TN;

  /* performance scores of model */
  svm_binary->model_scores[0] = (PetscReal) (TP + TN) / (PetscReal) (TP + TN + FP + FN); /* accuracy */
  svm_binary->model_scores[1] = (PetscReal) TP / (PetscReal) (TP + FP); /* precision */
  svm_binary->model_scores[2] = (PetscReal) TP / (TP + FN); /* sensitivity */
  /* F1 */
  svm_binary->model_scores[3] = 2. * (svm_binary->model_scores[1] * svm_binary->model_scores[2]) / (svm_binary->model_scores[1] + svm_binary->model_scores[2]);
  /* Matthews correlation coefficient */
  svm_binary->model_scores[4] = (PetscReal) (TP * TN - FP * FN);
  svm_binary->model_scores[4] /= (PetscSqrtReal(TP + FP) * PetscSqrtReal(TP + FN) * PetscSqrtReal(TN + FP) * PetscSqrtReal(TN + FN) );
  /* Area Under Curve (AUC) Receiver Operating Characteristics (ROC) */
  TPR = svm_binary->model_scores[2];
  specifity = (PetscReal) TN / (PetscReal) (TN + FP);
  FPR = 1 - specifity;
  /* Area under curve (trapezoidal rule) */
  x[0] = 0; x[1] = FPR; x[2] = 1;
  y[0] = 0; y[1] = TPR; y[2] = 1;

  svm_binary->model_scores[5] = 0.;
  for (i = 0; i < 2; ++i) {
    dx = x[i+1] - x[i];
    svm_binary->model_scores[5] += ((y[i] + y[i+1]) / 2.) * dx;
  }

  /* Gini coefficient */
  svm_binary->model_scores[6] = 2 * svm_binary->model_scores[5] - 1;
  PetscCall(VecDestroy(&label));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMTest_Binary"
PetscErrorCode SVMTest_Binary(SVM svm)
{
  Mat Xt_test;
  Vec y_known;
  Vec y_pred;

  PetscFunctionBegin;
  PetscCall(SVMGetTestDataset(svm,&Xt_test,&y_known));
  PetscCall(SVMPredict(svm,Xt_test,&y_pred));

  /* Evaluation of model performance scores */
  PetscCall(SVMComputeModelScores(svm,y_pred,y_known));
  PetscCall(VecDestroy(&y_pred));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetModelScore_Binary"
PetscErrorCode SVMGetModelScore_Binary(SVM svm,ModelScore score_type,PetscReal *s)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  PetscValidRealPointer(s,3);

  *s = svm_binary->model_scores[score_type];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMInitGridSearch_Binary_Private"
PetscErrorCode SVMInitGridSearch_Binary_Private(SVM svm,PetscInt *n_out,PetscReal *grid_out[])
{
  PetscInt  penalty_type;

  PetscReal base_1,start_1,end_1,step_1;
  PetscReal base_2,start_2,end_2,step_2;
  PetscReal tmp;

  PetscReal *grid;
  PetscInt  n,n_1,n_2;

  PetscInt  i,j,p;

  PetscFunctionBegin;
  PetscCall(SVMGetPenaltyType(svm,&penalty_type));
  if (penalty_type == 1) {
    PetscCall(SVMGridSearchGetBaseLogC(svm,&base_1));
    PetscCall(SVMGridSearchGetStrideLogC(svm,&start_1,&end_1,&step_1));

    n = PetscAbs(end_1 - start_1) / PetscAbs(step_1) + 1;
    PetscCall(PetscMalloc1(n,&grid));

    for (i = 0; i < n; ++i) grid[i] = PetscPowReal(base_1,start_1 + i * step_1);
  } else {
    PetscCall(SVMGridSearchGetPositiveBaseLogC(svm,&base_1));
    PetscCall(SVMGridSearchGetPositiveStrideLogC(svm,&start_1,&end_1,&step_1));
    PetscCall(SVMGridSearchGetNegativeBaseLogC(svm,&base_2));
    PetscCall(SVMGridSearchGetNegativeStrideLogC(svm,&start_2,&end_2,&step_2));

    n_1 = PetscAbs(end_1 - start_1) / PetscAbs(step_1) + 1;
    n_2 = PetscAbs(end_2 - start_2) / PetscAbs(step_2) + 1;
    n = 2 * n_1 * n_2;
    PetscCall(PetscMalloc1(n,&grid));

    p = -1;
    for (i = 0; i < n_1; ++i) {
      grid[++p] = PetscPowReal(base_1,start_1 + i * step_1);
      tmp = grid[p];
      grid[++p] = PetscPowReal(base_2,start_2);
      for (j = 1; j < n_2; ++j) {
        grid[++p] = tmp;
        grid[++p] = PetscPowReal(base_2,start_2 + j * step_2);
      }
    }
  }

  *n_out = n;
  *grid_out = grid;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearch_Binary"
PetscErrorCode SVMGridSearch_Binary(SVM svm)
{
  PetscReal *grid;
  PetscReal *scores,score_best;
  PetscInt  i,n,m,p,s;

  PetscFunctionBegin;
  PetscCall(SVMGetPenaltyType(svm,&m));

  /* Initialize grid */
  PetscCall(SVMInitGridSearch_Binary_Private(svm,&n,&grid));
  /* Perform cross-validation */
  s = n / m;
  PetscCall(PetscCalloc1(s,&scores));
  PetscCall(SVMCrossValidation(svm,grid,n,scores));

  /* Find best score */
  score_best = -1.;
  for (i = 0; i < s; ++i) {
    if (scores[i] > score_best) {
      p = i;
      score_best = scores[i];
    }
  }

  if (m == 1) {
    PetscInfo(svm,"selected best C=%.4f (score=%f)\n",grid[p],score_best);
  } else {
    PetscInfo(svm,"selected best C+=%.4f, C-=%.4f (score=%f)\n",grid[p * m],grid[p * m + 1],score_best);
  }

  PetscCall(SVMSetPenalty(svm,m,&grid[p * m]));

  PetscCall(PetscFree(grid));
  PetscCall(PetscFree(scores));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMLoadGramian_Binary"
PetscErrorCode SVMLoadGramian_Binary(SVM svm,PetscViewer v)
{
  Mat G;

  PetscFunctionBegin;
  /* Create matrix */
  PetscCall(MatCreate(PetscObjectComm((PetscObject) svm),&G));
  PetscCall(MatSetType(G,MATDENSE));
  PetscCall(PetscObjectSetName((PetscObject) G,"G"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) G,"G_"));
  PetscCall(MatSetFromOptions(G));

  PetscCall(MatLoad(G,v));
  PetscCall(SVMSetGramian(svm,G));

  /* Free memory */
  PetscCall(MatDestroy(&G));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMViewGramian_Binary"
PetscErrorCode SVMViewGramian_Binary(SVM svm,PetscViewer v)
{
  const char *type_name = NULL;

  Mat        G;
  PetscInt   M,N;

  PetscBool  isascii;

  PetscFunctionBegin;
  PetscCall(SVMGetGramian(svm,&G));
  if (!G) {
    SETERRQ(PetscObjectComm((PetscObject) v),PETSC_ERR_ARG_NULL,"Gramian (kernel) matrix is not set");
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject) svm,v));

    PetscCall(PetscViewerASCIIPushTab(v));

    PetscCall(MatGetSize(G,&M,&N));
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject) G,v));
    PetscCall(PetscViewerASCIIPushTab(v));
    PetscCall(PetscViewerASCIIPrintf(v,"dimensions: %D,%D\n",M,N));
    /* TODO information related to kernel type, parameters, mod etc. */
    PetscCall(PetscViewerASCIIPopTab(v));

    PetscCall(PetscViewerASCIIPopTab(v));
  } else {
    PetscCall(PetscObjectGetType((PetscObject) v,&type_name));
    SETERRQ(PetscObjectComm((PetscObject) v),PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewGramian",type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMLoadTrainingDataset_Binary"
PetscErrorCode SVMLoadTrainingDataset_Binary(SVM svm,PetscViewer v)
{
  MPI_Comm  comm;

  Mat       Xt_training;
  Mat       Xt_biased;
  Vec       y_training;

  PetscReal bias;
  PetscInt  mod;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) svm,&comm));
  /* Create matrix of training samples */
  PetscCall(MatCreate(comm,&Xt_training));
  PetscCall(PetscObjectSetName((PetscObject) Xt_training,"Xt_training"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) Xt_training,"Xt_training_"));
  PetscCall(MatSetFromOptions(Xt_training));

  PetscCall(VecCreate(comm,&y_training));
  PetscCall(VecSetFromOptions(y_training));
  PetscCall(PetscObjectSetName((PetscObject) y_training,"y_training"));

  PetscCall(PetscLogEventBegin(SVM_LoadDataset,svm,0,0,0));
  PetscCall(PetscViewerLoadSVMDataset(Xt_training,y_training,v));
  PetscCall(PetscLogEventEnd(SVM_LoadDataset,svm,0,0,0));

  PetscCall(SVMGetMod(svm,&mod));
  if (mod == 2) {
    PetscCall(SVMGetBias(svm,&bias));
    PetscCall(MatBiasedCreate(Xt_training,bias,&Xt_biased));
    Xt_training = Xt_biased;
  }
  PetscCall(SVMSetTrainingDataset(svm,Xt_training,y_training));

  /* Free memory */
  PetscCall(MatDestroy(&Xt_training));
  PetscCall(VecDestroy(&y_training));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMViewTrainingDataset_Binary"
PetscErrorCode SVMViewTrainingDataset_Binary(SVM svm,PetscViewer v)
{
  MPI_Comm   comm;

  Mat        Xt;
  Vec        y;

  PetscBool  isascii;
  const char *type_name = NULL;

  PetscFunctionBegin;
  PetscCall(SVMGetTrainingDataset(svm,&Xt,&y));
  if (!Xt || !y) {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    SETERRQ(comm,PETSC_ERR_ARG_NULL,"Training dataset is not set");
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

    SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewTrainingDataset",type_name);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVMViewTrainingPredictions_Binary(SVM svm,PetscViewer v)
{
  MPI_Comm   comm;

  Mat        Xt_training;
  Vec        y_pred;

  PetscFunctionBegin;
  PetscCall(SVMGetTrainingDataset(svm,&Xt_training,NULL));
  if (!Xt_training) {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    SETERRQ(comm,PETSC_ERR_ARG_NULL,"Training dataset is not set");
  }

  /* View predictions on training samples */
  PetscCall(SVMPredict(svm,Xt_training,&y_pred));
  PetscCall(PetscObjectSetName((PetscObject) y_pred,"y_training_predictions"));
  PetscCall(VecView(y_pred,v));

  /* Free memory */
  PetscCall(VecDestroy(&y_pred));
  PetscFunctionReturn(0);
}

PetscErrorCode SVMViewTestPredictions_Binary(SVM svm,PetscViewer v)
{
  MPI_Comm   comm;

  Mat        Xt_test;
  Vec        y_pred;

  PetscFunctionBegin;
  PetscCall(SVMGetTestDataset(svm,&Xt_test,NULL));
  if (!Xt_test) {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    SETERRQ(comm,PETSC_ERR_ARG_NULL,"Test dataset is not set");
  }

  /* View predictions on test samples */
  PetscCall(SVMPredict(svm,Xt_test,&y_pred));
  PetscCall(PetscObjectSetName((PetscObject) y_pred,"y_test_predictions"));
  PetscCall(VecView(y_pred,v));

  /* Free memory */
  PetscCall(VecDestroy(&y_pred));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMCreate_Binary"
PetscErrorCode SVMCreate_Binary(SVM svm)
{
  SVM_Binary *svm_binary;

  PetscInt   i;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(svm,&svm_binary));
  svm->data = (void *) svm_binary;

  svm_binary->w           = NULL;
  svm_binary->b           = 1.;
  svm_binary->qps         = NULL;
  svm_binary->Xt_training = NULL;
  svm_binary->G           = NULL;
  svm_binary->J           = NULL;
  svm_binary->diag        = NULL;
  svm_binary->y_training  = NULL;
  svm_binary->y_inner     = NULL;
  svm_binary->is_p        = NULL;
  svm_binary->is_n        = NULL;

  svm_binary->nsv         = 0;
  svm_binary->is_sv       = NULL;
  svm_binary->chop_tol    = PETSC_MACHINE_EPSILON;

  PetscCall(PetscMemzero(svm_binary->y_map,2 * sizeof(PetscScalar)));
  PetscCall(PetscMemzero(svm_binary->confusion_matrix,4 * sizeof(PetscInt)));
  PetscCall(PetscMemzero(svm_binary->model_scores,7 * sizeof(PetscReal)));

  for (i = 0; i < 3; ++i) {
    svm_binary->work[i] = NULL;
  }

  svm->ops->setup                   = SVMSetUp_Binary;
  svm->ops->reset                   = SVMReset_Binary;
  svm->ops->destroy                 = SVMDestroy_Binary;
  svm->ops->setfromoptions          = SVMSetFromOptions_Binary;
  svm->ops->train                   = SVMTrain_Binary;
  svm->ops->posttrain               = SVMPostTrain_Binary;
  svm->ops->reconstructhyperplane   = SVMReconstructHyperplane_Binary;
  svm->ops->predict                 = SVMPredict_Binary;
  svm->ops->test                    = SVMTest_Binary;
  svm->ops->crossvalidation         = SVMCrossValidation_Binary;
  svm->ops->gridsearch              = SVMGridSearch_Binary;
  svm->ops->view                    = SVMView_Binary;
  svm->ops->viewscore               = SVMViewScore_Binary;
  svm->ops->computemodelscores      = SVMComputeModelScores_Binary;
  svm->ops->computehingeloss        = SVMComputeHingeLoss_Binary;
  svm->ops->computemodelparams      = SVMComputeModelParams_Binary;
  svm->ops->loadgramian             = SVMLoadGramian_Binary;
  svm->ops->viewgramian             = SVMViewGramian_Binary;
  svm->ops->loadtrainingdataset     = SVMLoadTrainingDataset_Binary;
  svm->ops->viewtrainingdataset     = SVMViewTrainingDataset_Binary;
  svm->ops->viewtrainingpredictions = SVMViewTrainingPredictions_Binary;
  svm->ops->viewtestpredictions     = SVMViewTestPredictions_Binary;

  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetGramian_C",SVMSetGramian_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetGramian_C",SVMGetGramian_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetOperator_C",SVMSetOperator_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetOperator_C",SVMGetOperator_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetTrainingDataset_C",SVMSetTrainingDataset_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetTrainingDataset_C",SVMGetTrainingDataset_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMComputeOperator_C",SVMComputeOperator_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetQPS_C",SVMSetQPS_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetQPS_C",SVMGetQPS_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetQP_C",SVMGetQP_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetBias_C",SVMSetBias_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetBias_C",SVMGetBias_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetSeparatingHyperplane_C",SVMSetSeparatingHyperplane_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetSeparatingHyperplane_C",SVMGetSeparatingHyperplane_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetModelScore_C",SVMGetModelScore_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetOptionsPrefix_C",SVMSetOptionsPrefix_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetOptionsPrefix_C",SVMGetOptionsPrefix_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMAppendOptionsPrefix_C",SVMAppendOptionsPrefix_Binary));

  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMKFoldCrossValidation_C",SVMKFoldCrossValidation_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMStratifiedKFoldCrossValidation_C",SVMStratifiedKFoldCrossValidation_Binary));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMMonitorCreateMCtx_Binary"
PetscErrorCode SVMMonitorCreateCtx_Binary(void **mctx,SVM svm)
{
  SVM_Binary_mctx *mctx_inner;

  PetscFunctionBegin;
  PetscValidPointer(mctx,1);
  PetscValidHeaderSpecific(svm,SVM_CLASSID,2);

  PetscCall(PetscNew(&mctx_inner));
  mctx_inner->svm_inner = svm;
  *mctx = mctx_inner;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMMonitorDestroyMCtx_Binary"
PetscErrorCode SVMMonitorDestroyCtx_Binary(void **mctx)
{
  SVM_Binary_mctx *mctx_inner = (SVM_Binary_mctx *) *mctx;

  PetscFunctionBegin;
  mctx_inner->svm_inner = NULL;
  PetscCall(PetscFree(mctx_inner));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMMonitorDefault_Binary"
PetscErrorCode SVMMonitorDefault_Binary(QPS qps,PetscInt it,PetscReal rnorm,void *mctx)
{
  MPI_Comm comm;

  SVM         svm_inner;
  SVM_Binary *svm_binary;

  PetscViewer v;

  PetscFunctionBegin;
  svm_inner = ((SVM_Binary_mctx *) mctx)->svm_inner;
  svm_binary = (SVM_Binary *) svm_inner->data;

  PetscCall(SVMReconstructHyperplane(svm_inner));
  PetscCall(SVMComputeModelParams(svm_inner));

  comm = PetscObjectComm((PetscObject) svm_inner);
  v = PETSC_VIEWER_STDOUT_(comm);

  PetscCall(PetscViewerASCIIPrintf(v,"%3D SVM ||w||=%.10e",it,svm_binary->norm_w));
  PetscCall(PetscViewerASCIIPrintf(v,",\tmargin=%.10e",svm_binary->margin));
  PetscCall(PetscViewerASCIIPrintf(v,",\tbias=%.10e",svm_binary->b));
  PetscCall(PetscViewerASCIIPrintf(v,",\tNSV=%3D\n",svm_binary->nsv));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__
PetscErrorCode SVMMonitorObjFuncs_Binary(QPS qps,PetscInt it,PetscReal rnorm,void *mctx)
{
  MPI_Comm comm;

  SVM         svm_inner;
  SVM_Binary *svm_binary;

  PetscViewer v;

  PetscInt    p;        /* penalty type */
  SVMLossType loss_type;

  PetscFunctionBegin;
  svm_inner  = ((SVM_Binary_mctx *) mctx)->svm_inner;
  svm_binary = (SVM_Binary *) svm_inner->data;

  PetscCall(SVMGetLossType(svm_inner,&loss_type));
  PetscCall(SVMGetPenaltyType(svm_inner,&p));

  PetscCall(SVMReconstructHyperplane(svm_inner));
  PetscCall(SVMComputeObjFuncValues_Binary_Private(svm_inner));

  comm = PetscObjectComm((PetscObject) svm_inner);
  v = PETSC_VIEWER_STDOUT_(comm);

  PetscCall(PetscViewerASCIIPrintf(v,"%3D SVM primalObj=%.10e,",it,svm_binary->primalObj));
  PetscCall(PetscViewerASCIIPushTab(v));
  PetscCall(PetscViewerASCIIPrintf(v,"dualObj=%.10e,",svm_binary->dualObj));
  PetscCall(PetscViewerASCIIPrintf(v,"gap=%.10e,",svm_binary->primalObj - svm_binary->dualObj));
  if (p == 1) {
    PetscCall(PetscViewerASCIIPrintf(v,"%s-HingeLoss=%.10e\n",SVMLossTypes[loss_type],svm_binary->hinge_loss));
  } else {
    PetscCall(PetscViewerASCIIPrintf(v,"%s-HingeLoss+=%.10e %s-HingeLoss-=%.10e\n",SVMLossTypes[loss_type],svm_binary->hinge_loss_p, SVMLossTypes[loss_type],svm_binary->hinge_loss_n));
  }
  PetscCall(PetscViewerASCIIPopTab(v));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMMonitorScores_Binary"
PetscErrorCode SVMMonitorScores_Binary(QPS qps,PetscInt it,PetscReal rnorm,void *mctx)
{
  MPI_Comm    comm;
  SVM         svm_inner;
  SVM_Binary *svm_binary;

  Mat         Xt_test;
  Vec         y_known;
  Vec         y_pred;

  PetscViewer v;

  PetscFunctionBegin;
  svm_inner  = ((SVM_Binary_mctx *) mctx)->svm_inner;
  svm_binary = (SVM_Binary *) svm_inner->data;

  PetscCall(SVMGetTestDataset(svm_inner,&Xt_test,&y_known));

  svm_inner->posttraincalled = PETSC_TRUE;
  PetscCall(SVMReconstructHyperplane(svm_inner));
  PetscCall(SVMPredict(svm_inner,Xt_test,&y_pred));
  svm_inner->posttraincalled = PETSC_FALSE;

  /* Evaluation of model performance scores */
  PetscCall(SVMComputeModelScores(svm_inner,y_pred,y_known));

  comm = PetscObjectComm((PetscObject) svm_inner);
  v = PETSC_VIEWER_STDOUT_(comm);

  PetscCall(PetscViewerASCIIPrintf(v,"%3D SVM accuracy_test=%.2f%%,",it,svm_binary->model_scores[0] * 100.));
  PetscCall(PetscViewerASCIIPushTab(v));
  PetscCall(PetscViewerASCIIPrintf(v,"precision_test=%.2f%%,",svm_binary->model_scores[1] * 100.));
  PetscCall(PetscViewerASCIIPrintf(v,"sensitivity_test=%.2f%%,",svm_binary->model_scores[2] * 100.));
  PetscCall(PetscViewerASCIIPrintf(v,"F1_test=%.2f,",svm_binary->model_scores[3]));
  PetscCall(PetscViewerASCIIPrintf(v,"MCC_test=%.2f",svm_binary->model_scores[4]));
  PetscCall(PetscViewerASCIIPrintf(v,"AUC_ROC_test=%.2f",svm_binary->model_scores[5]));
  PetscCall(PetscViewerASCIIPrintf(v,"G1_test=%.2f\n",svm_binary->model_scores[6]));
  PetscCall(PetscViewerASCIIPopTab(v));

  PetscCall(VecDestroy(&y_pred));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMMonitorTrainingScores_Binary"
PetscErrorCode SVMMonitorTrainingScores_Binary(QPS qps,PetscInt it,PetscReal rnorm,void *mctx)
{
  MPI_Comm    comm;

  SVM         svm_inner;
  SVM_Binary  *svm_binary;

  Mat         Xt_training;
  Vec         y_known;
  Vec         y_pred;

  PetscViewer v;

  PetscFunctionBegin;
  svm_inner  = ((SVM_Binary_mctx *) mctx)->svm_inner;
  svm_binary = (SVM_Binary *) svm_inner->data;

  PetscCall(SVMGetTrainingDataset(svm_inner,&Xt_training,&y_known));

  svm_inner->posttraincalled = PETSC_TRUE;
  PetscCall(SVMReconstructHyperplane(svm_inner));
  PetscCall(SVMPredict(svm_inner,Xt_training,&y_pred));
  svm_inner->posttraincalled = PETSC_FALSE;

  /* Evaluation of model performance scores */
  PetscCall(SVMComputeModelScores(svm_inner,y_pred,y_known));

  comm = PetscObjectComm((PetscObject) svm_inner);
  v = PETSC_VIEWER_STDOUT_(comm);

  PetscCall(PetscViewerASCIIPrintf(v,"%3D SVM accuracy_training=%.2f%%,",it,svm_binary->model_scores[0] * 100.));
  PetscCall(PetscViewerASCIIPushTab(v));
  PetscCall(PetscViewerASCIIPrintf(v,"precision_training=%.2f%%,",svm_binary->model_scores[1] * 100.));
  PetscCall(PetscViewerASCIIPrintf(v,"sensitivity_training=%.2f%%,",svm_binary->model_scores[2] * 100.));
  PetscCall(PetscViewerASCIIPrintf(v,"F1_training=%.2f,",svm_binary->model_scores[3]));
  PetscCall(PetscViewerASCIIPrintf(v,"MCC_training=%.2f",svm_binary->model_scores[4]));
  PetscCall(PetscViewerASCIIPrintf(v,"AUC_ROC_training=%.2f",svm_binary->model_scores[5]));
  PetscCall(PetscViewerASCIIPrintf(v,"G1_training=%.2f\n",svm_binary->model_scores[6]));
  PetscCall(PetscViewerASCIIPopTab(v));

  PetscCall(VecDestroy(&y_pred));
  PetscFunctionReturn(0);
}
