
#include "binaryimpl.h"

PetscClassId SVM_CLASSID;

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
    TRY( QPSDestroy(&svm_binary->qps) );
    svm_binary->qps = NULL;
  }
  TRY( VecDestroy(&svm_binary->w) );
  TRY( MatDestroy(&svm_binary->Xt_training) );
  TRY( MatDestroy(&svm_binary->G) );
  TRY( MatDestroy(&svm_binary->J) );
  TRY( VecDestroy(&svm_binary->diag) );
  TRY( VecDestroy(&svm_binary->y_training) );
  TRY( VecDestroy(&svm_binary->y_inner) );
  TRY( ISDestroy(&svm_binary->is_p) );
  TRY( ISDestroy(&svm_binary->is_n) );

  TRY( PetscMemzero(svm_binary->y_map,2 * sizeof(PetscScalar)) );
  TRY( PetscMemzero(svm_binary->confusion_matrix,4 * sizeof(PetscInt)) );
  TRY( PetscMemzero(svm_binary->model_scores,7 * sizeof(PetscReal)) );

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
  TRY( ISDestroy(&svm_binary->is_sv) );
  svm_binary->is_sv       = NULL;

  for (i = 0; i < 3; ++i) {
    TRY( VecDestroy(&svm_binary->work[i]) );
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
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetGramian_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetGramian_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetOperator_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetOperator_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetTrainingDataset_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetTrainingDataset_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMComputeOperator_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetQPS_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetQPS_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetQP_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetBias_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetBias_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetSeparatingHyperplane_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetSeparatingHyperplane_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetModelScore_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetOptionsPrefix_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetOptionsPrefix_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMAppendOptionsPrefix_C",NULL) );

  TRY( QPSDestroy(&svm_binary->qps) );
  TRY( SVMDestroyDefault(svm) );
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

  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&isascii) );

  if (isascii) {
    TRY( PetscViewerASCIIPrintf(v,"=====================\n") );
    TRY( PetscObjectPrintClassNamePrefixType((PetscObject) svm,v) );

    TRY( PetscViewerASCIIPushTab(v) );
    TRY( PetscViewerASCIIPrintf(v,"model parameters:\n") );
    TRY( PetscViewerASCIIPushTab(v) );
    TRY( PetscViewerASCIIPrintf(v,"||w||=%.4f",svm_binary->norm_w) );
    TRY( PetscViewerASCIIPrintf(v,"bias=%.4f",svm_binary->b) );
    TRY( PetscViewerASCIIPrintf(v,"margin=%.4f",svm_binary->margin) );
    TRY( PetscViewerASCIIPrintf(v,"NSV=%d\n",svm_binary->nsv) );
    TRY( PetscViewerASCIIPopTab(v) );

    TRY( SVMGetLossType(svm,&loss_type) );
    TRY( SVMGetPenaltyType(svm,&p) );
    TRY( PetscViewerASCIIPrintf(v,"%s hinge loss:\n",SVMLossTypes[loss_type]) );
    TRY( PetscViewerASCIIPushTab(v) );
    if (p == 1) {
      if (loss_type == SVM_L1) {
        TRY( PetscViewerASCIIPrintf(v,"sum(xi_i)=%.4f\n",svm_binary->hinge_loss) );
      } else {
        TRY( PetscViewerASCIIPrintf(v,"sum(xi_i^2)=%.4f\n",svm_binary->hinge_loss) );
      }
    } else {
      if (loss_type == SVM_L1) {
        TRY( PetscViewerASCIIPrintf(v,"sum(xi_i+)=%.4f",svm_binary->hinge_loss_p) );
        TRY( PetscViewerASCIIPrintf(v,"sum(xi_i-)=%.4f\n",svm_binary->hinge_loss_n) );
      } else {
        TRY( PetscViewerASCIIPrintf(v,"sum(xi_i+^2)=%.4f",svm_binary->hinge_loss_n) );
        TRY( PetscViewerASCIIPrintf(v,"sum(xi_i-^2)=%.4f\n",svm_binary->hinge_loss_n) );
      }
    }
    TRY( PetscViewerASCIIPopTab(v) );

    TRY( PetscViewerASCIIPrintf(v,"objective functions:\n",SVMLossTypes[loss_type]) );
    TRY( PetscViewerASCIIPushTab(v) );
    TRY( PetscViewerASCIIPrintf(v,"primalObj=%.4f",svm_binary->primalObj) );
    TRY( PetscViewerASCIIPrintf(v,"dualObj=%.4f",svm_binary->dualObj) );
    TRY( PetscViewerASCIIPrintf(v,"gap=%.4f\n",svm_binary->primalObj - svm_binary->dualObj) );
    TRY( PetscViewerASCIIPopTab(v) );

    TRY( PetscViewerASCIIPopTab(v) );

    TRY( PetscViewerASCIIPrintf(v,"=====================\n") );
  } else {
    FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewScore", ((PetscObject)v)->type_name);
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

  TRY( PetscObjectTypeCompare((PetscObject) v,PETSCVIEWERASCII,&isascii) );
  if (isascii) {
    TRY( SVMGetPenaltyType(svm,&p) );
    TRY( SVMGetMod(svm,&mod) );
    TRY( SVMGetLossType(svm,&loss_type) );

    TRY( PetscViewerASCIIPrintf(v,"=====================\n") );
    TRY( PetscObjectPrintClassNamePrefixType((PetscObject) svm,v) );

    TRY( PetscViewerASCIIPushTab(v) );
    if (p == 1) {
      TRY( SVMGetC(svm,&C) );
      TRY( PetscViewerASCIIPrintf(v,"model performance score with training parameters C=%.3f, mod=%d, loss=%s:\n",C,mod,SVMLossTypes[loss_type]) );
    } else {
      TRY( SVMGetCp(svm,&Cp) );
      TRY( SVMGetCn(svm,&Cn) );
      TRY( PetscViewerASCIIPrintf(v,"model performance score with training parameters C+=%.3f, C-=%.3f, mod=%d, loss=%s:\n",Cp,Cn,mod,SVMLossTypes[loss_type]) );
    }
    TRY( PetscViewerASCIIPushTab(v) );

    TRY( PetscViewerASCIIPrintf(v,"Confusion matrix:\n") );
    TRY( PetscViewerASCIIPushTab(v) );
    TRY( PetscViewerASCIIPrintf(v,"TP = %4d",svm_binary->confusion_matrix[0]) );
    TRY( PetscViewerASCIIPrintf(v,"FP = %4d\n",svm_binary->confusion_matrix[1]) );
    TRY( PetscViewerASCIIPrintf(v,"FN = %4d",svm_binary->confusion_matrix[2]) );
    TRY( PetscViewerASCIIPrintf(v,"TN = %4d\n",svm_binary->confusion_matrix[3]) );
    TRY( PetscViewerASCIIPopTab(v) );

    TRY( PetscViewerASCIIPrintf(v,"accuracy=%.2f%%",svm_binary->model_scores[0] * 100.) );
    TRY( PetscViewerASCIIPrintf(v,"precision=%.2f%%",svm_binary->model_scores[1] * 100.) );
    TRY( PetscViewerASCIIPrintf(v,"sensitivity=%.2f%%\n",svm_binary->model_scores[2] * 100.) );
    TRY( PetscViewerASCIIPrintf(v,"F1=%.2f",svm_binary->model_scores[3]) );
    TRY( PetscViewerASCIIPrintf(v,"MCC=%.2f",svm_binary->model_scores[4]) );
    TRY( PetscViewerASCIIPrintf(v,"AUC_ROC=%.2f",svm_binary->model_scores[5]) );
    TRY( PetscViewerASCIIPrintf(v,"G1=%.2f\n",svm_binary->model_scores[6]) );
    TRY( PetscViewerASCIIPopTab(v) );

    TRY( PetscViewerASCIIPopTab(v) );

    TRY( PetscViewerASCIIPrintf(v,"=====================\n") );
  } else {
    FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewScore", ((PetscObject)v)->type_name);
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
  /* Checking that Gramian is rectangular */
  TRY( MatGetSize(G,&m,&n) );
  if (m != n) {
    FLLOP_SETERRQ2(PetscObjectComm((PetscObject) G),PETSC_ERR_ARG_SIZ,"Gramian (kernel) matrix must be rectangular, G(%D,%D)",m,n);
  }
  /* Checking dimension compatibility between training data matrix and Gramian */
  TRY( SVMGetTrainingDataset(svm,&Xt,NULL) );
  if (Xt) {
    TRY( MatGetSize(Xt,&n,NULL) );
    if (m != n) {
      FLLOP_SETERRQ2(PetscObjectComm((PetscObject) G),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, G(%D,) != X_training(%D,)",m,n);
    }
  }
  TRY( MatDestroy(&svm_binary->G) );
  TRY( PetscObjectReference((PetscObject) G) );
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
  /* Checking that operator (Hessian) is rectangular */
  TRY( MatGetSize(A,&m,&n) );
  if (m != n) {
    FLLOP_SETERRQ2(PetscObjectComm((PetscObject) A),PETSC_ERR_ARG_SIZ,"Hessian matrix must be rectangular, G(%D,%D)",m,n);
  }
  /* Checking dimension compatibility between Hessian (operator) and training data matrices */
  TRY( SVMGetTrainingDataset(svm,&Xt,NULL) );
  if (Xt) {
    TRY( MatGetSize(Xt,&n,NULL) );
    if (m != n) {
      FLLOP_SETERRQ2(PetscObjectComm((PetscObject) A),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, A(%D,) != X_training(%D,)",m,n);
    }
  }

  TRY( SVMGetQP(svm,&qp) );
  TRY( QPSetOperator(qp,A) );

  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetOperator_Binary"
PetscErrorCode SVMGetOperator_Binary(SVM svm,Mat *A)
{
  QP  qp;

  PetscFunctionBegin;
  TRY( SVMGetQP(svm,&qp) );
  TRY( QPGetOperator(qp,A) );
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
  TRY( MatGetSize(Xt_training,&m,NULL) );
  TRY( VecGetSize(y_training,&n) );
  if (m != n) {
    FLLOP_SETERRQ2(PetscObjectComm((PetscObject) Xt_training),PETSC_ERR_ARG_SIZ,"Dimensions are incompatible, X_training(%D,) != y_training(%D)",m,n);
  }

  TRY( SVMGetGramian(svm,&G) );
  if (G) {
    TRY( MatGetSize(G,&n,NULL) );
    if (m != n) {
      FLLOP_SETERRQ2(PetscObjectComm((PetscObject) G),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, X_training(%D,) != G(%D,)",m,n);
    }
  }

  TRY( MatDestroy(&svm_binary->Xt_training) );
  svm_binary->Xt_training = Xt_training;
  TRY( PetscObjectReference((PetscObject) Xt_training) );

  TRY( VecDestroy(&svm_binary->y_training) );
  svm_binary->y_training = y_training;
  TRY( PetscObjectReference((PetscObject) y_training) );

  /* Determine index sets of positive and negative samples */
  TRY( VecGetOwnershipRange(y_training,&lo,&hi) );
  TRY( VecMax(y_training,NULL,&max) );
  TRY( VecDuplicate(y_training,&tmp) );
  TRY( VecSet(tmp,max) );

  /* Index set for positive samples */
  TRY( ISDestroy(&svm_binary->is_p) );
  TRY( VecWhichEqual(y_training,tmp,&svm_binary->is_p) );

  /* Index set for negative samples */
  TRY( ISDestroy(&svm_binary->is_n) );
  TRY( ISComplement(svm_binary->is_p,lo,hi,&svm_binary->is_n) );

  /* Free memory */
  TRY( VecDestroy(&tmp) );
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
  TRY( SVMGetTrainingDataset(svm,NULL,&y) );
  TRY( VecMin(y,NULL,&min) );
  TRY( VecMax(y,NULL,&max) );

  if (min == -1.0 && max == 1.0) {
    TRY( VecDestroy(&svm_binary->y_inner) );
    svm_binary->y_inner = y;
    TRY( PetscObjectReference((PetscObject) y) );
  } else {
    TRY( VecGetLocalSize(y,&n) );
    TRY( VecDuplicate(y,&svm_binary->y_inner) );
    TRY( VecGetArrayRead(y,&y_arr) );
    TRY( VecGetArray(svm_binary->y_inner,&y_inner_arr) );
    for (i = 0; i < n; ++i) {
      if (y_arr[i]==min) {
        y_inner_arr[i] = -1.0;
      } else if (y_arr[i] == max) {
        y_inner_arr[i] = 1.0;
      }
    }
    TRY( VecRestoreArrayRead(y,&y_arr) );
    TRY( VecRestoreArray(svm_binary->y_inner,&y_inner_arr) );
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

  TRY( SVMGetMod(svm,&svm_mod) );

  TRY( QPSDestroy(&svm_binary->qps) );
  TRY( QPSCreate(PetscObjectComm((PetscObject) svm),&qps_inner) );

  if (svm_mod == 1) {
    TRY( QPSSetType(qps_inner,QPSSMALXE) );
    TRY( QPSSMALXEGetInnerQPS(qps_inner,&qps_smalxe_inner) );
    TRY( QPSSetType(qps_smalxe_inner,QPSMPGP) );
    TRY( QPSSMALXESetOperatorMaxEigenvalueTolerance(qps_inner,max_eig_tol) );
    TRY( QPSSMALXESetOperatorMaxEigenvalueIterations(qps_inner,max_eig_it) );
    TRY( QPSSetTolerances(qps_smalxe_inner,rtol,PETSC_DEFAULT,divtol,max_it) );
    TRY( QPSSMALXESetM1Initial(qps_inner,1.0,QPS_ARG_MULTIPLE) );
    TRY( QPSSMALXESetRhoInitial(qps_inner,1.0,QPS_ARG_MULTIPLE) );
  } else {
    TRY( QPSSetType(qps_inner,QPSMPGP) );
    TRY( QPSSetTolerances(qps_inner,rtol,PETSC_DEFAULT,divtol,max_it) );
    TRY( QPSMPGPSetOperatorMaxEigenvalueTolerance(qps_inner,max_eig_tol) );
    TRY( QPSMPGPSetOperatorMaxEigenvalueIterations(qps_inner,max_eig_it) );
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
    TRY( SVMCreateQPS_Binary_Private(svm,&qps_inner) );
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

  TRY( QPSDestroy(&svm_binary->qps) );
  svm_binary->qps = qps;
  TRY( PetscObjectReference((PetscObject) qps) );

  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetQP_Binary"
PetscErrorCode SVMGetQP_Binary(SVM svm,QP *qp)
{
  QPS qps;

  PetscFunctionBegin;
  TRY( SVMGetQPS(svm,&qps) );
  TRY( QPSGetQP(qps,qp) );
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
  TRY( SVMGetLossType(svm,&loss_type) );
  if (loss_type == SVM_L1) PetscFunctionReturn(0);

  /* Update regularization of Hessian */
  TRY( SVMGetPenaltyType(svm,&p) );
  if (p == 1) {
    TRY( SVMGetC(svm,&C) );

    TRY( MatScale(svm_binary->J,0.) );
    TRY( MatShift(svm_binary->J,1. / C) );
  } else {
    TRY( SVMGetCp(svm,&Cp) );
    TRY( SVMGetCn(svm,&Cn) );

    TRY( VecGetSubVector(svm_binary->diag,svm_binary->is_p,&diag_p) );
    TRY( VecSet(diag_p,1. / Cp) );
    TRY( VecRestoreSubVector(svm_binary->diag,svm_binary->is_p,&diag_p) );

    TRY( VecGetSubVector(svm_binary->diag,svm_binary->is_n,&diag_n) );
    TRY( VecSet(diag_n,1. / Cn) );
    TRY( VecRestoreSubVector(svm_binary->diag,svm_binary->is_n,&diag_n) );
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
  TRY( SVMUpdateOperator_Binary_Private(svm) );

  TRY( SVMGetPenaltyType(svm,&p) );
  if (p == 1) {
    TRY( SVMGetC(svm,&C) );
  } else {
    TRY( SVMGetCp(svm,&Cp) );
    TRY( SVMGetCn(svm,&Cn) );
  }

  /* Update initial guess */
  TRY( SVMGetQP(svm,&qp) );
  TRY( QPGetSolutionVector(qp,&x_init) );

  /* TODO SVMUpdateInitialVector_Binary_Private */
  if (svm->warm_start) {
    if (p == 1) {
      TRY( VecScale(x_init,1. / svm->C_old) );
      TRY( VecScale(x_init,C) );
    } else {
      TRY( VecGetSubVector(x_init,svm_binary->is_p,&x_init_p) );
      TRY( VecScale(x_init_p,1. / svm->Cp_old) );
      TRY( VecScale(x_init_p,Cp) );
      TRY( VecRestoreSubVector(x_init,svm_binary->is_p,&x_init_p) );

      TRY( VecGetSubVector(x_init,svm_binary->is_n,&x_init_n) );
      TRY( VecScale(x_init_n,1. / svm->Cn_old) );
      TRY( VecScale(x_init_n,Cn) );
      TRY( VecRestoreSubVector(x_init,svm_binary->is_n,&x_init_n) );
    }
  } else {
    if (p == 1) {
      TRY( VecSet(x_init,C - 100 * PETSC_MACHINE_EPSILON) );
    } else {
      TRY( VecGetSubVector(x_init,svm_binary->is_p,&x_init_p) );
      TRY( VecSet(x_init_p,Cp - 100 * PETSC_MACHINE_EPSILON) );
      TRY( VecRestoreSubVector(x_init,svm_binary->is_p,&x_init_p) );

      TRY( VecGetSubVector(x_init,svm_binary->is_n,&x_init_n) );
      TRY( VecSet(x_init_n,Cn - 100 * PETSC_MACHINE_EPSILON) );
      TRY( VecRestoreSubVector(x_init,svm_binary->is_n,&x_init_n) );
    }
  }

  /* Update upper bound vector */
  TRY( SVMGetLossType(svm,&loss_type) );
  if (loss_type == SVM_L2) PetscFunctionReturn(0);

  TRY( QPGetBox(qp,NULL,NULL,&ub) );
  if (p == 1) {
    TRY( VecSet(ub,C) );
  } else {
    TRY( VecGetSubVector(ub,svm_binary->is_p,&ub_p) );
    TRY( VecSet(ub_p,Cp) );
    TRY( VecRestoreSubVector(ub,svm_binary->is_p,&ub_p) );

    TRY( VecGetSubVector(ub,svm_binary->is_n,&ub_n) );
    TRY( VecSet(ub_n,Cn) );
    TRY( VecRestoreSubVector(ub,svm_binary->is_n,&ub_n) );
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
  TRY( SVMGetOperator(svm,&H) );
  if (H) PetscFunctionReturn(0);

  TRY( SVMGetPenaltyType(svm,&p) );
  TRY( SVMGetLossType(svm,&loss_type) );

  TRY( SVMGetGramian(svm,&G) );
  /* Create Gramian matrix (X^t * X) implicitly if it is not set by user, i.e. G == NULL */
  if (!G) {
    /* TODO add option for computing Gramian explicitly */
    TRY( SVMGetTrainingDataset(svm,&Xt,NULL) );
    TRY( PermonMatTranspose(Xt,MAT_TRANSPOSE_CHEAPEST,&X) );
    TRY( MatCreateNormal(X,&G) );  /* G = X^t * X */
  } else {
    TRY( PetscObjectReference((PetscObject) G) );
  }

  /* Remap y to -1,1 values if needed */
  TRY( SVMSetUp_Remapy_Binary_Private(svm) );
  y = svm_binary->y_inner;

  /* Create Hessian matrix */
  H = G;
  TRY( MatDiagonalScale(H,y,y) ); /* H = diag(y) * G * diag(y) */

  /* Regularize Hessian in case of l2-loss SVM */
  if (loss_type == SVM_L2) {
    /* 1 / 2t = C / 2 => t = 1 / C */
    /* H = H + t * I */
    /* https://link.springer.com/article/10.1134/S1054661812010129 */
    /* http://www.lib.kobe-u.ac.jp/repository/90000225.pdf */
    TRY( PetscObjectGetComm((PetscObject) H,&comm) );
    TRY( MatDestroy(&svm_binary->J) );

    if (p == 1) { /* Penalty type 1 */
      TRY( SVMGetC(svm,&C) );

      TRY( MatGetLocalSize(H,&m,&n) );
      TRY( MatGetSize(H,&M,&N) );

      TRY( MatCreateConstantDiagonal(comm,m,n,M,N,1. / C,&svm_binary->J) );
      TRY( MatAssemblyBegin(svm_binary->J,MAT_FINAL_ASSEMBLY) );
      TRY( MatAssemblyEnd(svm_binary->J,MAT_FINAL_ASSEMBLY) );
    } else { /* Penalty type 2 */
      TRY( SVMGetCp(svm,&Cp) );
      TRY( SVMGetCn(svm,&Cn) );

      TRY( VecDestroy(&svm_binary->diag) );
      TRY( VecDuplicate(y,&svm_binary->diag) );
      diag = svm_binary->diag;

      TRY( VecGetSubVector(diag,svm_binary->is_p,&diag_p) );
      TRY( VecSet(diag_p,1. / Cp) );
      TRY( VecRestoreSubVector(diag,svm_binary->is_p,&diag_p) );

      TRY( VecGetSubVector(diag,svm_binary->is_n,&diag_n) );
      TRY( VecSet(diag_n,1. / Cn) );
      TRY( VecRestoreSubVector(diag,svm_binary->is_n,&diag_n) );

      TRY( MatCreateDiag(diag,&svm_binary->J) );
    }

    mats[0] = svm_binary->J;
    mats[1] = H;
    TRY( MatCreateSum(comm,2,mats,&HpJ) ); /* H = H + J */
    TRY( MatDestroy(&H) );

    H  = HpJ;
  }

  *A = H;
  /* Decreasing reference counts */
  TRY( MatDestroy(&X) );
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
    TRY( SVMUpdate_Binary_Private(svm) );
    PetscFunctionReturn(0);
  }

  /* TODO generalize implementation of hyper parameter optimization */
  if (svm->hyperoptset) {
    TRY( SVMGridSearch(svm) );
  }

  TRY( SVMComputeOperator(svm,&H) ); /* compute Hessian of QP problem */
  TRY( SVMSetOperator(svm,H) );

  TRY( SVMGetTrainingDataset(svm,&Xt,&y) ); /* get samples and label vector for latter computing */

  TRY( SVMGetQPS(svm,&qps) );
  TRY( QPSGetQP(qps,&qp) );

  /* Set RHS */
  TRY( VecDuplicate(y,&e) );  /* creating vector e same size and type as y_training */
  TRY( VecSet(e,1.) );
  TRY( QPSetRhs(qp,e) );      /* set linear term of QP problem */

  /* Set equality constraint for SVM mod 1 */
  TRY( SVMGetMod(svm,&svm_mod) );
  if (svm_mod == 1) {
    TRY( MatCreateOneRow(y,&Be) );   /* Be = y^t */
    TRY( VecNorm(y,NORM_2,&norm) );
    TRY( MatScale(Be,1. / norm) );    /* ||Be|| = 1 */
    TRY( QPSetEq(qp,Be,NULL) );
  }

  /* Create box constraint */
  TRY( VecDuplicate(y,&lb) );  /* create lower bound constraint */
  TRY( VecSet(lb,0.) );

  TRY( SVMGetPenaltyType(svm,&p) );
  if (p == 1) {
    TRY( SVMGetC(svm,&C) );
  } else {
    TRY( SVMGetCp(svm,&Cp) );
    TRY( SVMGetCn(svm,&Cn) );
  }

  TRY( SVMGetLossType(svm,&loss_type) );
  if (loss_type == SVM_L1) {
    TRY( VecDuplicate(lb,&ub) );

    if (p == 1) {
      TRY( VecSet(ub,C) );
    } else {
      /* Set upper bound constrain related to positive samples */
      TRY( VecGetSubVector(ub,svm_binary->is_p,&ub_p) );
      TRY( VecSet(ub_p,Cp) );
      TRY( VecRestoreSubVector(ub,svm_binary->is_p,&ub_p) );

      /* Set upper bound constrain related to negative samples */
      TRY( VecGetSubVector(ub,svm_binary->is_n,&ub_n) );
      TRY( VecSet(ub_n,Cn) );
      TRY( VecRestoreSubVector(ub,svm_binary->is_n,&ub_n) );
    }
  }

  TRY( QPSetBox(qp,NULL,lb,ub) );

  /* TODO create public method for setting initial vector */
  /* Set initial guess */
  TRY( VecDuplicate(lb,&x_init) );
  if (p == 1) {
    TRY( VecSet(x_init,C - 100 * PETSC_MACHINE_EPSILON) );
  } else {
    TRY( VecGetSubVector(x_init,svm_binary->is_p,&x_init_p) );
    TRY( VecSet(x_init_p,Cp - 100 * PETSC_MACHINE_EPSILON) );
    TRY( VecRestoreSubVector(x_init,svm_binary->is_p,&x_init_p) );

    TRY( VecGetSubVector(x_init,svm_binary->is_n,&x_init_n) );
    TRY( VecSet(x_init_n,Cn - 100 * PETSC_MACHINE_EPSILON) );
    TRY( VecRestoreSubVector(x_init,svm_binary->is_n,&x_init_n) );
  }
  TRY( QPSetInitialVector(qp,x_init) );
  TRY( VecDestroy(&x_init) );

  /* TODO create public method for setting monitors */
  /* Set monitors */
  TRY( PetscOptionsHasName(NULL,((PetscObject) svm)->prefix,"-svm_monitor",&svm_monitor_set) );
  if (svm_monitor_set) {
    TRY( SVMMonitorCreateCtx_Binary(&mctx,svm) );
    if (svm_mod == 1) {
      QPS qps_inner;
      TRY( QPSSMALXEGetInnerQPS(qps,&qps_inner) );
      TRY( QPSMonitorSet(qps_inner,SVMMonitorDefault_Binary,mctx,SVMMonitorDestroyCtx_Binary) );
    } else {
      TRY( QPSMonitorSet(qps,SVMMonitorDefault_Binary,mctx,SVMMonitorDestroyCtx_Binary) );
    }
  }
  TRY( PetscOptionsHasName(NULL,((PetscObject) svm)->prefix,"-svm_monitor_obj_funcs",&svm_monitor_set) );
  if (svm_monitor_set) {
    TRY( SVMMonitorCreateCtx_Binary(&mctx,svm) );
    if (svm_mod == 1) {
      QPS qps_inner;
      TRY( QPSSMALXEGetInnerQPS(qps,&qps_inner) );
      TRY( QPSMonitorSet(qps_inner,SVMMonitorObjFuncs_Binary,mctx,SVMMonitorDestroyCtx_Binary) );
    } else {
      TRY( QPSMonitorSet(qps,SVMMonitorObjFuncs_Binary,mctx,SVMMonitorDestroyCtx_Binary) );
    }
  }
  TRY( PetscOptionsHasName(NULL,((PetscObject) svm)->prefix,"-svm_monitor_training_scores",&svm_monitor_set) );
  if (svm_monitor_set) {
    TRY( SVMMonitorCreateCtx_Binary(&mctx,svm) );
    if (svm_mod == 1) {
      QPS qps_inner;
      TRY( QPSSMALXEGetInnerQPS(qps,&qps_inner) );
      TRY( QPSMonitorSet(qps_inner,SVMMonitorTrainingScores_Binary,mctx,SVMMonitorDestroyCtx_Binary) );
    } else {
      TRY( QPSMonitorSet(qps,SVMMonitorTrainingScores_Binary,mctx,SVMMonitorDestroyCtx_Binary) );
    }
  }
  TRY( PetscOptionsHasName(NULL,((PetscObject) svm)->prefix,"-svm_monitor_scores",&svm_monitor_set) );
  if (svm_monitor_set) {
    Mat Xt_test;
    Vec y_test;

    TRY( SVMGetTestDataset(svm,&Xt_test,&y_test) );
    if (!Xt_test && !y_test) {
      FLLOP_SETERRQ(((PetscObject) svm)->comm,PETSC_ERR_ARG_NULL,"Test dataset must be set for using -svm_monitor_scores.");
    }
    TRY( SVMMonitorCreateCtx_Binary(&mctx,svm) );
    if (svm_mod == 1) {
      QPS qps_inner;
      TRY( QPSSMALXEGetInnerQPS(qps,&qps_inner) );
      TRY( QPSMonitorSet(qps_inner,SVMMonitorScores_Binary,mctx,SVMMonitorDestroyCtx_Binary) );
    } else {
      TRY( QPSMonitorSet(qps,SVMMonitorScores_Binary,mctx,SVMMonitorDestroyCtx_Binary) );
    }
  }

  /* Set QPS */
  if (svm->setfromoptionscalled) {
    TRY( QPTFromOptions(qp) );
    TRY( QPSSetFromOptions(qps) );
  }
  TRY( QPSSetUp(qps) );

  /* Create work vectors */
  for (i = 0; i < 3; ++i) {
    TRY( VecDestroy(&svm_binary->work[i]) );
  }
  TRY( MatCreateVecs(Xt,NULL,&svm_binary->work[0]) ); /* TODO use duplicated vector y instead of creating vec? */
  TRY( VecDuplicate(svm_binary->work[0],&svm_binary->work[1]) );
  TRY( VecDuplicate(svm_binary->work[0],&svm_binary->work[2]) );

  /* Decreasing reference counts using destroy methods */
  TRY( MatDestroy(&H) );
  TRY( MatDestroy(&Be) );
  TRY( VecDestroy(&e) );
  TRY( VecDestroy(&lb) );
  TRY( VecDestroy(&ub) );

  svm->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetOptionsPrefix_Binary"
PetscErrorCode SVMSetOptionsPrefix_Binary(SVM svm,const char prefix[])
{
  QPS qps;

  PetscFunctionBegin;
  TRY( PetscObjectSetOptionsPrefix((PetscObject) svm,prefix) );
  TRY( SVMGetQPS(svm,&qps) );
  TRY( QPSSetOptionsPrefix(qps,prefix) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMAppendOptionsPrefix_Binary"
PetscErrorCode SVMAppendOptionsPrefix_Binary(SVM svm,const char prefix[])
{
  QPS qps;

  PetscFunctionBegin;
  TRY( PetscObjectAppendOptionsPrefix((PetscObject) svm,prefix) );
  TRY( SVMGetQPS(svm,&qps) );
  TRY( QPSAppendOptionsPrefix(qps,prefix) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetOptionsPrefix_Binary"
PetscErrorCode SVMGetOptionsPrefix_Binary(SVM svm,const char *prefix[])
{

  PetscFunctionBegin;
  TRY( PetscObjectGetOptionsPrefix((PetscObject) svm,prefix) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMTrain_Binary"
PetscErrorCode SVMTrain_Binary(SVM svm)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  TRY( SVMSetUp(svm) );
  TRY( QPSSetAutoPostSolve(svm_binary->qps,PETSC_FALSE) );
  TRY( QPSSolve(svm_binary->qps) );
  if (svm->autoposttrain) {
    TRY( SVMPostTrain(svm) );
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
  TRY( SVMGetTrainingDataset(svm,&Xt,NULL) );
  y = svm_binary->y_inner;

  /* Reconstruction of hyperplane normal */
  TRY( SVMGetQP(svm,&qp) );
  TRY( QPGetSolutionVector(qp,&x) );
  TRY( VecDuplicate(x,&yx) );

  TRY( VecPointwiseMult(yx,y,x) ); /* yx = y.*x */
  TRY( MatCreateVecs(Xt,&w_inner,NULL) );
  TRY( MatMultTranspose(Xt,yx,w_inner) ); /* w = (X^t)^t * yx = X * yx */

  /* Reconstruction of the hyperplane bias */
  TRY( SVMGetMod(svm,&svm_mod) );
  if (svm_mod == 1) {
    TRY( QPGetBox(qp,NULL,&lb,&ub) );

    TRY( ISDestroy(&svm_binary->is_sv) );
    if (svm->loss_type == SVM_L1) {
      TRY( VecWhichBetween(lb,x,ub,&svm_binary->is_sv) );
    } else {
      TRY( VecWhichGreaterThan(x,lb,&svm_binary->is_sv) );
    }
    TRY( ISGetSize(svm_binary->is_sv,&svm_binary->nsv) );

    TRY( MatMult(Xt,w_inner,svm_binary->work[2]) );

    TRY( VecGetSubVector(y,svm_binary->is_sv,&y_sv) );     /* y_sv = y(is_sv) */
    TRY( VecGetSubVector(svm_binary->work[2],svm_binary->is_sv,&Xtw_sv) ); /* Xtw_sv = Xt(is_sv) */
    TRY( VecDuplicate(y_sv,&t) );
    TRY( VecWAXPY(t,-1.,Xtw_sv,y_sv) );
    TRY( VecRestoreSubVector(y,svm_binary->is_sv,&y_sv) );
    TRY( VecRestoreSubVector(svm_binary->work[2],svm_binary->is_sv,&Xtw_sv) );
    TRY( VecSum(t,&b_inner) );

    b_inner /= svm_binary->nsv;

    TRY( VecDestroy(&t) );
  } else {
    b_inner = 0.;
  }

  TRY( SVMSetSeparatingHyperplane(svm,w_inner,b_inner) );

  TRY( VecDestroy(&w_inner) );
  TRY( VecDestroy(&yx) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetSeparatingHyperplane_Binary"
PetscErrorCode SVMSetSeparatingHyperplane_Binary(SVM svm,Vec w,PetscReal b)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  PetscCheckSameComm(svm,1,w,2);

  TRY( VecDestroy(&svm_binary->w) );
  svm_binary->w = w;
  svm_binary->b = b;
  TRY( PetscObjectReference((PetscObject) w) );
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

  SVMLossType loss_type;
  PetscInt    svm_mod;

  PetscFunctionBegin;
  TRY( SVMGetQP(svm,&qp) );
  TRY( SVMGetLossType(svm,&loss_type) );
  TRY( SVMGetMod(svm,&svm_mod) );

  TRY( SVMGetSeparatingHyperplane(svm,&w,NULL) );
  if (!w) {
    TRY( SVMReconstructHyperplane(svm) );
  }

  TRY( VecNorm(w,NORM_2,&svm_binary->norm_w) );
  svm_binary->margin = 2. / svm_binary->norm_w;

  TRY( QPGetSolutionVector(qp,&x) );
  TRY( QPGetBox(qp,NULL,&lb,&ub) );

  if (svm_mod == 2) {
    TRY( ISDestroy(&svm_binary->is_sv) );
    if (loss_type == SVM_L1) {
      TRY( VecWhichBetween(lb, x, ub, &svm_binary->is_sv) );
    } else {
      TRY( VecWhichGreaterThan(x, lb, &svm_binary->is_sv) );
    }
    TRY( ISGetSize(svm_binary->is_sv, &svm_binary->nsv) );
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
  TRY( SVMGetMod(svm,&svm_mod) );
  TRY( SVMGetLossType(svm,&loss_type) );
  TRY( SVMGetPenaltyType(svm,&p) );
  TRY( SVMGetTrainingDataset(svm,&Xt,&y) );

  TRY( SVMGetSeparatingHyperplane(svm,&w,NULL) );
  if (!w) {
    TRY( SVMReconstructHyperplane(svm) );
    TRY( SVMGetSeparatingHyperplane(svm,&w,NULL) );
  }

  if (svm_mod == 1) {
    TRY( SVMGetSeparatingHyperplane(svm,NULL,&b) );
    TRY( VecSet(svm_binary->work[1],-b) );
    TRY( VecCopy(svm_binary->work[2],svm_binary->work[0]) );
    TRY( VecAYPX(svm_binary->work[0],1.,svm_binary->work[1]) ); /* xi = Xtw - b */
  } else {
    TRY( MatMult(Xt,w,svm_binary->work[0]) ); /* xi = Xtw */
  }

  TRY( VecPointwiseMult(svm_binary->work[0],y,svm_binary->work[0]) ); /* xi = y .* xi */

  TRY( VecSet(svm_binary->work[1],1.) );
  TRY( VecAYPX(svm_binary->work[0],-1.,svm_binary->work[1]) );       /* xi = 1 - xi */
  TRY( VecSet(svm_binary->work[1],0.) );
  TRY( VecPointwiseMax(svm_binary->work[0],svm_binary->work[1],svm_binary->work[0]) ); /* max(0,xi) */

  if (loss_type == SVM_L1) {
    if (p == 1) {
      TRY( VecSum(svm_binary->work[0],&svm_binary->hinge_loss) ); /* hinge_loss = sum(xi) */
    } else {
      TRY( VecGetSubVector(svm_binary->work[0],svm_binary->is_p,&work_p) );
      TRY( VecSum(work_p,&svm_binary->hinge_loss_p) ); /* hinge_loss_p = sum(xi_p) */
      TRY( VecRestoreSubVector(svm_binary->work[0],svm_binary->is_p,&work_p) );

      TRY( VecGetSubVector(svm_binary->work[0],svm_binary->is_n,&work_n) );
      TRY( VecSum(work_n,&svm_binary->hinge_loss_n) ); /* hinge_loss_n = sum(xi_n) */
      TRY( VecRestoreSubVector(svm_binary->work[0],svm_binary->is_n,&work_n) );
    }
  } else {
    if (p == 1) {
      TRY( VecDot(svm_binary->work[0],svm_binary->work[0],&svm_binary->hinge_loss) ); /* hinge_loss = sum(xi^2) */
    } else {
      TRY( VecGetSubVector(svm_binary->work[0],svm_binary->is_p,&work_p) );
      TRY( VecDot(work_p,work_p,&svm_binary->hinge_loss_p) ); /* hinge_loss_p = sum(xi_p^2) */
      TRY( VecRestoreSubVector(svm_binary->work[0],svm_binary->is_p,&work_p) );

      TRY( VecGetSubVector(svm_binary->work[0],svm_binary->is_n,&work_n) );
      TRY( VecDot(work_n,work_n,&svm_binary->hinge_loss_n) ); /* hinge_loss_n = sum(xi_n^2) */
      TRY( VecRestoreSubVector(svm_binary->work[0],svm_binary->is_n,&work_n) );
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
  TRY( SVMGetLossType(svm,&loss_type) );
  TRY( SVMGetPenaltyType(svm,&p) );

  TRY( SVMComputeHingeLoss(svm) );

  /* Compute value of primal objective function */
  TRY( SVMGetSeparatingHyperplane(svm,&w,NULL) );

  TRY( VecDot(w,w,&svm_binary->primalObj) );
  svm_binary->primalObj *= 0.5;
  if (p == 1) {
    TRY( SVMGetC(svm,&C) );

    tmp = C * svm_binary->hinge_loss;
  } else {
    TRY( SVMGetCp(svm,&Cp) );
    TRY( SVMGetCn(svm,&Cn) );

    tmp = Cp * svm_binary->hinge_loss_p;
    tmp += Cn * svm_binary->hinge_loss_n;
  }

  if (loss_type == SVM_L1) {
    svm_binary->primalObj += tmp;
  } else {
    svm_binary->primalObj += tmp / 2.;
  }

  /* Compute value of dual objective function */
  TRY( SVMGetQP(svm,&qp) );
  TRY( QPGetSolutionVector(qp,&x) );
  TRY( QPComputeObjective(qp,x,&svm_binary->dualObj) );
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

  TRY( SVMGetQPS(svm,&qps) );
  TRY( QPSPostSolve(qps)  );

  TRY( SVMReconstructHyperplane(svm) );
  TRY( SVMComputeObjFuncValues_Binary_Private(svm) );
  TRY( SVMComputeModelParams(svm) );

  svm->posttraincalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetFromOptions_Binary"
PetscErrorCode SVMSetFromOptions_Binary(PetscOptionItems *PetscOptionsObject,SVM svm)
{
  PetscErrorCode ierr;

  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscReal b;
  PetscBool flg;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin((PetscObject) svm);CHKERRQ(ierr);
  TRY( PetscOptionsReal("-svm_bias","","SVMSetBias",svm_binary->b,&b,&flg) );
  if (flg) {
    TRY( SVMSetBias(svm,b) );
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
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
  TRY( SVMGetMod(svm,&mod) );

  TRY( MatGetSize(Xt_predict,NULL,&N_predict) );
  TRY( MatGetOwnershipRangeColumn(Xt_predict,&lo,&hi) );

  TRY( SVMGetSeparatingHyperplane(svm,&w,NULL) );
  TRY( PetscObjectGetComm((PetscObject) w,&comm) );

  n = 0;
  N_predict += (1 - mod);
  if (hi < N_predict) {
    n = hi - lo;
  } else if (lo < N_predict) {
    n = N_predict - lo;
  }
  TRY( ISCreateStride(comm,n,lo,1,&is) );

  if (mod == 2) {
    TRY( VecGetSize(w,&N) );
    TRY( VecGetOwnershipRange(w,NULL,&hi) );

    TRY( ISCreateStride(comm,(hi == N) ? 1 : 0,hi - 1,1,&is_last) );
    /* Concatenate is and is_last */
    TRY( ISExpand(is,is_last,&is_tmp) );
    /* Free memory */
    TRY( ISDestroy(&is) );
    TRY( ISDestroy(&is_last) );

    is = is_tmp;
  }

  TRY( VecGetSubVector(w,is,&w_sub_inner) );

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

  TRY( SVMGetMod(svm,&mod) );

  TRY( SVMGetSeparatingHyperplane(svm,&w,NULL) );

  TRY( VecGetOwnershipRange(w,&lo,&hi) );
  TRY( PetscObjectGetComm((PetscObject) Xt_predict,&comm) );

  TRY( MatGetOwnershipIS(Xt_predict,&is_rows,NULL) );
  if (mod == 2) {
    TRY( VecGetSize(w,&N) );
    tmp = (hi == N) ? 1 : 0;
  }
  TRY( ISCreateStride(comm,hi - lo - tmp,lo,1,&is_cols) );

  TRY( MatCreateSubMatrix(Xt_predict,is_rows,is_cols,MAT_INITIAL_MATRIX,&Xt_sub) );

  *Xt_out = Xt_sub;

  /* Free memory */
  TRY( ISDestroy(&is_rows) );
  TRY( ISDestroy(&is_cols) );
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
    TRY( SVMPostTrain(svm) );
  }
  TRY( SVMGetSeparatingHyperplane(svm,NULL,&b) );
  TRY( SVMGetTrainingDataset(svm,&Xt_training,NULL) );

  TRY( MatGetSize(Xt_training,NULL,&N_training) );
  TRY( MatGetSize(Xt_pred,NULL,&N_predict) );
  /* Check number of features */
  if (N_training > N_predict) {
    TRY( SVMGetHyperplaneSubNormal_Binary_Private(svm,Xt_pred,&is_w,&w) );
  } else {
    TRY( SVMGetSeparatingHyperplane(svm,&w,NULL) );
    if (N_training < N_predict) {
      TRY( SVMCreateSubPredictDataset_Binary_Private(svm,Xt_pred,&Xt_sub) );
      Xt_pred = Xt_sub;
    }
  }

  /* Predict labels of unseen samples */
  TRY( MatCreateVecs(Xt_pred,NULL,&Xtw) );
  TRY( VecGetOwnershipRange(Xtw,&lo,&hi) );
  TRY( VecDuplicate(Xtw,&y) );
  TRY( VecDuplicate(Xtw,&o) );
  TRY( VecSet(o,0) );

  TRY( MatMult(Xt_pred,w,Xtw) );
  TRY( VecShift(Xtw,b) ); /* shifting is not performed in case of b = 0 (inner implementation) */

  TRY( VecWhichGreaterThan(Xtw,o,&is_p) );
  TRY( ISComplement(is_p,lo,hi,&is_n) );

  TRY( VecGetSubVector(y,is_n,&sub_y) );
  TRY( VecSet(sub_y,svm_binary->y_map[0]) );
  TRY( VecRestoreSubVector(y,is_n,&sub_y) );

  TRY( VecGetSubVector(y,is_p,&sub_y) );
  TRY( VecSet(sub_y,svm_binary->y_map[1]) );
  TRY( VecRestoreSubVector(y,is_p,&sub_y) );

  *y_out = y;

  /* Free memory */
  TRY( VecDestroy(&Xtw) );
  TRY( VecDestroy(&o) );
  TRY( ISDestroy(&is_n) );
  TRY( ISDestroy(&is_p) );
  if (N_training > N_predict) {
    TRY( SVMGetSeparatingHyperplane(svm,&w_tmp,NULL) );
    TRY( VecRestoreSubVector(w_tmp,is_w,&w) );
    TRY( ISDestroy(&is_w) );
  } else if (N_training < N_predict) {
    TRY( MatDestroy(&Xt_pred) );
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
  TRY( VecDuplicate(y_known,&label) );

  /* TN and FN samples */
  TRY( VecSet(label,svm_binary->y_map[0]) );
  TRY( VecWhichEqual(y_known,label,&is_label) );
  TRY( VecGetSubVector(y_known,is_label,&y_known_sub) );
  TRY( VecGetSubVector(y_pred,is_label,&y_pred_sub) );
  TRY( VecWhichEqual(y_known_sub,y_pred_sub,&is_eq) );

  TRY( ISGetSize(is_eq,&TN) );
  TRY( VecGetSize(y_known_sub,&N) );
  FN = N - TN;

  TRY( VecRestoreSubVector(y_known,is_label,&y_known_sub) );
  TRY( VecRestoreSubVector(y_pred,is_label,&y_pred_sub) );
  TRY( ISDestroy(&is_label) );
  TRY( ISDestroy(&is_eq) );

  /* TP and FP samples */
  TRY( VecSet(label,svm_binary->y_map[1]) );
  TRY( VecWhichEqual(y_known,label,&is_label) );
  TRY( VecGetSubVector(y_known,is_label,&y_known_sub) );
  TRY( VecGetSubVector(y_pred,is_label,&y_pred_sub) );
  TRY( VecWhichEqual(y_known_sub,y_pred_sub,&is_eq) );

  TRY( ISGetSize(is_eq,&TP) );
  TRY( VecGetSize(y_known_sub,&N) );
  FP = N - TP;

  TRY( VecRestoreSubVector(y_known,is_label,&y_known_sub) );
  TRY( VecRestoreSubVector(y_pred,is_label,&y_pred_sub) );
  TRY( ISDestroy(&is_label) );
  TRY( ISDestroy(&is_eq) );

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
  TRY( VecDestroy(&label) );
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
  TRY( SVMGetTestDataset(svm,&Xt_test,&y_known) );
  TRY( SVMPredict(svm,Xt_test,&y_pred) );

  /* Evaluation of model performance scores */
  TRY( SVMComputeModelScores(svm,y_pred,y_known) );
  TRY( VecDestroy(&y_pred) );
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
PetscErrorCode SVMInitGridSearch_Binary_Private(SVM svm,PetscInt *n,PetscReal *c_arr[])
{
  PetscInt  penalty_type;

  PetscReal logC_min,logC_max,logC_base;
  PetscReal logCp_min,logCp_max,logCp_base;
  PetscReal logCn_min,logCn_max,logCn_base;

  PetscReal Cp,Cn;
  PetscReal C_min,Cp_min,Cn_min;

  PetscReal *c_arr_inner;
  PetscInt  n_inner,np,nn;
  PetscInt  i,j,p;

  PetscFunctionBegin;
  TRY( SVMGetPenaltyType(svm,&penalty_type) );

  if (penalty_type == 1) {
    TRY( SVMGetLogCMin(svm,&logC_min) );
    TRY( SVMGetLogCMax(svm,&logC_max) );
    TRY( SVMGetLogCBase(svm,&logC_base) );

    C_min = PetscPowReal(logC_base,logC_min);

    n_inner = (PetscInt) (logC_max - logC_min) + 1;
    TRY( PetscMalloc1(n_inner,&c_arr_inner) );

    c_arr_inner[0] = C_min;
    for (i = 1; i < n_inner; ++i) {
      c_arr_inner[i] = c_arr_inner[i-1] * logC_base;
    }
  /* Penalty type 2: different penalty for each one class */
  } else {
    TRY( SVMGetLogCpMin(svm,&logCp_min) );
    TRY( SVMGetLogCpMax(svm,&logCp_max) );
    TRY( SVMGetLogCpBase(svm,&logCp_base) );

    TRY( SVMGetLogCnMin(svm,&logCn_min) );
    TRY( SVMGetLogCnMax(svm,&logCn_max) );
    TRY( SVMGetLogCnBase(svm,&logCn_base) );

    Cp_min = PetscPowReal(logCp_base,logCp_min);
    Cn_min = PetscPowReal(logCn_base,logCn_min);

    np = (PetscInt) (logCp_max - logCp_min) + 1;
    nn = (PetscInt) (logCn_max - logCn_min) + 1;
    n_inner = 2 * np * nn;

    TRY( PetscMalloc1(n_inner,&c_arr_inner) );

    /* Generate Cp and Cn values */
    Cp = Cp_min;
    p  = 0;
    for (i = 0; i < np; ++i) {
      c_arr_inner[p++] = Cp;
      c_arr_inner[p++] = Cn_min;
      Cn = Cn_min;
      for (j = 1; j < nn; ++j) {
        c_arr_inner[p++] = Cp;
        Cn *= logCn_base;
        c_arr_inner[p++] = Cn;
      }
      Cp *= logCp_base;
    }
  }

  *n = n_inner;
  *c_arr = c_arr_inner;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearch_Binary"
PetscErrorCode SVMGridSearch_Binary(SVM svm)
{
  MPI_Comm   comm;

  PetscReal  *c_arr,*score;
  PetscReal  score_best;
  PetscInt   m,n,i,p;

  PetscBool  info_set;
  const char *prefix;

  const ModelScore *hyperopt_score_types;
  PetscInt   nscores;

  PetscFunctionBegin;
  TRY( SVMInitGridSearch_Binary_Private(svm,&n,&c_arr) );
  TRY( SVMGetPenaltyType(svm,&m) );

  TRY( PetscMalloc1((n / m),&score) );
  TRY( PetscMemzero(score,(n / m) * sizeof(PetscReal)) );

  TRY( SVMCrossValidation(svm,c_arr,n,score) );

  /* Select penalty */
  n /= m;
  score_best = score[0];
  p = 0;
  for (i = 1; i < n; ++i) {
    if (score[i] > score_best) {
      p = i;
      score_best = score[i];
    }
  }
  TRY( SVMSetPenalty(svm,m,&c_arr[p * m]) );

  TRY( SVMGetOptionsPrefix(svm,&prefix) );
  TRY( PetscOptionsHasName(NULL,prefix,"-svm_info",&info_set) );

  if (info_set) {
    TRY( PetscObjectGetComm((PetscObject) svm,&comm) );
    TRY( SVMGetHyperOptScoreTypes(svm,&hyperopt_score_types) );
    TRY( SVMGetHyperOptNScoreTypes(svm,&nscores) );
    TRY( PetscPrintf(comm,"SVM (grid-search): selected ") );
    if (m == 1) {
      TRY( PetscPrintf(comm,"C_best=%f, ",c_arr[p]) );
    } else {
      TRY( PetscPrintf(comm,"C+_best=%f, ",c_arr[p * m]) );
      TRY( PetscPrintf(comm,"C-_best=%f, ",c_arr[p * m + 1]) );
    }
    TRY( PetscPrintf(comm,"acc_score=%f (",score_best) );
    for (i = 0; i < nscores; ++i) {
      TRY( PetscPrintf(comm,"%s",ModelScores[hyperopt_score_types[i]]) );
      if (i < nscores - 1) {
        TRY( PetscPrintf(comm,",") );
      } else {
        TRY( PetscPrintf(comm,")\n") );
      }
    }
  }
  TRY( PetscFree(c_arr) );
  TRY( PetscFree(score) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMLoadGramian_Binary"
PetscErrorCode SVMLoadGramian_Binary(SVM svm,PetscViewer v)
{
  Mat G;

  PetscFunctionBegin;
  /* Create matrix */
  TRY( MatCreate(PetscObjectComm((PetscObject) svm),&G) );
  TRY( MatSetType(G,MATDENSE) );
  TRY( PetscObjectSetName((PetscObject) G,"G") );
  TRY( PetscObjectSetOptionsPrefix((PetscObject) G,"G_") );
  TRY( MatSetFromOptions(G) );

  TRY( MatLoad(G,v) );
  TRY( SVMSetGramian(svm,G) );

  /* Free memory */
  TRY( MatDestroy(&G) );
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
  TRY( SVMGetGramian(svm,&G) );
  if (!G) {
    FLLOP_SETERRQ(PetscObjectComm((PetscObject) v),PETSC_ERR_ARG_NULL,"Gramian (kernel) matrix is not set");
  }

  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&isascii) );
  if (isascii) {
    TRY( PetscObjectPrintClassNamePrefixType((PetscObject) svm,v) );

    TRY( PetscViewerASCIIPushTab(v) );

    TRY( MatGetSize(G,&M,&N) );
    TRY( PetscObjectPrintClassNamePrefixType((PetscObject) G,v) );
    TRY( PetscViewerASCIIPushTab(v) );
    TRY( PetscViewerASCIIPrintf(v,"dimensions: %D,%D\n",M,N) );
    /* TODO information related to kernel type, parameters, mod etc. */
    TRY( PetscViewerASCIIPopTab(v) );

    TRY( PetscViewerASCIIPopTab(v) );
  } else {
    TRY( PetscObjectGetType((PetscObject) v,&type_name) );
    FLLOP_SETERRQ1(PetscObjectComm((PetscObject) v),PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewGramian",type_name);
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
  TRY( PetscObjectGetComm((PetscObject) svm,&comm) );
  /* Create matrix of training samples */
  TRY( MatCreate(comm,&Xt_training) );
  TRY( PetscObjectSetName((PetscObject) Xt_training,"Xt_training") );
  TRY( PetscObjectSetOptionsPrefix((PetscObject) Xt_training,"Xt_training_") );
  TRY( MatSetFromOptions(Xt_training) );

  TRY( VecCreate(comm,&y_training) );
  TRY( PetscObjectSetName((PetscObject) y_training,"y_training") );

  TRY( PetscLogEventBegin(SVM_LoadDataset,svm,0,0,0) );
  TRY( PetscViewerLoadSVMDataset(Xt_training,y_training,v) );
  TRY( PetscLogEventEnd(SVM_LoadDataset,svm,0,0,0) );

  TRY( SVMGetMod(svm,&mod) );
  if (mod == 2) {
    TRY( SVMGetBias(svm,&bias) );
    TRY( MatBiasedCreate(Xt_training,bias,&Xt_biased) );
    Xt_training = Xt_biased;
  }
  TRY( SVMSetTrainingDataset(svm,Xt_training,y_training) );

  /* Free memory */
  TRY( MatDestroy(&Xt_training) );
  TRY( VecDestroy(&y_training) );
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
  TRY( SVMGetTrainingDataset(svm,&Xt,&y) );
  if (!Xt || !y) {
    TRY( PetscObjectGetComm((PetscObject) v,&comm) );
    FLLOP_SETERRQ(comm,PETSC_ERR_ARG_NULL,"Training dataset is not set");
  }

  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&isascii) );
  if (isascii) {
    /* Print info related to svm type */
    TRY( PetscObjectPrintClassNamePrefixType((PetscObject) svm,v) );

    TRY( PetscViewerASCIIPushTab(v) );
    TRY( SVMViewDataset(svm,Xt,y,v) );
    TRY( PetscViewerASCIIPopTab(v) );
  } else {
    TRY( PetscObjectGetComm((PetscObject) v,&comm) );
    TRY( PetscObjectGetType((PetscObject) v,&type_name) );

    FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewTrainingDataset",type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMCreate_Binary"
PetscErrorCode SVMCreate_Binary(SVM svm)
{
  SVM_Binary *svm_binary;

  PetscInt   i;

  PetscFunctionBegin;
  TRY( PetscNewLog(svm,&svm_binary) );
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

  TRY( PetscMemzero(svm_binary->y_map,2 * sizeof(PetscScalar)) );
  TRY( PetscMemzero(svm_binary->confusion_matrix,4 * sizeof(PetscInt)) );
  TRY( PetscMemzero(svm_binary->model_scores,7 * sizeof(PetscReal)) );

  for (i = 0; i < 3; ++i) {
    svm_binary->work[i] = NULL;
  }

  svm->ops->setup                 = SVMSetUp_Binary;
  svm->ops->reset                 = SVMReset_Binary;
  svm->ops->destroy               = SVMDestroy_Binary;
  svm->ops->setfromoptions        = SVMSetFromOptions_Binary;
  svm->ops->train                 = SVMTrain_Binary;
  svm->ops->posttrain             = SVMPostTrain_Binary;
  svm->ops->reconstructhyperplane = SVMReconstructHyperplane_Binary;
  svm->ops->predict               = SVMPredict_Binary;
  svm->ops->test                  = SVMTest_Binary;
  svm->ops->crossvalidation       = SVMCrossValidation_Binary;
  svm->ops->gridsearch            = SVMGridSearch_Binary;
  svm->ops->view                  = SVMView_Binary;
  svm->ops->viewscore             = SVMViewScore_Binary;
  svm->ops->computemodelscores    = SVMComputeModelScores_Binary;
  svm->ops->computehingeloss      = SVMComputeHingeLoss_Binary;
  svm->ops->computemodelparams    = SVMComputeModelParams_Binary;
  svm->ops->loadgramian           = SVMLoadGramian_Binary;
  svm->ops->viewgramian           = SVMViewGramian_Binary;
  svm->ops->loadtrainingdataset   = SVMLoadTrainingDataset_Binary;
  svm->ops->viewtrainingdataset   = SVMViewTrainingDataset_Binary;

  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetGramian_C",SVMSetGramian_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetGramian_C",SVMGetGramian_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetOperator_C",SVMSetOperator_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetOperator_C",SVMGetOperator_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetTrainingDataset_C",SVMSetTrainingDataset_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetTrainingDataset_C",SVMGetTrainingDataset_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMComputeOperator_C",SVMComputeOperator_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetQPS_C",SVMSetQPS_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetQPS_C",SVMGetQPS_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetQP_C",SVMGetQP_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetBias_C",SVMSetBias_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetBias_C",SVMGetBias_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetSeparatingHyperplane_C",SVMSetSeparatingHyperplane_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetSeparatingHyperplane_C",SVMGetSeparatingHyperplane_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetModelScore_C",SVMGetModelScore_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetOptionsPrefix_C",SVMSetOptionsPrefix_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetOptionsPrefix_C",SVMGetOptionsPrefix_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMAppendOptionsPrefix_C",SVMAppendOptionsPrefix_Binary) );

  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMKFoldCrossValidation_C",SVMKFoldCrossValidation_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMStratifiedKFoldCrossValidation_C",SVMStratifiedKFoldCrossValidation_Binary) );
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

  TRY( PetscNew(&mctx_inner) );
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
  TRY( PetscFree(mctx_inner) );
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

  TRY( SVMReconstructHyperplane(svm_inner) );
  TRY( SVMComputeModelParams(svm_inner) );

  comm = PetscObjectComm((PetscObject) svm_inner);
  v = PETSC_VIEWER_STDOUT_(comm);

  TRY( PetscViewerASCIIPrintf(v,"%3D SVM ||w||=%.10e",it,svm_binary->norm_w) );
  TRY( PetscViewerASCIIPrintf(v,",\tmargin=%.10e",svm_binary->margin) );
  TRY( PetscViewerASCIIPrintf(v,",\tbias=%.10e",svm_binary->b) );
  TRY( PetscViewerASCIIPrintf(v,",\tNSV=%3D\n",svm_binary->nsv) );
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

  TRY( SVMGetLossType(svm_inner,&loss_type) );
  TRY( SVMGetPenaltyType(svm_inner,&p) );

  TRY( SVMReconstructHyperplane(svm_inner) );
  TRY( SVMComputeObjFuncValues_Binary_Private(svm_inner) );

  comm = PetscObjectComm((PetscObject) svm_inner);
  v = PETSC_VIEWER_STDOUT_(comm);

  TRY( PetscViewerASCIIPrintf(v,"%3D SVM primalObj=%.10e,",it,svm_binary->primalObj) );
  TRY( PetscViewerASCIIPushTab(v) );
  TRY( PetscViewerASCIIPrintf(v,"dualObj=%.10e,",svm_binary->dualObj) );
  TRY( PetscViewerASCIIPrintf(v,"gap=%.10e,",svm_binary->primalObj - svm_binary->dualObj) );
  if (p == 1) {
    TRY( PetscViewerASCIIPrintf(v,"%s-HingeLoss=%.10e\n",SVMLossTypes[loss_type],svm_binary->hinge_loss));
  } else {
    TRY( PetscViewerASCIIPrintf(v,"%s-HingeLoss+=%.10e %s-HingeLoss-=%.10e\n",SVMLossTypes[loss_type],svm_binary->hinge_loss_p, SVMLossTypes[loss_type],svm_binary->hinge_loss_n) );
  }
  TRY( PetscViewerASCIIPopTab(v) );
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

  TRY( SVMGetTestDataset(svm_inner,&Xt_test,&y_known) );

  svm_inner->posttraincalled = PETSC_TRUE;
  TRY( SVMReconstructHyperplane(svm_inner) );
  TRY( SVMPredict(svm_inner,Xt_test,&y_pred) );
  svm_inner->posttraincalled = PETSC_FALSE;

  /* Evaluation of model performance scores */
  TRY( SVMComputeModelScores(svm_inner,y_pred,y_known) );

  comm = PetscObjectComm((PetscObject) svm_inner);
  v = PETSC_VIEWER_STDOUT_(comm);

  TRY( PetscViewerASCIIPrintf(v,"%3D SVM accuracy_test=%.2f%%,",it,svm_binary->model_scores[0] * 100.) );
  TRY( PetscViewerASCIIPushTab(v) );
  TRY( PetscViewerASCIIPrintf(v,"precision_test=%.2f%%,",svm_binary->model_scores[1] * 100.) );
  TRY( PetscViewerASCIIPrintf(v,"sensitivity_test=%.2f%%,",svm_binary->model_scores[2] * 100.) );
  TRY( PetscViewerASCIIPrintf(v,"F1_test=%.2f,",svm_binary->model_scores[3]) );
  TRY( PetscViewerASCIIPrintf(v,"MCC_test=%.2f",svm_binary->model_scores[4]) );
  TRY( PetscViewerASCIIPrintf(v,"AUC_ROC_test=%.2f",svm_binary->model_scores[5]) );
  TRY( PetscViewerASCIIPrintf(v,"G1_test=%.2f\n",svm_binary->model_scores[6]) );
  TRY( PetscViewerASCIIPopTab(v) );

  TRY( VecDestroy(&y_pred) );
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

  TRY( SVMGetTrainingDataset(svm_inner,&Xt_training,&y_known) );

  svm_inner->posttraincalled = PETSC_TRUE;
  TRY( SVMReconstructHyperplane(svm_inner) );
  TRY( SVMPredict(svm_inner,Xt_training,&y_pred) );
  svm_inner->posttraincalled = PETSC_FALSE;

  /* Evaluation of model performance scores */
  TRY( SVMComputeModelScores(svm_inner,y_pred,y_known) );

  comm = PetscObjectComm((PetscObject) svm_inner);
  v = PETSC_VIEWER_STDOUT_(comm);

  TRY( PetscViewerASCIIPrintf(v,"%3D SVM accuracy_training=%.2f%%,",it,svm_binary->model_scores[0] * 100.) );
  TRY( PetscViewerASCIIPushTab(v) );
  TRY( PetscViewerASCIIPrintf(v,"precision_training=%.2f%%,",svm_binary->model_scores[1] * 100.) );
  TRY( PetscViewerASCIIPrintf(v,"sensitivity_training=%.2f%%,",svm_binary->model_scores[2] * 100.) );
  TRY( PetscViewerASCIIPrintf(v,"F1_training=%.2f,",svm_binary->model_scores[3]) );
  TRY( PetscViewerASCIIPrintf(v,"MCC_training=%.2f",svm_binary->model_scores[4]) );
  TRY( PetscViewerASCIIPrintf(v,"AUC_ROC_training=%.2f",svm_binary->model_scores[5]) );
  TRY( PetscViewerASCIIPrintf(v,"G1_training=%.2f\n",svm_binary->model_scores[6]) );
  TRY( PetscViewerASCIIPopTab(v) );

  TRY( VecDestroy(&y_pred) );
  PetscFunctionReturn(0);
}
