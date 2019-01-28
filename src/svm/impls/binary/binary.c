
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
    if (svm_binary->svm_mod == 1) {
      QPS  qps_inner;
      TRY( QPSSMALXEGetInnerQPS(svm_binary->qps,&qps_inner) );
      TRY( QPSMonitorCancel(qps_inner) );
    } else {
      TRY( QPSMonitorCancel(svm_binary->qps) );
    }
    TRY( QPSReset(svm_binary->qps) );
  }
  TRY( VecDestroy(&svm_binary->w) );
  TRY( MatDestroy(&svm_binary->Xt_training) );
  TRY( MatDestroy(&svm_binary->D) );
  TRY( VecDestroy(&svm_binary->diag) );
  TRY( VecDestroy(&svm_binary->y_training) );
  TRY( VecDestroy(&svm_binary->y_inner) );
  TRY( ISDestroy(&svm_binary->is_p) );
  TRY( ISDestroy(&svm_binary->is_n) );

  TRY( PetscMemzero(svm_binary->y_map,2 * sizeof(PetscScalar)) );
  TRY( PetscMemzero(svm_binary->confusion_matrix,4 * sizeof(PetscInt)) );
  TRY( PetscMemzero(svm_binary->model_scores,5 * sizeof(PetscReal)) );

  svm_binary->w           = NULL;
  svm_binary->Xt_training = NULL;
  svm_binary->y_training  = NULL;
  svm_binary->y_inner     = NULL;
  svm_binary->is_p        = NULL;
  svm_binary->is_n        = NULL;
  svm_binary->D           = NULL;
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
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetTrainingDataset_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetTrainingDataset_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetQPS_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetQPS_C",NULL) );
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
    TRY( PetscViewerASCIIPrintf(v,"%s hinge loss:\n",SVMLossTypes[loss_type]) );
    TRY( PetscViewerASCIIPushTab(v) );
    if (loss_type == SVM_L1) {
      TRY( PetscViewerASCIIPrintf(v,"sum(xi_i)=%.4f\n",svm_binary->hinge_loss) );
    } else {
      TRY( PetscViewerASCIIPrintf(v,"sum(xi_i^2)=%.4f\n",svm_binary->hinge_loss) );
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

  PetscReal   C;
  PetscInt    mod;
  SVMLossType loss_type;

  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;
  PetscBool isascii;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject) svm);
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);

  TRY( PetscObjectTypeCompare((PetscObject) v,PETSCVIEWERASCII,&isascii) );
  if (isascii) {
    TRY( SVMGetC(svm,&C) );
    TRY( SVMGetMod(svm,&mod) );
    TRY( SVMGetLossType(svm,&loss_type) );

    TRY( PetscViewerASCIIPrintf(v,"=====================\n") );
    TRY( PetscObjectPrintClassNamePrefixType((PetscObject) svm,v) );

    TRY( PetscViewerASCIIPushTab(v) );
    TRY( PetscViewerASCIIPrintf(v,"model performance score with training parameters C=%.3f, mod=%d, loss=%s:\n",C,mod,SVMLossTypes[loss_type]) );

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
    TRY( PetscViewerASCIIPrintf(v,"F1.0_score=%.2f",svm_binary->model_scores[3]) );
    TRY( PetscViewerASCIIPrintf(v,"mmc=%.2f\n",svm_binary->model_scores[4]) );
    TRY( PetscViewerASCIIPopTab(v) );

    TRY( PetscViewerASCIIPopTab(v) );

    TRY( PetscViewerASCIIPrintf(v,"=====================\n") );
  } else {
    FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewScore", ((PetscObject)v)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetTrainingDataset"
PetscErrorCode SVMSetTrainingDataset_Binary(SVM svm,Mat Xt_training,Vec y_training)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscReal  max;
  PetscInt   lo,hi;
  Vec        tmp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(Xt_training,MAT_CLASSID,2);
  PetscCheckSameComm(svm,1,Xt_training,2);
  PetscValidHeaderSpecific(y_training,VEC_CLASSID,3);
  PetscCheckSameComm(svm,1,y_training,3);

  TRY( MatDestroy(&svm_binary->Xt_training) );
  svm_binary->Xt_training = Xt_training;
  TRY( PetscObjectReference((PetscObject) Xt_training) );

  TRY( VecDestroy(&svm_binary->y_training) );
  svm_binary->y_training = y_training;
  TRY( PetscObjectReference((PetscObject) y_training) );

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
PetscErrorCode SVMCreateQPS_Binary_Private(SVM svm,QPS *qps) {
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
#define __FUNCT__ "SVMSetMod_Binary"
PetscErrorCode SVMSetMod_Binary(SVM svm,PetscInt mod)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  if (svm_binary->svm_mod != mod) {
    svm_binary->svm_mod = mod;
    svm->setupcalled = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetMod_Binary"
PetscErrorCode SVMGetMod_Binary(SVM svm,PetscInt *mod)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  *mod = svm_binary->svm_mod;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMUpdate_Binary_Private"
PetscErrorCode SVMUpdate_Binary_Private(SVM svm)
{
  SVM_Binary  *svm_binary = (SVM_Binary *) svm->data;

  QP          qp;
  QPS         qps;
  Vec         ub,ub_p,ub_n;
  Vec         diag_p,diag_n;
  Vec         x_init,x_init_p,x_init_n;

  PetscReal   C,Cn,Cp;
  SVMLossType loss_type;

  PetscFunctionBegin;
  TRY( SVMGetC(svm,&C) );
  TRY( SVMGetCp(svm,&Cp) );
  TRY( SVMGetCn(svm,&Cn) );
  TRY( SVMGetLossType(svm,&loss_type) );

  TRY( SVMGetQPS(svm,&qps) );
  TRY( QPSGetQP(qps,&qp) );
  /* Update upper boundary or Hessian */
  if (loss_type == SVM_L1) {
    TRY( QPGetBox(qp,NULL,NULL,&ub) );
    /* Update upper boundary */
    if (C > 0.) {
      TRY( VecSet(ub,C) );
    } else {
      TRY( VecGetSubVector(ub,svm_binary->is_p,&ub_p) );
      TRY( VecSet(ub_p,Cp) );
      TRY( VecRestoreSubVector(ub,svm_binary->is_p,&ub_p) );

      TRY( VecGetSubVector(ub,svm_binary->is_n,&ub_n) );
      TRY( VecSet(ub_n,Cn) );
      TRY( VecRestoreSubVector(ub,svm_binary->is_n,&ub_n) );
    }
  } else {
    /* Update Hessian */
    if (C > 0.) {
      TRY( VecSet(svm_binary->diag,1. / C) );
    } else {
      TRY( VecGetSubVector(svm_binary->diag,svm_binary->is_p,&diag_p) );
      TRY( VecSet(diag_p,1. / Cp) );
      TRY( VecRestoreSubVector(svm_binary->diag,svm_binary->is_p,&diag_p) );

      TRY( VecGetSubVector(svm_binary->diag,svm_binary->is_n,&diag_n) );
      TRY( VecSet(diag_n,1. / Cn) );
      TRY( VecRestoreSubVector(svm_binary->diag,svm_binary->is_n,&diag_n) );
    }
    TRY( MatDiagonalSet(svm_binary->D,svm_binary->diag,INSERT_VALUES) );
  }

  /* Update initial guess */
  TRY( QPGetSolutionVector(qp,&x_init) );
  if (svm->warm_start) {
    if (C > 0.) {
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
    if (C > 0.) {
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
  svm->setupcalled     = PETSC_TRUE;
  svm->posttraincalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetUp_Binary"
PetscErrorCode SVMSetUp_Binary(SVM svm)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  QP          qp;
  QPS         qps;

  Vec         diag,diag_p,diag_n;
  Mat         H,HpE,mats[2];
  Vec         e;
  Vec         x_init,x_init_p,x_init_n;

  Mat         Be = NULL;
  PetscReal   norm;
  Vec         lb;
  Vec         ub,ub_p,ub_n;

  Mat         Xt_training,X_training;
  Vec         y;
  PetscInt    n,m,N;

  PetscReal   C,Cp,Cn;
  PetscInt    svm_mod;
  SVMLossType loss_type;

  PetscBool   svm_monitor_set;
  void        *mctx;  /* monitor context */

  PetscFunctionBegin;
  if (svm->setupcalled) PetscFunctionReturn(0);

  TRY( SVMGetLossType(svm,&loss_type) );
  TRY( SVMGetMod(svm,&svm_mod) );
  TRY( SVMGetC(svm,&C) );
  TRY( SVMGetCp(svm,&Cp) );
  TRY( SVMGetCn(svm,&Cn) );

  if (svm->posttraincalled) {
    TRY( SVMUpdate_Binary_Private(svm) );
    PetscFunctionReturn(0);
  }

  if (C == PETSC_DECIDE) {
    TRY( SVMGridSearch(svm) );
    TRY( SVMGetC(svm,&C) );
  } else if (Cp == PETSC_DECIDE && Cn == PETSC_DECIDE) {
    TRY( SVMGridSearch(svm) );
    TRY( SVMGetCp(svm,&Cp) );
    TRY( SVMGetCn(svm,&Cn) );
  }
  TRY( SVMGetTrainingDataset(svm,&Xt_training,NULL) );

  /* Set QP and QPS solver */
  TRY( SVMGetQPS(svm,&qps) );
  TRY( QPSGetQP(qps,&qp) );

  /* Remap y to -1,1 values if needed */
  TRY( SVMSetUp_Remapy_Binary_Private(svm) );
  y = svm_binary->y_inner;

  /* Create Hessian */
  TRY( PermonMatTranspose(Xt_training,MAT_TRANSPOSE_CHEAPEST,&X_training) );
  TRY( MatCreateNormal(X_training,&H) );  /* H = X^t * X */
  TRY( MatDiagonalScale(H,y,y) );         /* H = diag(y)*H*diag(y) */

  /* Create right hand side vector*/
  TRY( VecDuplicate(y,&e) );  /* creating vector e same size and type as y_training */
  TRY( VecSet(e,1.0) );
  TRY( QPSetRhs(qp,e) );      /* set linear term of QP problem */

  /* Set equality constrain for solver type 1 (with equality constraint) */
  if (svm_mod == 1) {
    TRY( MatCreateOneRow(y,&Be) );   /* Be = y^t */
    TRY( VecNorm(y,NORM_2,&norm) );
    TRY( MatScale(Be,1.0/norm) );    /* ||Be|| = 1 */
    TRY( QPSetEq(qp,Be,NULL) );
  }

  /* Create box constraints */
  TRY( VecDuplicate(y,&lb) );  /* create lower bound constraint vector */
  TRY( VecSet(lb,0.) );

  if (loss_type == SVM_L1) {
    TRY( VecDuplicate(lb,&ub) );
    if (C > 0) {
      TRY( VecSet(ub,C) );
    } else {
      TRY( VecGetSubVector(ub,svm_binary->is_p,&ub_p) );
      TRY( VecSet(ub_p,Cp) );
      TRY( VecRestoreSubVector(ub,svm_binary->is_p,&ub_p) );

      TRY( VecGetSubVector(ub,svm_binary->is_n,&ub_n) );
      TRY( VecSet(ub_n,Cn) );
      TRY( VecRestoreSubVector(ub,svm_binary->is_n,&ub_n) );
    }
  } else {
    /* 1 / 2t = C / 2 => t = 1 / C */
    /* H = H + t * I */
    /* https://link.springer.com/article/10.1134/S1054661812010129 */
    /* http://www.lib.kobe-u.ac.jp/repository/90000225.pdf */

    TRY( MatGetLocalSize(H,&m,&n) );
    TRY( MatGetSize(H,&N,NULL) );

    /* Set values of diagonal matrix */
    TRY( VecDuplicate(lb,&svm_binary->diag) );
    diag = svm_binary->diag;
    if (C > 0) {
      TRY( VecSet(diag,1. / C) );
    } else {
      TRY( VecGetSubVector(diag,svm_binary->is_p,&diag_p) );
      TRY( VecSet(diag_p,1. / Cp) );
      TRY( VecRestoreSubVector(diag,svm_binary->is_p,&diag_p) );

      TRY( VecGetSubVector(diag,svm_binary->is_n,&diag_n) );
      TRY( VecSet(diag_n,1. / Cn) );
      TRY( VecRestoreSubVector(diag,svm_binary->is_n,&diag_n) );
    }
    TRY( MatCreateDiag(diag,&svm_binary->D) );

    mats[0] = svm_binary->D;
    mats[1] = H;
    TRY( MatCreateSum(PetscObjectComm((PetscObject)svm),2,mats,&HpE) );
    TRY( MatDestroy(&H) );
    H  = HpE;
    ub = NULL;
  }

  TRY( VecDuplicate(lb,&x_init) );
  if (C > 0) {
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

  TRY( QPSetOperator(qp,H) );
  TRY( QPSetBox(qp,NULL,lb,ub) );

  /* set monitors */
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

  if (svm->setfromoptionscalled) {
    TRY( QPSSetFromOptions(qps) );
  }

  TRY( QPSSetUp(qps) );

  /* decreasing reference counts */
  TRY( MatDestroy(&X_training) );
  TRY( MatDestroy(&H) );
  TRY( MatDestroy(&Be) );
  TRY( VecDestroy(&e) );
  TRY( VecDestroy(&lb) );
  TRY( VecDestroy(&ub) );

  /* create work vecs */
  TRY( MatCreateVecs(Xt_training,NULL,&svm_binary->work[0]) );
  TRY( VecDuplicate(svm_binary->work[0],&svm_binary->work[1]) );
  TRY( VecDuplicate(svm_binary->work[0],&svm_binary->work[2]) );

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
  QPS qps;

  PetscFunctionBegin;
  TRY( PetscObjectGetOptionsPrefix((PetscObject) svm,prefix) );
  TRY( SVMGetQPS(svm,&qps) );
  TRY( QPSGetOptionsPrefix(qps,prefix) );
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

  QPS       qps;
  QP        qp;
  Vec       lb,ub;

  Mat       Xt;
  Vec       x,y,yx,y_sv,t;
  Vec       Xtw_sv;

  Vec       w_inner;
  PetscReal b_inner;

  PetscInt  svm_mod;

  PetscFunctionBegin;
  TRY( SVMGetMod(svm,&svm_mod) );

  TRY( SVMGetQPS(svm,&qps) );
  TRY( QPSGetQP(qps,&qp) );

  TRY( SVMGetTrainingDataset(svm,&Xt,NULL) );
  y = svm_binary->y_inner;

  /* Reconstruction of hyperplane normal */
  TRY( QPGetSolutionVector(qp,&x) );
  TRY( VecDuplicate(x,&yx) );

  TRY( VecPointwiseMult(yx,y,x) ); /* yx = y.*x */
  TRY( MatCreateVecs(Xt,&w_inner,NULL) );
  TRY( MatMultTranspose(Xt,yx,w_inner) ); /* w = (X^t)^t * yx = X * yx */

  /* Reconstruction of the hyperplane bias */
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
  QPS         qps;
  Vec         x,lb,ub;

  Vec         w;

  SVMLossType loss_type;
  PetscInt    svm_mod;

  PetscFunctionBegin;
  TRY( SVMGetQPS(svm,&qps) );
  TRY( QPSGetQP(qps,&qp) );
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

  PetscInt    svm_mod;
  SVMLossType loss_type;

  PetscFunctionBegin;
  TRY( SVMGetMod(svm,&svm_mod) );
  TRY( SVMGetLossType(svm,&loss_type) );
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
    TRY( VecSum(svm_binary->work[0],&svm_binary->hinge_loss) ); /* hinge_loss = sum(xi) */
  } else {
    TRY( VecDot(svm_binary->work[0],svm_binary->work[0],&svm_binary->hinge_loss) ); /* hinge_loss = sum(xi^2) */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMComputeObjFuncValues_Binary_Private"
PetscErrorCode SVMComputeObjFuncValues_Binary_Private(SVM svm)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  Vec         w;
  SVMLossType loss_type;

  QPS         qps,qps_inner;
  QP          qp;
  Vec         rhs,x;

  PetscBool   issmalxe;


  PetscFunctionBegin;
  TRY( SVMGetLossType(svm,&loss_type) );
  TRY( SVMComputeHingeLoss(svm) );

  /* Compute value of primal objective function */
  TRY( SVMGetSeparatingHyperplane(svm,&w,NULL) );
  TRY( VecDot(w,w,&svm_binary->primalObj) );
  svm_binary->primalObj *= 0.5;

  if (loss_type == SVM_L1) {
    svm_binary->primalObj += svm->C * svm_binary->hinge_loss;
  } else {
    svm_binary->primalObj += svm->C * svm_binary->hinge_loss / 2.;
  }

  /* Compute value of dual objective function */
  TRY( SVMGetQPS(svm,&qps) );
  TRY( QPSGetQP(qps,&qp) );
  TRY( QPGetRhs(qp,&rhs) );
  TRY( QPGetSolutionVector(qp,&x) );

  TRY( PetscObjectTypeCompare((PetscObject) qps,QPSSMALXE,&issmalxe) );
  if (issmalxe) {
    TRY( QPSSMALXEGetInnerQPS(qps,&qps_inner) );
    qps = qps_inner;
  }

  TRY( VecWAXPY(svm_binary->work[1],-1.,rhs,qps->work[3]) );
  TRY( VecDot(x,svm_binary->work[1],&svm_binary->dualObj ) );
  svm_binary->dualObj *= -.5;
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

  PetscInt  svm_mod;
  PetscReal b;
  PetscBool flg;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin((PetscObject) svm);CHKERRQ(ierr);
  TRY( PetscOptionsInt("-svm_binary_mod","","SVMSetMod",svm_binary->svm_mod,&svm_mod,&flg) );
  if (flg) {
    TRY( SVMSetMod(svm,svm_mod) );
  }
  TRY( PetscOptionsReal("-svm_bias","","SVMSetBias",svm_binary->b,&b,&flg) );
  if (flg) {
    TRY( SVMSetBias(svm,b) );
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMPredict_Binary"
PetscErrorCode SVMPredict_Binary(SVM svm,Mat Xt_pred,Vec *y_out)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  Vec        Xtw_pred,y,w;
  PetscReal  b;
  PetscInt   i,m;

  const PetscScalar *Xtw_pred_arr;
  PetscScalar *y_pred_arr;

  PetscFunctionBegin;
  if (!svm->posttraincalled) {
   TRY( SVMPostTrain(svm) );
  }
  TRY( SVMGetSeparatingHyperplane(svm,&w,&b) );

  TRY( MatCreateVecs(Xt_pred,NULL,&Xtw_pred) );
  TRY( MatMult(Xt_pred,w,Xtw_pred) );

  TRY( VecDuplicate(Xtw_pred,&y) );
  TRY( VecGetLocalSize(y,&m) );

  TRY( VecGetArrayRead(Xtw_pred, &Xtw_pred_arr) );
  TRY( VecGetArray(y, &y_pred_arr) );
  for (i = 0; i < m; ++i) {
    if (Xtw_pred_arr[i] + b > 0.0) {
      y_pred_arr[i] = svm_binary->y_map[1];
    } else {
      y_pred_arr[i] = svm_binary->y_map[0];
    }
  }
  TRY( VecRestoreArrayRead(Xtw_pred,&Xtw_pred_arr) );
  TRY( VecRestoreArray(y,&y_pred_arr) );

  *y_out = y;

  TRY( VecDestroy(&Xtw_pred) );
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
PetscErrorCode SVMInitGridSearch_Binary_Private(SVM svm,PetscInt *n,PetscInt *m,PetscReal *c_arr[])
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

  *m = penalty_type;
  *n = n_inner;
  *c_arr = c_arr_inner;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearch_Binary"
PetscErrorCode SVMGridSearch_Binary(SVM svm)
{

  PetscReal *c_arr,*score;
  PetscReal score_max,C_best;
  PetscInt  m,n,i;

  PetscFunctionBegin;
  TRY( SVMInitGridSearch_Binary_Private(svm,&n,&m,&c_arr) );

  TRY( PetscMalloc1(n,&score) );
  TRY( PetscMemzero(score,n * sizeof(PetscReal)) );
  /*TRY( SVMCrossValidation(svm,c_arr,n,score) );

  C_best = c_arr[0];
  score_max = score[0];
  for (i = 1; i < n; ++i) {
    if (score[i] > score_max) {
      score_max = score[i];
      C_best = c_arr[i];
    }
  }

  svm->C = C_best;*/
  TRY( PetscFree(c_arr) );
  TRY( PetscFree(score) );
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
  svm_binary->D           = NULL;
  svm_binary->diag        = NULL;
  svm_binary->y_training  = NULL;
  svm_binary->y_inner     = NULL;
  svm_binary->is_p        = NULL;
  svm_binary->is_n        = NULL;

  svm_binary->nsv         = 0;
  svm_binary->is_sv       = NULL;

  svm_binary->svm_mod     = 2;

  TRY( PetscMemzero(svm_binary->y_map,2 * sizeof(PetscScalar)) );
  TRY( PetscMemzero(svm_binary->confusion_matrix,4 * sizeof(PetscInt)) );
  TRY( PetscMemzero(svm_binary->model_scores,5 * sizeof(PetscReal)) );

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

  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetTrainingDataset_C",SVMSetTrainingDataset_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetTrainingDataset_C",SVMGetTrainingDataset_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetQPS_C",SVMSetQPS_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetQPS_C",SVMGetQPS_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetMod_C",SVMSetMod_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetMod_C",SVMGetMod_Binary) );
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

  SVMLossType loss_type;

  PetscFunctionBegin;
  svm_inner  = ((SVM_Binary_mctx *) mctx)->svm_inner;
  svm_binary = (SVM_Binary *) svm_inner->data;

  TRY( SVMGetLossType(svm_inner,&loss_type) );

  TRY( SVMReconstructHyperplane(svm_inner) );
  TRY( SVMComputeObjFuncValues_Binary_Private(svm_inner) );

  comm = PetscObjectComm((PetscObject) svm_inner);
  v = PETSC_VIEWER_STDOUT_(comm);

  TRY( PetscViewerASCIIPrintf(v,"%3D SVM primalObj=%.10e,",it,svm_binary->primalObj) );
  TRY( PetscViewerASCIIPushTab(v) );
  TRY( PetscViewerASCIIPrintf(v,"dualObj=%.10e,",svm_binary->dualObj) );
  TRY( PetscViewerASCIIPrintf(v,"gap=%.10e,",svm_binary->primalObj - svm_binary->dualObj) );
  TRY( PetscViewerASCIIPrintf(v,"%s-HingeLoss=%.10e\n",SVMLossTypes[loss_type],svm_binary->hinge_loss) );
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
  TRY( PetscViewerASCIIPrintf(v,"F1.0_score_test=%.2f,",svm_binary->model_scores[3]) );
  TRY( PetscViewerASCIIPrintf(v,"mmc_test=%.2f\n",svm_binary->model_scores[4]) );
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
  TRY( PetscViewerASCIIPrintf(v,"F1.0_score_training=%.2f,",svm_binary->model_scores[3]) );
  TRY( PetscViewerASCIIPrintf(v,"mmc_training=%.2f\n",svm_binary->model_scores[4]) );
  TRY( PetscViewerASCIIPopTab(v) );

  TRY( VecDestroy(&y_pred) );
  PetscFunctionReturn(0);
}
