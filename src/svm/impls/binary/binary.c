
#include <permon/private/svmimpl.h>

PetscClassId SVM_CLASSID;

const char *const SVMLossTypes[]={"L1","L2","SVMLossType","SVM_",0};

typedef struct {
  Mat         Xt_training;
  Vec         y_training;
  Vec         y_inner;

  PetscScalar y_map[2];
  Mat         D;

  Vec         w;
  PetscScalar b;

  QPS         qps;
  PetscInt    svm_mod;

  PetscInt    N_eq,N_all;
} SVM_Binary;

typedef struct {
  SVM svm_inner;
} SVM_Binary_mctx;

static PetscErrorCode SVMMonitorCreateMtx_Binary(void **,SVM);
static PetscErrorCode SVMMonitorDestroyMtx_Binary(void **);
static PetscErrorCode SVMMonitorDefault_Binary(QPS,PetscInt,PetscReal,void *);

#undef __FUNCT__
#define __FUNCT__ "SVMReset_Binary"
PetscErrorCode SVMReset_Binary(SVM svm)
{
  SVM_Binary *svm_binary;

  PetscFunctionBegin;
  svm_binary = (SVM_Binary *) svm->data;

  if (svm_binary->qps) {
    TRY( QPSDestroy(&svm_binary->qps) );
  }
  TRY( VecDestroy(&svm_binary->w) );
  TRY( MatDestroy(&svm_binary->Xt_training) );
  TRY( MatDestroy(&svm_binary->D) );
  TRY( VecDestroy(&svm_binary->y_training) );
  TRY( VecDestroy(&svm_binary->y_inner) );

  PetscMemzero(svm_binary->y_map,2 * sizeof(PetscScalar) );
  svm_binary->b = PETSC_INFINITY;

  svm_binary->w           = NULL;
  svm_binary->Xt_training = NULL;
  svm_binary->y_training  = NULL;
  svm_binary->y_inner     = NULL;
  svm_binary->D           = NULL;

  svm_binary->svm_mod     = 1;
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
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetSeparatingHyperplane_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetSeparatingHyperplane_C",NULL) );
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
  SVM_Binary *svm_binary = (SVM_Binary *) svm;

  PetscInt   N_eq,N_all;
  PetscBool  isascii;

  PetscFunctionBegin;
  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&isascii) );
  if (isascii) {
    N_all = svm_binary->N_all;
    N_eq  = svm_binary->N_eq;
    TRY( PetscViewerASCIIPrintf(v,"SVM: %d of %d test samples classified correctly (%.2f%%)\n",N_eq,N_all,((PetscReal)N_eq)/((PetscReal)N_all)*100.0) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetTrainingDataset"
PetscErrorCode SVMSetTrainingDataset_Binary(SVM svm,Mat Xt_training,Vec y_training)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

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

  svm->setupcalled = PETSC_FALSE;
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
PetscErrorCode SVMCreateQPS_Binary_Private(SVM svm,QPS *qps)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;
  PetscReal  rtol,divtol,max_eig_tol;
  PetscInt   max_it,max_eig_it;
  QPS        qps_inner;

  PetscFunctionBegin;
  rtol        = 1e-1;
  divtol      = 1e100;
  max_it      = 10000;
  max_eig_it  = 100;
  max_eig_tol = 1e-5;

  TRY( QPSDestroy(&svm_binary->qps) );
  TRY( QPSCreate(PetscObjectComm((PetscObject)svm),&qps_inner) );

  TRY( QPSSetType(qps_inner,QPSMPGP) );
  TRY( QPSSetTolerances(qps_inner,rtol,PETSC_DEFAULT,divtol,max_it) );
  TRY( QPSMPGPSetOperatorMaxEigenvalueTolerance(qps_inner,max_eig_tol) );
  TRY( QPSMPGPSetOperatorMaxEigenvalueIterations(qps_inner,max_eig_it) );
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
#define __FUNCT__ "SVMSetUp_Binary"
PetscErrorCode SVMSetUp_Binary(SVM svm)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  QP        qp;
  QPS       qps;

  Mat       H,HpE,mats[2];
  Vec       e,x_init;

  PetscReal C;
  Vec       lb,ub;

  Mat       Xt_training,X_training;
  Vec       y;

  PetscInt    n,m,N;
  SVMLossType loss_type;

  PetscBool   svm_monitor_set;
  void        *mctx;  /* monitor context */

  PetscFunctionBegin;
  if (svm->setupcalled) PetscFunctionReturn(0);
  TRY( SVMGetLossType(svm,&loss_type) );
  TRY( SVMGetC(svm,&C) );

  if (svm->warm_start && svm->posttraincalled) {
    if (loss_type == SVM_L1)
    {
      TRY( SVMGetQPS(svm,&qps) );
      TRY( QPSGetQP(qps,&qp) );
      TRY( QPGetBox(qp,NULL,&ub) );
      TRY( VecSet(ub,C) );
    } else {
      TRY( MatScale(svm_binary->D,svm->C_old) );
      TRY( MatScale(svm_binary->D,1. / C) );
    }
    svm->posttraincalled = PETSC_FALSE;
    svm->setupcalled = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  if (C == -1.0) {
    TRY( SVMGridSearch(svm) );
    TRY( SVMGetC(svm,&C) );
  }
  TRY( SVMGetTrainingDataset(svm,&Xt_training,NULL) );

  TRY( PetscOptionsHasName(NULL,((PetscObject) svm)->prefix,"-svm_monitor",&svm_monitor_set) );

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

  /* Create box constraints */
  TRY( VecDuplicate(y,&lb) );  /* create lower bound constraint vector */
  TRY( VecSet(lb,0.) );

  if (loss_type == SVM_L1) {
    TRY( VecDuplicate(lb,&ub) );
    TRY( VecSet(ub,C) );
  } else {
    /* 1 / 2t = C / 2 => t = 1 / C */
    /* H = H + t * I */
    /* https://link.springer.com/article/10.1134/S1054661812010129 */
    /* http://www.lib.kobe-u.ac.jp/repository/90000225.pdf */

    TRY( MatGetLocalSize(H,&m,&n) );
    TRY( MatGetSize(H,&N,NULL) );
    TRY( MatCreateIdentity(PetscObjectComm((PetscObject) svm),m,n,N,&svm_binary->D) );
    TRY( MatScale(svm_binary->D,1. / C) );

    mats[0] = svm_binary->D;
    mats[1] = H;
    TRY( MatCreateSum(PetscObjectComm((PetscObject)svm),2,mats,&HpE) );
    TRY( MatDestroy(&H) );
    H  = HpE;
    ub = NULL;
  }

  TRY( VecDuplicate(lb,&x_init) );
  TRY( VecSet(x_init,C - 10 * PETSC_MACHINE_EPSILON) );

  TRY( QPSetInitialVector(qp,x_init) );
  TRY( VecDestroy(&x_init) );

  TRY( QPSetOperator(qp,H) );
  TRY( QPSetBox(qp,lb,ub) );

  if (svm_monitor_set) {
    TRY( QPSMonitorCancel(qps) );
    TRY( SVMMonitorCreateMtx_Binary(&mctx,svm) );
    TRY( QPSMonitorSet(qps,SVMMonitorDefault_Binary,mctx,SVMMonitorDestroyMtx_Binary) );
  }

  if (svm->setfromoptionscalled) {
    TRY( QPSSetFromOptions(qps) );
  }

  TRY( QPSSetUp(qps) );

  /* decreasing reference counts */
  TRY( MatDestroy(&X_training) );
  TRY( MatDestroy(&H) );
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
#define __FUNCT__ "SVMReconstructHyperplane_Binary_Private"
PetscErrorCode SVMReconstructHyperplane_Binary_Private(SVM svm,Vec *w,PetscReal *b,PetscBool callpostsolve)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  QPS qps;
  QP  qp;
  Vec ub;

  Mat Xt;
  Vec x,y,yx,y_sv,t,w_inner,zeros;
  Vec Xtw,Xtw_sv;

  IS        is_sv;
  PetscInt  nsv;
  PetscReal b_inner;

  PetscFunctionBegin;
  TRY( SVMGetQPS(svm,&qps) );
  if (callpostsolve) {
    TRY( QPSPostSolve(qps) );
  }
  TRY( QPSGetQP(qps,&qp) );

  TRY( SVMGetTrainingDataset(svm,&Xt,NULL) );
  y = svm_binary->y_inner;
  TRY( QPGetBox(qp,NULL,&ub) );

  /* Reconstruction of hyperplane normal */
  TRY( QPGetSolutionVector(qp,&x) );
  TRY( VecDuplicate(x,&yx) );

  TRY( VecPointwiseMult(yx,y,x) ); /* yx = y.*x */
  TRY( MatCreateVecs(Xt,&w_inner,NULL) );
  TRY( MatMultTranspose(Xt,yx,w_inner) ); /* w = (X^t)^t * yx = X * yx */

  /* Reconstruction of the hyperplane bias */
  TRY( VecDuplicate(x,&zeros) );
  TRY( VecZeroEntries(zeros) );

  if (svm->loss_type == SVM_L1) {
    TRY( VecWhichBetween(zeros,x,ub,&is_sv) );
  } else {
    TRY( VecWhichGreaterThan(x,zeros,&is_sv) );
  }
  TRY( ISGetSize(is_sv,&nsv) );

  TRY( MatCreateVecs(Xt,NULL,&Xtw) );
  TRY( MatMult(Xt,w_inner,Xtw) );

  TRY( VecGetSubVector(y,is_sv,&y_sv) );     /* y_sv = y(is_sv) */
  TRY( VecGetSubVector(Xtw,is_sv,&Xtw_sv) ); /* Xtw_sv = Xt(is_sv) */
  TRY( VecDuplicate(y_sv,&t) );
  TRY( VecWAXPY(t,-1.,Xtw_sv,y_sv) );
  TRY( VecRestoreSubVector(y,is_sv,&y_sv) );
  TRY( VecRestoreSubVector(Xtw,is_sv,&Xtw_sv) );
  TRY( VecSum(t,&b_inner) );

  b_inner /= nsv;

  *w = w_inner;
  *b = b_inner;

  TRY( VecDestroy(&zeros) );
  TRY( VecDestroy(&Xtw) );
  TRY( VecDestroy(&yx) );
  TRY( VecDestroy(&t) );
  TRY( ISDestroy(&is_sv) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetSeparatingHyperplane_Binary"
PetscErrorCode SVMSetSeparatingHyperplane_Binary(SVM svm,Vec w,PetscReal b)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  PetscCheckSameComm(svm,1,w,2);

  TRY( PetscObjectReference((PetscObject) w) );
  svm_binary->w = w;
  svm_binary->b = b;
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
#define __FUNCT__ "SVMPostTrain_Binary"
PetscErrorCode SVMPostTrain_Binary(SVM svm)
{
  Vec       w;
  PetscReal b;

  PetscFunctionBegin;
  if (svm->posttraincalled) PetscFunctionReturn(0);

  TRY( SVMReconstructHyperplane_Binary_Private(svm,&w,&b,PETSC_TRUE) );
  TRY( SVMSetSeparatingHyperplane(svm,w,b) );
  TRY( VecDestroy(&w) );

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
  PetscBool flg;

  PetscFunctionBeginI;
  ierr = PetscObjectOptionsBegin((PetscObject) svm);CHKERRQ(ierr);
  TRY( PetscOptionsInt("-svm_binary_mod","","SVMSetMod",svm_binary->svm_mod,&svm_mod,&flg) );
  if (flg) {
    TRY( SVMSetMod(svm,svm_mod) );
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturnI(0);
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
#define __FUNCT__ "SVMTest_Binary"
PetscErrorCode SVMTest_Binary(SVM svm,Mat Xt_test,Vec y_known,PetscInt *N_all,PetscInt *N_eq)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm;

  Vec y;
  IS  is_eq;

  PetscViewer       v;
  PetscViewerFormat format;
  PetscBool         view;

  PetscFunctionBegin;
  TRY( SVMPredict(svm,Xt_test,&y) );
  TRY( VecWhichEqual(y,y_known,&is_eq) );
  TRY( VecGetSize(y,N_all) );
  TRY( ISGetSize(is_eq,N_eq) );
  TRY( VecDestroy(&y) );
  TRY( ISDestroy(&is_eq) );

  svm_binary->N_all = *N_all;
  svm_binary->N_eq  = *N_eq;

  TRY( PetscOptionsGetViewer(((PetscObject)svm)->comm,((PetscObject)svm)->prefix,"-svm_view",&v,&format,&view) );

  if (view) {
    TRY( PetscViewerPushFormat(v,format) );
    TRY( SVMView(svm,v) );
    TRY( PetscViewerPopFormat(v) );
    TRY( PetscViewerDestroy(&v) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearch_Binary"
PetscErrorCode SVMGridSearch_Binary(SVM svm)
{
  MPI_Comm    comm;
  PetscMPIInt rank;

  PetscReal logC_min,logC_max,logC_base;
  PetscReal C_min;

  PetscReal *c_arr,*score;
  PetscReal C,score_max;
  PetscInt  n,i;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject) svm,&comm) );
  TRY( MPI_Comm_rank(comm,&rank) );

  TRY( SVMGetLogCMin(svm,&logC_min) );
  TRY( SVMGetLogCMax(svm,&logC_max) );
  TRY( SVMGetLogCBase(svm,&logC_base) );

  C_min = PetscPowReal(logC_base,logC_min);

  n = (PetscInt) (logC_max - logC_min + 1);
  TRY( PetscMalloc1(n,&c_arr) );
  TRY( PetscMalloc1(n,&score) );

  c_arr[0] = C_min;
  for (i = 1; i < n; ++i) {
    c_arr[i] = c_arr[i-1] * logC_base;
  }
  TRY( PetscMemzero(score,n * sizeof(PetscReal)) );

  TRY( SVMCrossValidation(svm,c_arr,n,score) );

  C = c_arr[0];
  score_max = score[0];
  for (i = 1; i < n; ++i) {
    if (score[i] > score_max) {
      score_max = score[i];
      C = c_arr[i];
    }
  }

  svm->C = C;
  TRY( PetscFree(c_arr) );
  TRY( PetscFree(score) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMCrossValidation_Binary"
PetscErrorCode SVMCrossValidation_Binary(SVM svm,PetscReal c_arr[],PetscInt m,PetscReal score[])
{
  MPI_Comm comm;
  SVM cross_svm;

  QP  qp;
  QPS qps;

  Mat      Xt,Xt_training,Xt_test;
  Vec      x,y,y_training,y_test;

  PetscInt lo,hi,first,n;
  PetscInt i,j,nfolds;

  IS       is_training,is_test;
  PetscInt N_eq,N_all;

  const char *prefix;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject) svm,&comm) );
  TRY( SVMGetNfolds(svm,&nfolds) );

  TRY( SVMGetTrainingDataset(svm,&Xt,&y) );
  TRY( MatGetOwnershipRange(Xt,&lo,&hi) );

  TRY( SVMCreate(PetscObjectComm((PetscObject)svm),&cross_svm) );
  TRY( SVMSetType(cross_svm,SVM_BINARY) );
  TRY( SVMGetOptionsPrefix(svm,&prefix) );
  TRY( SVMAppendOptionsPrefix(cross_svm,"cross_") );
  TRY( SVMSetFromOptions(cross_svm) );

  TRY( ISCreate(PetscObjectComm((PetscObject) svm),&is_test) );
  TRY( ISSetType(is_test,ISSTRIDE) );

  for (i = 0; i < nfolds; ++i) {
    first = lo + i - 1;
    if (first < lo) first += nfolds;
    n = (hi + nfolds - first - 1) / nfolds;

    TRY( ISStrideSetStride(is_test,n,first,nfolds) );
    TRY( MatCreateSubMatrix(Xt,is_test,NULL,MAT_INITIAL_MATRIX,&Xt_test) );
    TRY( VecGetSubVector(y,is_test,&y_test) );

    TRY( ISComplement(is_test,lo,hi,&is_training) );
    TRY( MatCreateSubMatrix(Xt,is_training,NULL,MAT_INITIAL_MATRIX,&Xt_training) );
    TRY( VecGetSubVector(y,is_training,&y_training) );

    TRY( SVMSetTrainingDataset(cross_svm,Xt_training,y_training) );
    for (j = 0; j < m; ++j) {
      TRY( SVMSetC(cross_svm,c_arr[j]) );

      if (!cross_svm->warm_start) {
        TRY( SVMGetQPS(cross_svm,&qps) );
        TRY( QPSGetSolvedQP(qps,&qp) );
        if (qp) {
          Vec ub;
          TRY( QPGetSolutionVector(qp,&x) );
          TRY( VecSet(x,c_arr[j] - 10 * PETSC_MACHINE_EPSILON) );
          TRY( QPGetBox(qp,NULL,&ub) );
          if (ub) { TRY( VecSet(ub,c_arr[j]) ); }
        }
        cross_svm->posttraincalled = PETSC_FALSE;
      }

      TRY( SVMTrain(cross_svm) );
      TRY( SVMTest(cross_svm,Xt_test,y_test,&N_all,&N_eq) );

      score[j] += ((PetscReal) N_eq) / ((PetscReal) N_all);
    }

    TRY( SVMReset(cross_svm) );
  }

  for (i = 0; i < m; ++i) score[i] /= (PetscReal) nfolds;

  TRY( ISDestroy(&is_test) );
  TRY( SVMDestroy(&cross_svm) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMCreate_Binary"
PetscErrorCode SVMCreate_Binary(SVM svm)
{
  SVM_Binary *svm_binary;

  PetscFunctionBegin;
  TRY( PetscNewLog(svm,&svm_binary) );
  svm->data = (void *) svm_binary;

  svm_binary->w           = NULL;
  svm_binary->b           = PETSC_INFINITY;
  svm_binary->qps         = NULL;
  svm_binary->Xt_training = NULL;
  svm_binary->D           = NULL;
  svm_binary->y_training  = NULL;
  svm_binary->y_inner     = NULL;

  svm_binary->svm_mod     = 1;

  TRY( PetscMemzero(svm_binary->y_map,2 * sizeof(PetscScalar)) );

  svm->ops->setup           = SVMSetUp_Binary;
  svm->ops->reset           = SVMReset_Binary;
  svm->ops->destroy         = SVMDestroy_Binary;
  svm->ops->setfromoptions  = SVMSetFromOptions_Binary;
  svm->ops->train           = SVMTrain_Binary;
  svm->ops->posttrain       = SVMPostTrain_Binary;
  svm->ops->predict         = SVMPredict_Binary;
  svm->ops->test            = SVMTest_Binary;
  svm->ops->crossvalidation = SVMCrossValidation_Binary;
  svm->ops->gridsearch      = SVMGridSearch_Binary;
  svm->ops->view            = SVMView_Binary;

  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetTrainingDataset_C",SVMSetTrainingDataset_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetTrainingDataset_C",SVMGetTrainingDataset_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetQPS_C",SVMSetQPS_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetQPS_C",SVMGetQPS_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetMod_C",SVMSetMod_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetMod_C",SVMGetMod_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetSeparatingHyperplane_C",SVMSetSeparatingHyperplane_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetSeparatingHyperplane_C",SVMGetSeparatingHyperplane_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetOptionsPrefix_C",SVMSetOptionsPrefix_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetOptionsPrefix_C",SVMGetOptionsPrefix_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMAppendOptionsPrefix_C",SVMAppendOptionsPrefix_Binary) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMMonitorCreateMtx_Binary"
PetscErrorCode SVMMonitorCreateMtx_Binary(void **mctx,SVM svm)
{
  SVM_Binary_mctx *mctx_inner;

  PetscFunctionBegin;
  PetscValidPointer(mctx,1);
  PetscValidHeaderSpecific(svm,SVM_CLASSID,2);

  TRY( PetscNew(&mctx_inner) );
  mctx_inner->svm_inner = svm;
  TRY( PetscObjectReference((PetscObject) svm) );
  *mctx = mctx_inner;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMMonitorDestroyMtx_Binary"
PetscErrorCode SVMMonitorDestroyMtx_Binary(void **mctx)
{
  SVM_Binary_mctx *mctx_inner = (SVM_Binary_mctx *) *mctx;
  SVM              svm_inner;

  PetscFunctionBegin;
  svm_inner  = mctx_inner->svm_inner;
  TRY( PetscObjectDereference((PetscObject) svm_inner) );
  mctx_inner->svm_inner = NULL;
  TRY( PetscFree(mctx_inner) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMMonitorDefault_Binary"
PetscErrorCode SVMMonitorDefault_Binary(QPS qps,PetscInt it,PetscReal rnorm,void *mctx) {
  MPI_Comm comm;
  SVM svm_inner;

  QP qp;

  Vec       lb,ub,w_inner,x;
  PetscReal b_inner,norm_w,margin;

  IS       is_sv;
  PetscInt nsv;

  SVMLossType loss_type;

  PetscViewer v;

  PetscFunctionBegin;
  svm_inner = ((SVM_Binary_mctx *) mctx)->svm_inner;
  comm = PetscObjectComm((PetscObject) svm_inner);

  TRY( SVMGetLossType(svm_inner,&loss_type) );

  TRY( SVMReconstructHyperplane_Binary_Private(svm_inner,&w_inner,&b_inner,PETSC_FALSE) );
  TRY( VecNorm(w_inner,NORM_2,&norm_w) );
  margin = 2.0 / norm_w;

  /* Get number of support vectors */
  TRY( SVMGetQPS(svm_inner,&qps) );
  TRY( QPSGetQP(qps,&qp) );
  TRY( QPGetBox(qp,&lb,&ub) );
  TRY( QPGetSolutionVector(qp,&x) );

  if (loss_type == SVM_L1) {
    TRY( QPGetBox(qp,&lb,&ub) );
    TRY( VecWhichBetween(lb,x,ub,&is_sv) );
  } else {
    TRY( VecWhichGreaterThan(x,lb,&is_sv));
  }
  TRY( ISGetSize(is_sv,&nsv) );

  v = PETSC_VIEWER_STDOUT_(comm);
  TRY( PetscViewerASCIIPrintf(v,"%3D SVM ||w||=%.10e",it,norm_w) );
  TRY( PetscViewerASCIIPrintf(v,",\tmargin=%.10e",margin) );
  TRY( PetscViewerASCIIPrintf(v,",\tbias=%.10e",b_inner) );
  TRY( PetscViewerASCIIPrintf(v,",\tNSV=%3D\n",nsv) );

  TRY( ISDestroy(&is_sv) );
  PetscFunctionReturn(0);
}