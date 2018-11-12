
#include <permon/private/svmimpl.h>

PetscClassId SVM_CLASSID;

const char *const SVMLossTypes[]={"L1","L2","SVMLossType","SVM_",0};

typedef struct {
    PetscBool autoPostSolve;
    PetscBool setupcalled;
    PetscBool setfromoptionscalled;

    PetscReal C, LogCMin, LogCMax, LogCBase;
    PetscInt nfolds;
    SVMLossType loss_type;

    PetscBool warm_start;

    Mat Xt_training;
    Vec y_training;
    Vec y_inner;
    PetscScalar y_map[2];
    Mat D;

    Vec w;
    PetscScalar b;

    QPS qps;
} SVM_Binary;

static PetscErrorCode SVMQPSConvergedTrainRateCreate(SVM svm,void **ctx);
static PetscErrorCode SVMQPSConvergedTrainRateDestroy(void *ctx);
static PetscErrorCode SVMQPSConvergedTrainRate(QPS qps,QP qp,PetscInt it,PetscReal rnorm,KSPConvergedReason *reason,void *cctx);

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

  PetscMemzero(svm_binary->y_map,2*sizeof(PetscScalar) );
  svm_binary->b = PETSC_INFINITY;
  
  svm_binary->w           = NULL;
  svm_binary->Xt_training = NULL;
  svm_binary->y_training  = NULL;
  svm_binary->y_inner     = NULL;
  svm_binary->D           = NULL;
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

  TRY( QPSDestroy(&svm_binary->qps) );
  TRY( SVMDestroyDefault(svm) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMView_Binary"
PetscErrorCode SVMView_Binary(SVM svm, PetscViewer v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetLogCMin"
/*@
   SVMSetC - Sets the minimal C parameter value.

   Input Parameter:
+  svm - the SVM
-  LogCMin - minimal C parameter value
@*/
PetscErrorCode SVMSetLogCMin(SVM svm, PetscReal LogCMin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(svm, LogCMin, 2);
  svm->LogCMin = LogCMin;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetLogCMin"
/*@
   SVMGetC - Gets the minimal C parameter value.

   Input Parameter:
.  svm - the SVM

   Output Parameter:
.  LogCMin - minimal C parameter value
@*/
PetscErrorCode SVMGetLogCMin(SVM svm, PetscReal *LogCMin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidRealPointer(LogCMin, 2);
  *LogCMin = svm->LogCMin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetLogCBase"
/*@
   SVMSetLogCBase - Sets the step C value.

   Input Parameter:
+  svm - the SVM
-  LogCBase - step C value
@*/
PetscErrorCode SVMSetLogCBase(SVM svm, PetscReal LogCBase)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(svm, LogCBase, 2);

  if (LogCBase <= 0) FLLOP_SETERRQ(((PetscObject) svm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Argument must be positive");
  svm->LogCBase = LogCBase;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetLogCBase"
/*@
   SVMGetC - Gets the step C value.

   Input Parameter:
.  svm - the SVM

   Output Parameter:
.  LogCBase - step C value
@*/
PetscErrorCode SVMGetLogCBase(SVM svm, PetscReal *LogCBase)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidRealPointer(LogCBase, 2);
  *LogCBase = svm->LogCBase;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetLogCMax"
/*@
   SVMSetLogCMax - Sets the max C parameter value.

   Input Parameter:
+  svm - the SVM
-  LogCMax - max C parameter value
@*/
PetscErrorCode SVMSetLogCMax(SVM svm, PetscReal LogCMax)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(svm, LogCMax, 2);
  svm->LogCMax = LogCMax;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetLogCMax"
/*@
   SVMGetLogCMax - Gets the max C parameter value.

   Input Parameter:
.  svm - the SVM

   Output Parameter:
.  LogCMax - maximal C parameter value
@*/
PetscErrorCode SVMGetLogCMax(SVM svm, PetscReal *LogCMax)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidRealPointer(LogCMax, 2);
  *LogCMax = svm->LogCMax;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetNfolds"
/*@
   SVMSetC - Sets the C parameter.

   Input Parameter:
+  svm - the SVM
-  C - C parameter
@*/
PetscErrorCode SVMSetNfolds(SVM svm, PetscInt nfolds)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(svm, nfolds, 2);

  if (nfolds < 2) FLLOP_SETERRQ(((PetscObject) svm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Argument must be greater than 1.");
  svm->nfolds = nfolds;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetNfolds"
/*@
   SVMGetNfolds - Gets the number of folds.

   Input Parameter:
.  svm - the SVM

   Output Parameter:
.  nfolds - number of folds
@*/
PetscErrorCode SVMGetNfolds(SVM svm, PetscInt *nfolds)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidRealPointer(nfolds, 2);
  *nfolds = svm->nfolds;
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

  svm_binary->setupcalled = PETSC_FALSE;
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
#define __FUNCT__ "SVMSetLossType"
/*@
   SVMSetLossType - Sets the type of the loss function.

   Logically Collective on SVM

   Input Parameters:
+  svm - the SVM
-  type - type of loss function

   Level: beginner

.seealso PermonSVMType, SVMGetLossType()
@*/
PetscErrorCode SVMSetLossType(SVM svm, SVMLossType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(svm, type, 2);
  if (type != svm->loss_type) {
    svm->loss_type = type;
    svm->setupcalled = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetLossType"
/*@
   SVMGetLossType - Gets the type of the loss function.

   Not Collective

   Input Parameter:
.  svm - the SVM

   Output Parameter:
.  type - type of loss function

   Level: beginner

.seealso PermonSVMType, SVMSetLossType()
@*/
PetscErrorCode SVMGetLossType(SVM svm, SVMLossType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidPointer(type, 2);
  *type = svm->loss_type;
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

  PetscInt  n,m,N;

  PetscFunctionBegin;
  if (svm_binary->setupcalled) PetscFunctionReturn(0);

  TRY( SVMGetC(svm,&C) );
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

  /* Create box constraints */
  TRY( VecDuplicate(y,&lb) );  /* create lower bound constraint vector */
  TRY( VecSet(lb,0.) );

  if (svm->loss_type == SVM_L1) {
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
#define __FUNCT__ "SVMSetOptionsPrefix"
PetscErrorCode SVMSetOptionsPrefix(SVM svm,const char prefix[])
{
  QPS qps;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( PetscObjectSetOptionsPrefix((PetscObject)svm,prefix) );
  TRY( SVMGetQPS(svm,&qps) );
  TRY( QPSSetOptionsPrefix(qps,prefix) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMAppendOptionsPrefix"
PetscErrorCode SVMAppendOptionsPrefix(SVM svm,const char prefix[])
{
  QPS qps;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( PetscObjectAppendOptionsPrefix((PetscObject)svm,prefix) );
  TRY( SVMGetQPS(svm,&qps) );
  TRY( QPSAppendOptionsPrefix(qps,prefix) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetOptionsPrefix"
PetscErrorCode SVMGetOptionsPrefix(SVM svm,const char *prefix[])
{
  QPS qps;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( PetscObjectGetOptionsPrefix((PetscObject)svm,prefix) );
  TRY(SVMGetQPS(svm,&qps) );
  TRY( QPSGetOptionsPrefix(qps,prefix) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMTrain_Binary"
PetscErrorCode SVMTrain_Binary(SVM svm)
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  PetscFunctionBegin;
  TRY( PetscPrintf(PetscObjectComm((PetscObject)svm),"### PermonSVM:   train with loss_type %s, C = %.2e\n",SVMLossTypes[svm->loss_type],svm->C) );
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
PetscErrorCode SVMReconstructHyperplane_Binary_Private(SVM svm,Vec *w,PetscReal *b)
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
  TRY( QPSPostSolve(qps) );
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

  if (svm_binary->loss_type == SVM_L1) {
    TRY( VecWhichBetween(x,zeros,ub,&is_sv) );
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

  TRY( SVMReconstructHyperplane_Binary_Private(svm,&w,&b) );
  TRY( SVMSetSeparatingHyperplane(svm,w,b) );
  TRY( VecDestroy(&w) );

  svm->posttraincalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetFromOptions_Binary"
PetscErrorCode SVMSetFromOptions_Binary(PetscOptionItems *PetscOptionsObject,SVM svm)
{
  PetscReal C, LogCMin, LogCMax, LogCBase;
  PetscInt nfolds;
  PetscBool flg, flg1;
  SVMLossType loss_type;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  _fllop_ierr = PetscObjectOptionsBegin((PetscObject)svm);CHKERRQ(_fllop_ierr);
  TRY( PetscOptionsReal("-svm_C","Set SVM C (C).","SVMSetC",svm->C,&C,&flg) );
  if (flg) TRY(SVMSetC(svm, C) );
  TRY( PetscOptionsReal("-svm_logC_min","Set SVM minimal C value (LogCMin).","SVMSetLogCMin",svm->LogCMin,&LogCMin,&flg) );
  if (flg) TRY(SVMSetLogCMin(svm, LogCMin) );
  TRY( PetscOptionsReal("-svm_logC_max","Set SVM maximal C value (LogCMax).","SVMSetLogCMax",svm->LogCMax,&LogCMax,&flg) );
  if (flg) TRY(SVMSetLogCMax(svm, LogCMax) );
  TRY( PetscOptionsReal("-svm_logC_base","Set power base of SVM parameter C (LogCBase).","SVMSetLogCBase",svm->LogCBase,&LogCBase,&flg) );
  if (flg) TRY(SVMSetLogCBase(svm, LogCBase) );
  TRY( PetscOptionsInt("-svm_nfolds","Set number of folds (nfolds).","SVMSetNfolds",svm->nfolds,&nfolds,&flg) );
  if (flg) TRY(SVMSetNfolds(svm, nfolds) );
  TRY( PetscOptionsEnum("-svm_loss_type","Specify the loss function for soft-margin SVM (non-separable samples).","SVMSetNfolds",SVMLossTypes,(PetscEnum)svm->loss_type,(PetscEnum*)&loss_type,&flg) );
  if (flg) TRY(SVMSetLossType(svm, loss_type) );
  TRY( PetscOptionsBool("-svm_warm_start","Specify whether warm start is used in cross-validation.","SVMSetWarmStart",svm->warm_start,&flg1,&flg) );
  if (flg) TRY(SVMSetWarmStart(svm, flg1) );
  svm->setfromoptionscalled = PETSC_TRUE;
  _fllop_ierr = PetscOptionsEnd();CHKERRQ(_fllop_ierr);
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
PetscErrorCode SVMTest_Binary(SVM svm, Mat Xt_test, Vec y_known, PetscInt *N_all, PetscInt *N_eq)
{
  Vec y;
  IS is_eq;

  PetscFunctionBeginI;
  TRY( SVMPredict(svm, Xt_test, &y) );
  TRY( VecWhichEqual(y,y_known,&is_eq) );
  TRY( VecGetSize(y,N_all) );
  TRY( ISGetSize(is_eq,N_eq) );
  TRY( VecDestroy(&y) );
  TRY( ISDestroy(&is_eq) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMCreate_Binary"
PetscErrorCode SVMCreate_Binary(SVM svm)
{
  SVM_Binary *svm_binary;

  PetscFunctionBegin;
  TRY( PetscNewLog(svm,&svm_binary) );
  svm->data = (void *) svm_binary;

  svm_binary->loss_type   = SVM_L2;
  svm_binary->w           = NULL;
  svm_binary->b           = PETSC_INFINITY;
  svm_binary->qps         = NULL;
  svm_binary->Xt_training = NULL;
  svm_binary->D           = NULL;
  svm_binary->y_training  = NULL;
  svm_binary->y_inner     = NULL;

  TRY( PetscMemzero(svm->y_map,2*sizeof(PetscScalar)) );

  svm->ops->setup          = SVMSetUp_Binary;
  svm->ops->reset          = SVMReset_Binary;
  svm->ops->destroy        = SVMDestroy_Binary;
  svm->ops->setfromoptions = SVMSetFromOptions_Binary;
  svm->ops->train          = SVMTrain_Binary;
  svm->ops->posttrain      = SVMPostTrain_Binary;
  svm->ops->predict        = SVMPredict_Binary;
  svm->ops->view           = SVMView_Binary;

  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetTrainingDataset_C",SVMSetTrainingDataset_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetTrainingDataset_C",SVMGetTrainingDataset_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetQPS_C",SVMSetQPS_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetQPS_C",SVMGetQPS_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMSetSeparatingHyperplane_C",SVMSetSeparatingHyperplane_Binary) );
  TRY( PetscObjectComposeFunction((PetscObject) svm,"SVMGetSeparatingHyperplane_C",SVMGetSeparatingHyperplane_Binary) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetWarmStart"
/*@
   SVMSetWarmStart - Set flag specifying whether warm start is used in cross-validation.
   It is set to PETSC_TRUE by default.

   Logically Collective on SVM

   Input Parameters:
+  svm - the SVM
-  flg - use warm start in cross-validation

   Options Database Keys:
.  -svm_warm_start - use warm start in cross-validation

   Level: advanced

.seealso PermonSVMType, SVMGetLossType()
@*/
PetscErrorCode SVMSetWarmStart(SVM svm, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveBool(svm,flg,2);
  svm->warm_start = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMCrossValidate"
/*@
   SVMCrossValidate -

   Input Parameter:
.  svm - the SVM

@*/
PetscErrorCode SVMCrossValidate(SVM svm)
{
  MPI_Comm comm;
  PetscMPIInt rank;
  SVM cross_svm;
  IS is_test, is_train;
  PetscInt n_examples, n_attributes;  /* PETSC_DEFAULT or PETSC_DECIDE means all */
  PetscInt i, j, i_max;
  PetscInt nfolds, first, n;
  PetscInt lo, hi;
  PetscInt N_all, N_eq, c_count;
  PetscInt its;
  PetscReal C, LogCMin, LogCBase, LogCMax, C_min;
  PetscReal *array_rate = NULL, rate_max, rate;
  PetscReal *array_C = NULL;
  Mat Xt, Xt_test, Xt_train;
  Vec y, y_test, y_train;
  const char *prefix;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( PetscObjectGetComm((PetscObject)svm,&comm) );
  TRY( MPI_Comm_rank(comm, &rank) );
  TRY( SVMGetTrainingDataset(svm,&Xt,&y) );
  TRY( MatGetSize(Xt,&n_examples,&n_attributes) );
  TRY( MatGetOwnershipRange(Xt,&lo,&hi) );
  TRY( SVMGetLogCMin(svm, &LogCMin) );
  TRY( SVMGetLogCBase(svm, &LogCBase) );
  TRY( SVMGetLogCMax(svm, &LogCMax) );
  TRY( SVMGetNfolds(svm, &nfolds) );
  TRY( SVMGetOptionsPrefix(svm, &prefix) );

  if (nfolds > n_examples) FLLOP_SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"number of folds must not be greater than number of examples but %d > %d",nfolds,n_examples);

  C_min = PetscPowReal(LogCBase,LogCMin);
  c_count = LogCMax - LogCMin + 1;
  i=0;
  if (!c_count) FLLOP_SETERRQ(comm,PETSC_ERR_ARG_INCOMP,"LogCMin must be less than or equal to LogCMax");

  TRY( PetscMalloc1(c_count,&array_rate) );
  TRY( PetscMalloc1(c_count,&array_C) );
  TRY( PetscMemzero(array_rate,c_count*sizeof(PetscReal)) );
  array_C[0] = C_min;
  for (i = 1; i < c_count; i++) {
    array_C[i] = array_C[i-1] * LogCBase;
  }
  TRY( PetscPrintf(comm, "### PermonSVM: following values of C will be tested:\n") );
  if (!rank) TRY( PetscRealView(c_count,array_C,PETSC_VIEWER_STDOUT_SELF) );

  TRY( ISCreate(PetscObjectComm((PetscObject)svm),&is_test) );
  TRY( ISSetType(is_test,ISSTRIDE) );

  for (i = 0; i < nfolds; ++i) {
    TRY( PetscPrintf(comm, "### PermonSVM: fold %d of %d\n",i+1,nfolds) );

    first = (lo-1)/nfolds*nfolds + i;
    if (first < lo) first += nfolds;
    n = (hi + nfolds - first - 1)/nfolds;

    TRY( ISStrideSetStride(is_test,n,first,nfolds) );
    TRY( MatCreateSubMatrix(Xt,is_test,NULL,MAT_INITIAL_MATRIX,&Xt_test) );
    TRY( VecGetSubVector(y,is_test,&y_test) );
    TRY( ISComplement(is_test,lo,hi,&is_train) );
    TRY( MatCreateSubMatrix(Xt,is_train,NULL,MAT_INITIAL_MATRIX,&Xt_train) );
    TRY( VecGetSubVector(y,is_train,&y_train) );

    TRY( SVMCreate(PetscObjectComm((PetscObject)svm),&cross_svm) );
    TRY( SVMSetOptionsPrefix(cross_svm,prefix) );
    TRY( SVMAppendOptionsPrefix(cross_svm,"cross_") );
    TRY( SVMSetTrainingDataset(cross_svm,Xt_train,y_train) );
    TRY( SVMSetLossType(cross_svm,svm->loss_type) );
    TRY( SVMSetFromOptions(cross_svm) );
    for (j = 0; j < c_count; ++j) {
      TRY( SVMSetC(cross_svm,array_C[j]) );

      if (!svm->warm_start) {
        QP qp;
        Vec x;
        TRY( QPSGetSolvedQP(cross_svm->qps,&qp) );
        if (qp) {
          TRY( QPGetSolutionVector(qp,&x) );
          TRY( VecZeroEntries(x) );
        }
      }

      TRY( SVMTrain(cross_svm) );
      TRY( SVMTest(cross_svm,Xt_test,y_test,&N_all,&N_eq) );
      rate = ((PetscReal)N_eq) / ((PetscReal)N_all);
      array_rate[j] += rate;

      //TRY( PetscObjectTypeCompare((PetscObject)cross_svm->qps,QPSSMALXE,&flg) );
      /*if (flg) {
        QPS qps_inner;
        TRY( QPSSMALXEGetInnerQPS(cross_svm->qps,&qps_inner) );
        TRY( QPSGetAccumulatedIterationNumber(qps_inner,&its) );
      } else {*/
      TRY( QPSGetIterationNumber(cross_svm->qps,&its) );
      //}
      TRY( PetscPrintf(comm, "### PermonSVM:     %d of %d examples classified correctly (rate %.2f), accumulated rate for C=%.2e is %.2f, %d QPS iterations\n", N_eq, N_all, rate, array_C[j], array_rate[j], its) );
    }
    TRY( SVMDestroy(&cross_svm) );
    TRY( MatDestroy(&Xt_test) );
    TRY( VecRestoreSubVector(y,is_test,&y_test) );
    TRY( MatDestroy(&Xt_train) );
    TRY( VecRestoreSubVector(y,is_train,&y_train) );
    TRY( ISDestroy(&is_train) );
  }

  i_max = 0;
  for(i = 1; i < c_count; ++i) {
    if (array_rate[i] > array_rate[i_max]) {
      i_max = i;
    }
  }
  rate_max = array_rate[i_max] / nfolds;
  C = array_C[i_max];
  TRY( PetscPrintf(comm,"### PermonSVM: selecting C = %.2e with accumulated rate %f and average rate %f based on cross-validation with %d folds\n",C,array_rate[i_max],rate_max,nfolds) );
  TRY( SVMSetC(svm, C) );

  TRY( ISDestroy(&is_test) );
  TRY( PetscFree(array_rate) );
  TRY( PetscFree(array_C) );
  PetscFunctionReturnI(0);
}

typedef struct {
  SVM svm;
  void *defaultCtx;
} ConvergedTrainRateCtx;

#undef __FUNCT__
#define __FUNCT__ "SVMQPSConvergedTrainRateCreate"
static PetscErrorCode SVMQPSConvergedTrainRateCreate(SVM svm,void **ctx)
{
  ConvergedTrainRateCtx *cctx;

  PetscFunctionBegin;
  PetscNew(&cctx);
  *ctx = cctx;
  cctx->svm = svm;
  TRY( QPSConvergedDefaultCreate(&cctx->defaultCtx) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMQPSConvergedTrainRateDestroy"
static PetscErrorCode SVMQPSConvergedTrainRateDestroy(void *ctx)
{
  ConvergedTrainRateCtx *cctx = (ConvergedTrainRateCtx*) ctx;

  PetscFunctionBegin;
  TRY( QPSConvergedDefaultDestroy(cctx->defaultCtx) );
  TRY( PetscFree(cctx) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMQPSConvergedTrainRate"
static PetscErrorCode SVMQPSConvergedTrainRate(QPS qps,QP qp,PetscInt it,PetscReal rnorm,KSPConvergedReason *reason,void *cctx)
{
  SVM svm = ((ConvergedTrainRateCtx*) cctx)->svm;
  void *ctx = ((ConvergedTrainRateCtx*) cctx)->defaultCtx;
  Mat Xt;
  Vec y,y_orig;
  PetscInt N_all, N_eq;
  PetscReal rate,func,margin;
  IS is_sv;
  Vec o, y_sv, Xtw, Xtw_sv, t;
  PetscInt len_sv;
  Vec Yz, z, w;
  PetscScalar b;
  Vec grad,rhs;


  PetscFunctionBegin;
  TRY( SVMGetTrainingDataset(svm, &Xt, &y_orig) );
  y = svm->y_inner;

  /* reconstruct w from dual solution z */
  {
    TRY( QPGetSolutionVector(qp, &z) );
    TRY( VecDuplicate(z, &Yz) );
    TRY( VecPointwiseMult(Yz, y, z) );            /* YZ = Y*z = y.*z */
    TRY( MatCreateVecs(Xt, &w, NULL) );           /* create vector w such that Xt*w works */
    TRY( MatMultTranspose(Xt, Yz, w) );           /* Xt = X^t, w = Xt' * Yz = (X^t)^t * Yz = X * Yz */

    svm->w = w;

    PetscScalar r;
    VecDot(z, y, &r);
    PetscPrintf(MPI_COMM_WORLD, "Be * y = %f\n", r);
  }

  /* reconstruct b from dual solution z */
  {
    TRY( VecDuplicate(z, &o) );
    TRY( VecZeroEntries(o) );
    TRY( MatCreateVecs(Xt, NULL, &Xtw) );

    TRY( VecWhichGreaterThan(z, o, &is_sv) );
    TRY( ISGetSize(is_sv, &len_sv) );
    TRY( MatMult(Xt, w, Xtw) );
    TRY( VecGetSubVector(y, is_sv, &y_sv) );      /* y_sv = y(is_sv) */
    TRY( VecGetSubVector(Xtw, is_sv, &Xtw_sv) );  /* Xtw_sv = Xtw(is_sv) */
    TRY( VecDuplicate(y_sv, &t) );
    TRY( VecWAXPY(t, -1.0, Xtw_sv, y_sv) );       /* t = y_sv - Xtw_sv */
    TRY( VecRestoreSubVector(y, is_sv, &y_sv) );
    TRY( VecRestoreSubVector(Xtw, is_sv, &Xtw_sv) );
    TRY( VecSum(t, &b) );                         /* b = sum(t) */
    b /= len_sv;                                  /* b = b / length(is_sv) */

    svm->b = b;
  }

  TRY( SVMTest(svm, Xt, y_orig, &N_all, &N_eq) );
  rate = ((PetscReal)N_eq)/((PetscReal)N_all);

  /* compute value of the functional !!!MPGP ONLY!!! */
  TRY( QPGetRhs(qp,&rhs) );
  TRY( VecDuplicate(qps->work[3],&grad) );
  TRY( VecWAXPY(grad,-1.0,rhs,qps->work[3]) );
  TRY( VecDot(z,grad,&func) );
  func = .5*func;
  TRY( VecNorm(w,NORM_2,&margin) );
  margin = 2.0/margin;
  TRY( PetscPrintf(PETSC_COMM_WORLD, "it= %d RATE= %.8f% rnorm= %.8e func= %.8e margin= %.8e\n",it,rate*100.0,rnorm,func,margin) );
  TRY( QPSConvergedDefault(qps,qp,it,rnorm,reason,ctx) );

  TRY( ISDestroy(&is_sv) );
  TRY( VecDestroy(&o) );
  TRY( VecDestroy(&t) );
  TRY( VecDestroy(&Yz) );
  TRY( VecDestroy(&Xtw) );
  TRY( VecDestroy(&w) );
  TRY( VecDestroy(&grad) );
  PetscFunctionReturn(0);
}
