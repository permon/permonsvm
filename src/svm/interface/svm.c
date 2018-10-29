
#include <permon/private/svmimpl.h>

PetscClassId SVM_CLASSID;

const char *const SVMLossTypes[]={"L1","L2","SVMLossType","SVM_",0};

static PetscErrorCode SVMQPSConvergedTrainRateCreate(SVM svm,void **ctx);
static PetscErrorCode SVMQPSConvergedTrainRateDestroy(void *ctx);
static PetscErrorCode SVMQPSConvergedTrainRate(QPS qps,QP qp,PetscInt it,PetscReal rnorm,KSPConvergedReason *reason,void *cctx);

#undef __FUNCT__
#define __FUNCT__ "SVMCreate"
/*@
SVMCreate - create instance of support vector machine classifier

Parameters:
+ comm - MPI comm
- svm_out - pointer to created SVM
@*/
PetscErrorCode SVMCreate(MPI_Comm comm, SVM *svm_out)
{
  SVM svm;

  PetscFunctionBegin;
  PetscValidPointer(svm_out, 2);

#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  TRY( SVMInitializePackage() );
#endif
  TRY( PetscHeaderCreate(svm, SVM_CLASSID, "SVM", "SVM Classifier", "SVM", comm, SVMDestroy, SVMView) );

  svm->setupcalled = PETSC_FALSE;
  svm->setfromoptionscalled = PETSC_FALSE;
  svm->autoPostSolve = PETSC_TRUE;
  svm->qps = NULL;

  svm->C = 1e1;
  svm->LogCMin = -3.0;
  svm->LogCBase = 2.0;
  svm->LogCMax = 10.0;
  svm->nfolds = 5;
  svm->loss_type = SVM_L2;

  svm->warm_start = PETSC_FALSE;

  svm->Xt = NULL;
  svm->y = NULL;
  svm->y_inner = NULL;
  svm->D = NULL;
  svm->w = NULL;
  svm->b = PETSC_INFINITY;

  TRY( PetscMemzero(svm->y_map,2*sizeof(PetscScalar)) );

  *svm_out = svm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMReset"
/*@
   QPReset - Resets a QP context to the QPsetupcalled = 0 state, destroys child, PC, Vecs,  Mats, etc.

   Collective on SVM

   Input Parameter:
.  svm - the SVM
@*/
PetscErrorCode SVMReset(SVM svm)
{
  PetscFunctionBegin;
  TRY( QPSReset(svm->qps) );
  TRY( MatDestroy(&svm->Xt) );
  TRY( VecDestroy(&svm->y) );
  TRY( VecDestroy(&svm->y_inner) );
  TRY( MatDestroy(&svm->D) );
  TRY( PetscMemzero(svm->y_map,2*sizeof(PetscScalar)) );
  TRY( VecDestroy(&svm->w) );

  svm->Xt      = NULL;
  svm->y       = NULL;
  svm->y_inner = NULL;
  svm->D = NULL;
  svm->w = NULL;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SVMDestroy"
/*@
   SVMDestroy - Destroys SVM context.

   Collective on SVM

   Input Parameter:
.  svm - SVM context
@*/
PetscErrorCode SVMDestroy(SVM *svm)
{
  PetscFunctionBegin;
  if (!*svm) PetscFunctionReturn(0);

  PetscValidHeaderSpecific(*svm, SVM_CLASSID, 1);
  if (--((PetscObject) (*svm))->refct > 0) {
      *svm = 0;
      PetscFunctionReturn(0);
  }

  TRY( SVMReset(*svm) );
  TRY( QPSDestroy(&(*svm)->qps) );
  TRY( PetscHeaderDestroy(svm) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMView"
/*@
   SVMView - View information about SVM.

   Input Parameters:
+  svm - the SVM
-  v - visualization context
@*/
PetscErrorCode SVMView(SVM svm, PetscViewer v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetC"
/*@
   SVMSetC - Sets the C parameter.

   Input Parameter:
+  svm - the SVM
-  C - C parameter
@*/
PetscErrorCode SVMSetC(SVM svm, PetscReal C)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(svm, C, 2);

  if (C <= 0 && C != PETSC_DECIDE && C != PETSC_DEFAULT) FLLOP_SETERRQ(((PetscObject) svm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Argument must be positive");
  if (svm->setupcalled) {
    if (svm->loss_type == SVM_L1) {
      Vec ub;
      TRY( QPGetBox(svm->qps->solQP, NULL, &ub) );
      TRY( VecSet(ub, C) );

    } else {
      TRY( MatScale(svm->D,C/svm->C) );
    }
  }
  svm->C = C;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetC"
/*@
   SVMGetC - Gets the C parameter.

   Input Parameter:
.  svm - the SVM

   Output Parameter:
.  C - C parameter
@*/
PetscErrorCode SVMGetC(SVM svm, PetscReal *C)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidRealPointer(C, 2);
  *C = svm->C;
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
#define __FUNCT__ "SVMSetTrainingSamples"
/*@
   SVMSetTrainingSamples - Sets the training samples.

   Input Parameter:
+  svm - the SVM
.  Xt - samples data
-  y -
@*/
PetscErrorCode SVMSetTrainingSamples(SVM svm, Mat Xt, Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidHeaderSpecific(Xt, MAT_CLASSID, 2);
  PetscCheckSameComm(svm, 1, Xt, 2);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 3);
  PetscCheckSameComm(svm, 1, y, 3);

  TRY( MatDestroy(&svm->Xt) );
  svm->Xt = Xt;
  TRY( PetscObjectReference((PetscObject) Xt) );

  TRY( VecDestroy(&svm->y) );
  svm->y = y;
  TRY( PetscObjectReference((PetscObject) y) );

  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetTrainingSamples"
/*@
   SVMSetTrainingSamples - Sets the training samples.

   Input Parameter:
.  svm - the SVM

   Output Parameter:
+  Xt - samples data
-  y -
@*/
PetscErrorCode SVMGetTrainingSamples(SVM svm, Mat *Xt, Vec *y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  if (Xt) {
    PetscValidPointer(Xt, 2);
    *Xt = svm->Xt;
  }
  if (y) {
    PetscValidPointer(y, 3);
    *y = svm->y;
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
#define __FUNCT__ "SVMSetQPS"
/*@
   SVMSetQPS - Sets the QPS.

   Input Parameter:
.  svm - the SVM

   Output Parameter:
.  qps -
@*/
PetscErrorCode SVMSetQPS(SVM svm, QPS qps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 2);
  PetscCheckSameComm(svm, 1, qps, 2);

  TRY( QPSDestroy(&svm->qps) );
  svm->qps = qps;
  TRY( PetscObjectReference((PetscObject) qps) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetQPS"
/*@
   SVMGetQPS - Gets the QPS.

   Input Parameter:
.  svm - the SVM

   Output Parameter:
.  qps -
@*/
PetscErrorCode SVMGetQPS(SVM svm, QPS *qps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidPointer(qps,2);
  if (!svm->qps) {
    QPS qps; //qps_inner;
    //Tao tao;

    TRY( QPSCreate(PetscObjectComm((PetscObject)svm),&qps) );

    /* set default solver */
    TRY( QPSSetType(qps,QPSMPGP) );
    //TRY( QPSSMALXEGetInnerQPS(qps,&qps_inner) );
    //TRY( QPSSetType(qps_inner,QPSTAO) );
    //TRY( QPSTaoGetTao(qps_inner,&tao) );
    //TRY( TaoSetType(tao,TAOBLMVM) );

    /* set default solver settings */
    TRY( QPSSetTolerances(qps,1e-1,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT) );
    //TRY( QPSSMALXESetM1Initial(qps,1.0,QPS_ARG_MULTIPLE) );
    //TRY( QPSSMALXESetRhoInitial(qps,1.0,QPS_ARG_MULTIPLE) );

    svm->qps = qps;
  }
  *qps = svm->qps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetUp_Remapy_Private"
/* map y to -1,1 values if needed */
static PetscErrorCode SVMSetUp_Remapy_Private(SVM svm)
{
  Vec y;
  PetscScalar min,max;

  PetscFunctionBegin;
  TRY( SVMGetTrainingSamples(svm, NULL, &y) );
  TRY( VecMin(y,NULL,&min) );
  TRY( VecMax(y,NULL,&max) );
  if (min == -1.0 && max == 1.0) {
    svm->y_inner = y;
    TRY( PetscObjectReference((PetscObject)y) );
    svm->y_map[0] = -1.0;
    svm->y_map[1] = 1.0;
  } else {
    const PetscScalar *y_arr;
    PetscScalar *y_inner_arr;
    PetscInt i,n;
    TRY( VecGetLocalSize(y,&n) );
    TRY( VecDuplicate(y, &svm->y_inner) );
    TRY( VecGetArrayRead(y,&y_arr) );
    TRY( VecGetArray(svm->y_inner,&y_inner_arr) );
    for (i=0; i<n; i++) {
      if (y_arr[i]==min) {
        y_inner_arr[i] = -1.0;
      } else if (y_arr[i]==max) {
        y_inner_arr[i] = 1.0;
      } else {
        FLLOP_SETERRQ4(PetscObjectComm((PetscObject)svm),PETSC_ERR_ARG_OUTOFRANGE,"index %d: value %.1f is between max %.1f and min %.1f",i,y_arr[i],min,max);
      }
    }
    TRY( VecRestoreArrayRead(y,&y_arr) );
    TRY( VecRestoreArray(svm->y_inner,&y_inner_arr) );
    svm->y_map[0] = min;
    svm->y_map[1] = max;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetUp"
/*@
   SVMSetUp -

   Input Parameter:
.  svm - the SVM
@*/
PetscErrorCode SVMSetUp(SVM svm)
{
  QPS qps;
  QP qp;
  PetscReal C;
  Mat Xt;
  Vec y;
  Mat X,H;
  Vec e,lb,ub;
  Mat BE;
  PetscReal norm;
  Vec x_init;

  FllopTracedFunctionBegin;
  if (svm->setupcalled) PetscFunctionReturn(0);

  FllopTraceBegin;
  TRY( SVMGetQPS(svm, &qps) );
  TRY( QPSGetQP(qps, &qp) );
  TRY( SVMGetC(svm, &C) );
  TRY( SVMGetTrainingSamples(svm, &Xt, &y) );

  /* map y to -1,1 values if needed */
  TRY( SVMSetUp_Remapy_Private(svm) );

  if (C == PETSC_DECIDE || C == PETSC_DEFAULT) {
    TRY( SVMCrossValidate(svm) );
    TRY( SVMGetC(svm, &C) );
  }

  y = svm->y_inner;

  /* creating Hessian */
  TRY( PermonMatTranspose(Xt,MAT_TRANSPOSE_CHEAPEST,&X) );
  TRY( MatCreateNormal(X,&H) );                   /* H = X^t * X */
  TRY( MatDiagonalScale(H,y,y) );                 /* H = diag(y)*H*diag(y) */
  TRY( MatDestroy(&svm->D) );
  if (svm->loss_type == SVM_L2) {
    PetscInt m,n,N;
    Mat mats[2], HpE;
    //TODO use MatShift(H,0.5*C) once implemented
    TRY( MatGetLocalSize(H,&m,&n) );
    TRY( MatGetSize(H,&N,NULL) );
    TRY( MatCreateIdentity(PetscObjectComm((PetscObject)svm),m,n,N,&svm->D) );
    TRY( MatScale(svm->D,0.5*C) );
    mats[1] = H; mats[0] = svm->D;
    TRY( MatCreateSum(PetscObjectComm((PetscObject)svm),2,mats,&HpE) );
    TRY( MatDestroy(&H) );
    H = HpE;
  } else {
    FLLOP_ASSERT(svm->loss_type==SVM_L1,"svm->loss_type==SVM_L1");
  }

  TRY( QPSetOperator(qp,H) );                     /* set Hessian of QP problem */

  /* creating linear term */
  TRY( VecDuplicate(y,&e) );                      /* creating vector e same size and type as y */
  TRY( VecSet(e,1.0) );
  TRY( QPSetRhs(qp, e) );                         /* set linear term of QP problem */

  TRY( QPTNormalizeObjective(qp) );

  /* creating matrix of equality constraint */
  TRY( MatCreateOneRow(y,&BE) );                  /* Be = y^t */
  TRY( VecNorm(y, NORM_2, &norm) );
  TRY( MatScale(BE,1.0/norm) );                   /* ||Be|| = 1 */
  //TRY( QPSetEq(qp, BE, NULL) );                   /* set equality constraint to QP problem */

  {
    PetscInt m;
    TRY( MatGetSize(BE,&m,NULL) );
    FLLOP_ASSERT(m==1,"m==1");
  }

  /* creating box constraints */
  TRY( VecDuplicate(y,&lb) );                     /* create lower bound constraint vector */
  TRY( VecSet(lb, 0.0) );
  if (svm->loss_type == SVM_L1) {
    TRY( VecDuplicate(y,&ub) );                   /* create upper bound constraint vector */
    TRY( VecSet(ub, C) );
  } else {
    ub = NULL;
  }
  TRY( QPSetBox(qp, lb, ub) );                    /* set box constraints to QP problem */

  /* set init guess */
  if (svm->loss_type == SVM_L1) {
    VecDuplicate(lb, &x_init);
    VecSet(x_init, 0.);
    QPSetInitialVector(qp, x_init);
    VecDestroy(&x_init);
  }

  /* permorm QP transforms */
  TRY( QPTFromOptions(qp) );                      /* transform QP problem e.g. scaling */

  /* set solver settings from options if SVMSetFromOptions has been called */
  if (svm->setfromoptionscalled) {
    TRY( QPSSetFromOptions(qps) );
  }

  /* monitor Test rate */
  PetscBool printRate = PETSC_FALSE;
  TRY( PetscOptionsGetBool(NULL,NULL,"-print_rate", &printRate, NULL));
  if (printRate) {
    void *cctx;
    TRY( SVMQPSConvergedTrainRateCreate(svm,&cctx) );
    TRY( QPSSetConvergenceTest(qps,SVMQPSConvergedTrainRate,cctx,SVMQPSConvergedTrainRateDestroy) );
  }

  /* setup solver */
  TRY( QPSSetUp(qps) );

  /* decreasing reference counts */
  TRY( MatDestroy(&X) );
  TRY( MatDestroy(&H) );
  TRY( VecDestroy(&e) );
  TRY( MatDestroy(&BE) );
  TRY( VecDestroy(&lb) );
  TRY( VecDestroy(&ub) );
  svm->setupcalled = PETSC_TRUE;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetAutoPostTrain"
/*@
   SVMTrain -

   Input Parameter:
.  svm - the SVM

   Output Parameter:
.  qps -
@*/
PetscErrorCode SVMSetAutoPostTrain(SVM svm, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  svm->autoPostSolve = flg;
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
#define __FUNCT__ "SVMTrain"
/*@
   SVMTrain -

   Input Parameter:
.  svm - the SVM

   Output Parameter:
.  qps -
@*/
PetscErrorCode SVMTrain(SVM svm)
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  TRY( PetscPrintf(PetscObjectComm((PetscObject)svm),"### PermonSVM:   train with loss_type %s, C = %.2e\n",SVMLossTypes[svm->loss_type],svm->C) );
  TRY( SVMSetUp(svm) );
  TRY( QPSSetAutoPostSolve(svm->qps, PETSC_FALSE) );
  TRY( QPSSolve(svm->qps) );
  if (svm->autoPostSolve) TRY( SVMPostTrain(svm) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMPostTrain"
/*@
   SVMPostTrain -

   Input Parameter:
.  svm - the SVM

   Output Parameter:
.  qps -
@*/
PetscErrorCode SVMPostTrain(SVM svm)
{
  QPS qps;
  QP qp;
  IS is_sv;
  Vec o, y_sv, Xtw, Xtw_sv, t;
  PetscInt len_sv;
  Mat Xt;
  Vec Yz, y, z, w;
  PetscScalar b;

  PetscFunctionBeginI;
  TRY( SVMGetQPS(svm, &qps) );
  TRY( QPSPostSolve(qps) );
  TRY( QPSGetQP(qps, &qp) );
  TRY( SVMGetTrainingSamples(svm, &Xt, NULL) );
  y = svm->y_inner;

  /* reconstruct w from dual solution z */
  {
    TRY( QPGetSolutionVector(qp, &z) );
    TRY( VecDuplicate(z, &Yz) );

    TRY( VecPointwiseMult(Yz, y, z) );            /* YZ = Y*z = y.*z */
    TRY( MatCreateVecs(Xt, &w, NULL) );           /* create vector w such that Xt*w works */
    TRY( MatMultTranspose(Xt, Yz, w) );           /* Xt = X^t, w = Xt' * Yz = (X^t)^t * Yz = X * Yz */

    svm->w = w;

    TRY( VecDestroy(&Yz) );
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

    TRY( ISDestroy(&is_sv) );
    TRY( VecDestroy(&o) );
    TRY( VecDestroy(&t) );
    TRY( VecDestroy(&Xtw) );
  }
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetFromOptions"
/*@
   SVMSetFromOptions -

   Input Parameter:
.  svm - the SVM
@*/
PetscErrorCode SVMSetFromOptions(SVM svm)
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
#define __FUNCT__ "SVMGetSeparatingHyperplane"
/*@
   SVMGetSeparatingHyperplane - Return the classifier (separator) w*x - b = 0 computed by PermonSVMTrain()

   Not Collective

   Input Parameter:
.  svm - the SVM context

   Output Parameters:
+  w - the normal vector to the separating hyperplane
-  b - the offset of the hyperplane is given by b/||w||

 .seealso: SVMTrain(), SVMClassify(), SVMTest()
@*/
PetscErrorCode SVMGetSeparatingHyperplane(SVM svm, Vec *w, PetscReal *b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidPointer(w,2);
  PetscValidRealPointer(b,3);
  *w = svm->w;
  *b = svm->b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMClassify"
/*@
   SVMClassify -

   Input Parameter:
+  svm - the SVM
-  Xt_test

   Output Parameter:
.  y - labels {-1, 1}
@*/
PetscErrorCode SVMClassify(SVM svm, Mat Xt_test, Vec *y_out)
{
  PetscInt i, m;
  Vec Xtw_test, y, w;
  PetscReal b;

  const PetscScalar *Xtw_arr;
  PetscScalar *y_arr;

  PetscFunctionBeginI;
  TRY( SVMGetSeparatingHyperplane(svm, &w, &b) );
  TRY( MatCreateVecs(Xt_test,NULL,&Xtw_test) );
  TRY( MatMult(Xt_test,w,Xtw_test) );

  TRY( VecDuplicate(Xtw_test, &y) );
  TRY( VecGetLocalSize(Xtw_test, &m) );

  TRY( VecGetArrayRead(Xtw_test, &Xtw_arr) );
  TRY( VecGetArray(y, &y_arr) );
  for (i=0; i<m; i++) {
    if (Xtw_arr[i] + b > 0) {
      y_arr[i] = svm->y_map[1];
    } else {
      y_arr[i] = svm->y_map[0];
    }
  }
  TRY( VecRestoreArrayRead(Xtw_test, &Xtw_arr) );
  TRY( VecRestoreArray(y, &y_arr) );

  *y_out = y;
  TRY( VecDestroy(&Xtw_test) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMTest"
/*@
   SVMTest -

   Input Parameter:
+  svm - the SVM
-  Xt_test

   Output Parameter:
.
@*/
PetscErrorCode SVMTest(SVM svm, Mat Xt_test, Vec y_known, PetscInt *N_all, PetscInt *N_eq)
{
  Vec y;
  IS is_eq;

  PetscFunctionBeginI;
  TRY( SVMClassify(svm, Xt_test, &y) );
  TRY( VecWhichEqual(y,y_known,&is_eq) );
  TRY( VecGetSize(y,N_all) );
  TRY( ISGetSize(is_eq,N_eq) );
  TRY( VecDestroy(&y) );
  TRY( ISDestroy(&is_eq) );
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
  TRY( SVMGetTrainingSamples(svm, &Xt, &y_orig) );
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
