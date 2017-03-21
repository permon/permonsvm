
#include <private/svmimpl.h>

PetscClassId SVM_CLASSID;

#undef __FUNCT__
#define __FUNCT__ "PermonSVMCreate"
/*@
PermonSVMCreate - create instance of support vector machine classifier

Parameters:
+ comm - MPI comm 
- svm_out - pointer to created SVM 
@*/
PetscErrorCode PermonSVMCreate(MPI_Comm comm, PermonSVM *svm_out) 
{
  PermonSVM svm;

  PetscFunctionBeginI;
  PetscValidPointer(svm_out, 2);

#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  TRY(PermonSVMInitializePackage());
#endif
  TRY(PetscHeaderCreate(svm, SVM_CLASSID, "SVM", "SVM Classifier", "SVM", comm, PermonSVMDestroy, PermonSVMView));

  svm->setupcalled = PETSC_FALSE;
  svm->autoPostSolve = PETSC_TRUE;
  svm->qps = NULL;

  svm->C = PETSC_DECIDE;
  svm->C_min = 1e-3;
  svm->C_step = 1e1;
  svm->C_max = 1e3;
  svm->Xt = NULL;
  svm->y = NULL;
  svm->w = NULL;
  svm->b = 0.;

  *svm_out = svm;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMReset"
/*@
   QPReset - Resets a QP context to the QPsetupcalled = 0 state, destroys child, PC, Vecs,  Mats, etc.

   Collective on SVM

   Input Parameter:
.  svm - the SVM
@*/
PetscErrorCode PermonSVMReset(PermonSVM svm) 
{
  PetscFunctionBeginI;

  TRY(QPSDestroy(&svm->qps));
  TRY(MatDestroy(&svm->Xt));
  TRY(VecDestroy(&svm->y));
  PetscFunctionReturnI(0);
}


#undef __FUNCT__
#define __FUNCT__ "PermonSVMDestroy"
/*@
   PermonSVMDestroy - Destroys SVM context.

   Collective on SVM

   Input Parameter:
.  svm - SVM context
@*/
PetscErrorCode PermonSVMDestroy(PermonSVM *svm) 
{
  PetscFunctionBeginI;
  if (!*svm) PetscFunctionReturnI(0);

  PetscValidHeaderSpecific(*svm, SVM_CLASSID, 1);
  if (--((PetscObject) (*svm))->refct > 0) {
      *svm = 0;
      PetscFunctionReturn(0);
  }

  TRY(PermonSVMReset(*svm));
  TRY(PetscHeaderDestroy(svm));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMView"
/*@
   PermonSVMView - View information about SVM.

   Input Parameters:
+  svm - the SVM
-  v - visualization context
@*/
PetscErrorCode PermonSVMView(PermonSVM svm, PetscViewer v) 
{

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMSetPenalty"
/*@
   PermonSVMSetPenalty - Sets the penalty parameter.

   Input Parameter:
+  svm - the SVM
-  C - penalty parameter
@*/
PetscErrorCode PermonSVMSetPenalty(PermonSVM svm, PetscReal C) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(svm, C, 2);

  if (C <= 0) FLLOP_SETERRQ(((PetscObject) svm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Argument must be positive");
  svm->C = C;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMGetPenalty"
/*@
   PermonSVMGetPenalty - Gets the penalty parameter.

   Input Parameter:
.  svm - the SVM
 
   Output Parameter: 
.  C - penalty parameter
@*/
PetscErrorCode PermonSVMGetPenalty(PermonSVM svm, PetscReal *C) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidRealPointer(C, 2);
  *C = svm->C;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMSetPenaltyMin"
/*@
   PermonSVMSetPenalty - Sets the minimal penalty parameter value.

   Input Parameter:
+  svm - the SVM
-  C_min - minimal penalty parameter value 
@*/
PetscErrorCode PermonSVMSetPenaltyMin(PermonSVM svm, PetscReal C_min) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(svm, C_min, 2);

  if (C_min <= 0) FLLOP_SETERRQ(((PetscObject) svm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Argument must be positive");
  svm->C_min = C_min;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMGetPenaltyMin"
/*@
   PermonSVMGetPenalty - Gets the minimal penalty parameter value.

   Input Parameter:
.  svm - the SVM
 
   Output Parameter: 
.  C_min - minimal penalty parameter value 
@*/
PetscErrorCode PermonSVMGetPenaltyMin(PermonSVM svm, PetscReal *C_min) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidRealPointer(C_min, 2);
  *C_min = svm->C_min;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMSetPenaltyStep"
/*@
   PermonSVMSetPenalty - Sets the step penalty value.

   Input Parameter:
+  svm - the SVM
-  C_step - step penalty value
@*/
PetscErrorCode PermonSVMSetPenaltyStep(PermonSVM svm, PetscReal C_step) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(svm, C_step, 2);

  if (C_step <= 0) FLLOP_SETERRQ(((PetscObject) svm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Argument must be positive");
  svm->C_step = C_step;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMGetPenaltyStep"
/*@
   PermonSVMGetPenalty - Gets the step penalty value.

   Input Parameter:
.  svm - the SVM
 
   Output Parameter: 
.  C_step - step penalty value 
@*/
PetscErrorCode PermonSVMGetPenaltyStep(PermonSVM svm, PetscReal *C_step) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidRealPointer(C_step, 2);
  *C_step = svm->C_step;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMSetPenaltyMax"
/*@
   PermonSVMSetPenaltyMax - Sets the max penalty parameter value.

   Input Parameter:
+  svm - the SVM
-  C_max - max penalty parameter value
@*/
PetscErrorCode PermonSVMSetPenaltyMax(PermonSVM svm, PetscReal C_max) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(svm, C_max, 2);

  if (C_max <= 0) FLLOP_SETERRQ(((PetscObject) svm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Argument must be positive");
  svm->C_max = C_max;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMGetPenaltyMax"
/*@
   PermonSVMGetPenaltyMax - Gets the max penalty parameter value.

   Input Parameter:
.  svm - the SVM
 
   Output Parameter: 
.  C_max - maximal penalty parameter value 
@*/
PetscErrorCode PermonSVMGetPenaltyMax(PermonSVM svm, PetscReal *C_max) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidRealPointer(C_max, 2);
  *C_max = svm->C_max;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMSetNfolds"
/*@
   PermonSVMSetPenalty - Sets the penalty parameter.

   Input Parameter:
+  svm - the SVM
-  C - penalty parameter
@*/
PetscErrorCode PermonSVMSetNfolds(PermonSVM svm, PetscInt nfolds) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(svm, nfolds, 2);

  if (nfolds < 2) FLLOP_SETERRQ(((PetscObject) svm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Argument must be greater than 1.");
  svm->nfolds = nfolds;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMGetNfolds"
/*@
   PermonSVMGetNfolds - Gets the number of folds.

   Input Parameter:
.  svm - the SVM
 
   Output Parameter: 
.  nfolds - number of folds
@*/
PetscErrorCode PermonSVMGetNfolds(PermonSVM svm, PetscInt *nfolds) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidRealPointer(nfolds, 2);
  *nfolds = svm->nfolds;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMSetTrainingSamples"
/*@
   PermonSVMSetTrainingSamples - Sets the training samples.

   Input Parameter:
+  svm - the SVM
.  Xt - samples data
-  y - 
@*/
PetscErrorCode PermonSVMSetTrainingSamples(PermonSVM svm, Mat Xt, Vec y) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidHeaderSpecific(Xt, MAT_CLASSID, 2);
  PetscCheckSameComm(svm, 1, Xt, 2);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 3);
  PetscCheckSameComm(svm, 1, y, 3);

  TRY(MatDestroy(&svm->Xt));
  svm->Xt = Xt;
  TRY(PetscObjectReference((PetscObject) Xt));

  TRY(VecDestroy(&svm->y));
  svm->y = y;
  TRY(PetscObjectReference((PetscObject) y));

  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMGetTrainingSamples"
/*@
   PermonSVMSetTrainingSamples - Sets the training samples.

   Input Parameter:
.  svm - the SVM

   Output Parameter:    
+  Xt - samples data
-  y - 
@*/
PetscErrorCode PermonSVMGetTrainingSamples(PermonSVM svm, Mat *Xt, Vec *y) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidPointer(Xt, 2);
  PetscValidPointer(y, 3);

  *Xt = svm->Xt;
  *y = svm->y;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMSetQPS"
/*@
   PermonSVMSetQPS - Sets the QPS.

   Input Parameter:
.  svm - the SVM

   Output Parameter:    
.  qps -  
@*/
PetscErrorCode PermonSVMSetQPS(PermonSVM svm, QPS qps) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 2);
  PetscCheckSameComm(svm, 1, qps, 2);

  TRY(QPSDestroy(&svm->qps));
  svm->qps = qps;
  TRY(PetscObjectReference((PetscObject) qps));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMGetQPS"
/*@
   PermonSVMGetQPS - Gets the QPS.

   Input Parameter:
.  svm - the SVM

   Output Parameter:    
.  qps - 
@*/
PetscErrorCode PermonSVMGetQPS(PermonSVM svm, QPS *qps) 
{  
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidPointer(qps,2);

  if (!svm->qps) {
    QP qp;
    TRY(QPSCreate(PetscObjectComm((PetscObject)svm), &svm->qps));
    TRY(QPCreate(PetscObjectComm((PetscObject)svm), &qp));
    QPSSetQP(svm->qps, qp);
    QPDestroy(&qp);
  }
  *qps = svm->qps;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMSetUp"
/*@
   PermonSVMSetUp - 

   Input Parameter:
.  svm - the SVM
@*/
PetscErrorCode PermonSVMSetUp(PermonSVM svm) 
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

  FllopTracedFunctionBegin;
  if (svm->setupcalled) PetscFunctionReturn(0);

  FllopTraceBegin;
  TRY( PermonSVMGetQPS(svm, &qps) );
  TRY( QPSGetQP(qps, &qp) );
  TRY( PermonSVMGetPenalty(svm, &C) );
  TRY( PermonSVMGetTrainingSamples(svm, &Xt, &y) );

  if (C == PETSC_DECIDE) {
    TRY( PermonSVMCrossValidate(svm) );
    TRY( PermonSVMGetPenalty(svm, &C) );
  }

  TRY( FllopMatTranspose(Xt,MAT_TRANSPOSE_CHEAPEST,&X) );
  TRY( MatCreateNormal(X,&H) ); //H = X^t * X
  TRY( MatDiagonalScale(H,y,y) ); //H = diag(y)*H*diag(y)
  TRY( QPSetOperator(qp,H) ); //set Hessian of QP problem

  //creating linear term
  TRY( VecDuplicate(y,&e) ); //creating vector e same size and type as y
  TRY( VecSet(e,1.0) );
  TRY( QPSetRhs(qp, e) ); //set linear term of QP problem

  //creating matrix of equality constraint
  TRY( MatCreateOneRow(y,&BE) ); //Be = y^t
  TRY( VecNorm(y, NORM_2, &norm) );
  TRY( MatScale(BE,1.0/norm) ); //||Be|| = 1
  TRY( QPSetEq(qp, BE, NULL)); //set equality constraint to QP problem
  
  {
    PetscInt m;
    TRY( MatGetSize(BE,&m,NULL) );
    FLLOP_ASSERT(m==1,"m==1");
  }

  //creating box constraints
  TRY( VecDuplicate(y,&lb) ); //create lower bound constraint vector
  TRY( VecDuplicate(y,&ub) ); //create upper bound constraint vector
  TRY( VecSet(lb, 0.0) );
  TRY( VecSet(ub, C) );
  TRY( QPSetBox(qp, lb, ub) ); //set box constraints to QP problem
  
  TRY( QPTFromOptions(qp) ); //transform QP problem e.g. scaling
  TRY( QPSSetUp(qps) );

  //decreasing reference counts
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
#define __FUNCT__ "PermonSVMSetAutoPostTrain"
/*@
   PermonSVMTrain - 

   Input Parameter:
.  svm - the SVM

   Output Parameter:    
.  qps - 
@*/
PetscErrorCode PermonSVMSetAutoPostTrain(PermonSVM svm, PetscBool flg) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  svm->autoPostSolve = flg;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMTrain"
/*@
   PermonSVMTrain - 

   Input Parameter:
.  svm - the SVM

   Output Parameter:    
.  qps - 
@*/
PetscErrorCode PermonSVMTrain(PermonSVM svm) 
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  TRY( PermonSVMSetUp(svm) );
  TRY( QPSSetAutoPostSolve(svm->qps, PETSC_FALSE) );
  TRY( QPSSolve(svm->qps) );
  if (svm->autoPostSolve) TRY( PermonSVMPostTrain(svm) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMPostTrain"
/*@
   PermonSVMPostTrain - 

   Input Parameter:
.  svm - the SVM

   Output Parameter:    
.  qps - 
@*/
PetscErrorCode PermonSVMPostTrain(PermonSVM svm) 
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
  TRY( PermonSVMGetQPS(svm, &qps) );
  TRY( QPSPostSolve(qps) );
  TRY( QPSGetQP(qps, &qp) );
  TRY( PermonSVMGetTrainingSamples(svm, &Xt, &y) );
  
  /* reconstruct w from dual solution z */
  {
    TRY( QPGetSolutionVector(qp, &z) );
    TRY( VecDuplicate(z, &Yz));

    TRY( VecPointwiseMult(Yz, y, z) ); // YZ = Y*z = y.*z
    TRY( MatCreateVecs(Xt, &w, NULL) ); // create vector w such that Xt*w works
    TRY( MatMultTranspose(Xt, Yz, w) ); // Xt = X^t, w = Xt' * Yz = (X^t)^t * Yz = X * Yz

    svm->w = w;
    
    TRY( VecDestroy(&Yz) );
  }
  
  /* reconstruct b from dual solution z */
  //PDF p. 12 lecture_svm.pdf
  {
    TRY( VecDuplicate(z, &o) );
    TRY( VecZeroEntries(o) );
    TRY( MatCreateVecs(Xt, NULL, &Xtw));

    TRY( VecWhichGreaterThan(z, o, &is_sv) );
    TRY( ISGetSize(is_sv, &len_sv) );
    TRY( MatMult(Xt, w, Xtw) );
    TRY( VecGetSubVector(y, is_sv, &y_sv) );  // y_sv = y(is_sv)
    TRY( VecGetSubVector(Xtw, is_sv, &Xtw_sv) );  // Xtw_sv = Xtw(is_sv)
    TRY( VecDuplicate(y_sv, &t) );
    TRY( VecWAXPY(t, -1.0, Xtw_sv, y_sv) );  // t = y_sv - Xtw_sv
    TRY( VecRestoreSubVector(y, is_sv, &y_sv) );
    TRY( VecRestoreSubVector(Xtw, is_sv, &Xtw_sv) );
    TRY( VecSum(t, &b) );  // b = sum(t)
    b /= len_sv;  // b = b / length(is_sv)
    
    svm->b = b;

    TRY( ISDestroy(&is_sv) );
    TRY( VecDestroy(&o) );
    TRY( VecDestroy(&t) );
  }
  
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMSetFromOptions"
/*@
   PermonSVMPostTrain - 

   Input Parameter:
.  svm - the SVM
@*/
PetscErrorCode PermonSVMSetFromOptions(PermonSVM svm) 
{
  PetscReal C, C_min, C_max, C_step;
  PetscInt nfolds;
  PetscBool flg;
  QPS qps;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  _fllop_ierr = PetscObjectOptionsBegin((PetscObject)svm);CHKERRQ(_fllop_ierr);
  TRY( PetscOptionsReal("-svm_penalty","Set SVM penalty (C).","PermonSVMSetPenalty",svm->C,&C,&flg) );
  if (flg) TRY( PermonSVMSetPenalty(svm, C) );
  TRY( PetscOptionsReal("-svm_penalty_min","Set SVM minimal penalty value (C_min).","PermonSVMSetPenaltyMin",svm->C_min,&C_min,&flg) );
  if (flg) TRY( PermonSVMSetPenaltyMin(svm, C_min) );
  TRY( PetscOptionsReal("-svm_penalty_max","Set SVM maximal penalty value (C_max).","PermonSVMSetPenaltyMax",svm->C_max,&C_max,&flg) );
  if (flg) TRY( PermonSVMSetPenaltyMax(svm, C_max) );
  TRY( PetscOptionsReal("-svm_penalty_step","Set SVM step penalty value (C_step).","PermonSVMSetPenaltyStep",svm->C_step,&C_step,&flg) );
  if (flg) TRY( PermonSVMSetPenaltyStep(svm, C_step) );
  TRY( PetscOptionsInt("-svm_nfolds","Set number of folds (nfolds).","PermonSVMSetNfolds",svm->nfolds,&nfolds,&flg) );
  if (flg) TRY( PermonSVMSetNfolds(svm, nfolds) );
  TRY( PermonSVMGetQPS(svm, &qps) );
  TRY( QPSSetFromOptions(qps) );
  _fllop_ierr = PetscOptionsEnd();CHKERRQ(_fllop_ierr);
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMGetSepparatingHyperplane"
/*@
   PermonSVMPostTrain - 

   Input Parameter:
+  svm - the SVM
-  Xt_test

   Output Parameter:    
.  
@*/
PetscErrorCode PermonSVMGetSepparatingHyperplane(PermonSVM svm, Vec *w, PetscReal *b)
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidPointer(w,2);
  PetscValidPointer(b,2);
  *w = svm->w;
  *b = svm->b;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMClassify"
/*@
   PermonSVMClassify - 

   Input Parameter:
+  svm - the SVM
-  Xt_test

   Output Parameter:    
.  y - labels {-1, 1}
@*/
PetscErrorCode PermonSVMClassify(PermonSVM svm, Mat Xt_test, Vec *y_out) 
{
  PetscInt i, m;
  Vec Xtw_test, y, w;
  PetscReal b;
  
  const PetscScalar *Xtw_arr;
  PetscScalar *y_arr;
  
  PetscFunctionBeginI;
  TRY( PermonSVMGetSepparatingHyperplane(svm, &w, &b) );
  TRY( MatCreateVecs(Xt_test,NULL,&Xtw_test) );
  TRY( MatMult(Xt_test,w,Xtw_test) );
  
  TRY( VecDuplicate(Xtw_test, &y) );
  TRY( VecGetLocalSize(Xtw_test, &m) );
  
  TRY( VecGetArrayRead(Xtw_test, &Xtw_arr) );
  TRY( VecGetArray(y, &y_arr) );
  for (i=0; i<m; i++) {
    if (Xtw_arr[i] + b > 0) {
      y_arr[i] = 1.0;
    } else {
      y_arr[i] = -1.0;
    }
  }
  TRY( VecRestoreArrayRead(Xtw_test, &Xtw_arr) );
  TRY( VecRestoreArray(y, &y_arr) );
  
  *y_out = y;
  
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMTest"
/*@
   PermonSVMTest - 

   Input Parameter:
+  svm - the SVM
-  Xt_test

   Output Parameter:    
.
@*/
PetscErrorCode PermonSVMTest(PermonSVM svm, Mat Xt_test, Vec y_known, PetscInt *N_all, PetscInt *N_eq)
{
  Vec y;
  IS is_eq;
  
  PetscFunctionBeginI; 
  TRY( PermonSVMClassify(svm, Xt_test, &y) );
  TRY( VecWhichEqual(y,y_known,&is_eq) );
  TRY( VecGetSize(y,N_all) );
  TRY( ISGetSize(is_eq,N_eq) );
  TRY( VecDestroy(&y) );
  TRY( ISDestroy(&is_eq) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMCrossValidate"
/*@
   PermonSVMCrossValidate - 

   Input Parameter:
.  svm - the SVM

@*/
PetscErrorCode PermonSVMCrossValidate(PermonSVM svm) 
{
  PermonSVM cross_svm;
  IS is_test, is_train;
  PetscInt i, j;
  PetscInt nfolds, first, step, n;
  PetscInt lo, hi, N, max;
  PetscInt N_all, N_eq, c_count;
  PetscReal C, C_min, C_step, C_max, C_i;
  PetscReal *array_rate = NULL, rate_max, rate;
  Mat Xt, Xt_test, Xt_train;
  Vec y, y_test, y_train;
  
  PetscFunctionBeginI;
  TRY( PermonSVMGetTrainingSamples(svm,&Xt,&y) );
  TRY( MatGetOwnershipRange(Xt,&lo,&hi) );
  TRY( PermonSVMGetPenaltyMin(svm, &C_min) );
  TRY( PermonSVMGetPenaltyStep(svm, &C_step) );
  TRY( PermonSVMGetPenaltyMax(svm, &C_max) );
  TRY( PermonSVMGetNfolds(svm, &nfolds) );
  step = nfolds;
  max = hi - 1;
  /*c_count = (PetscInt)(C_max - C_min) / C_step + 1;*/
  c_count = 0;
  for (C_i = C_min; C_i < C_max; C_i*=C_step) c_count = c_count + 1;
  TRY( ISCreate(PetscObjectComm((PetscObject)svm),&is_test) );
  TRY( ISSetType(is_test,ISSTRIDE) );
  TRY( PetscMalloc1(c_count,&array_rate) );
  TRY( PetscMemzero(array_rate,c_count*sizeof(PetscReal)) );
  
  for (i = 0; i < nfolds; ++i) {
    first = i + lo;
    n = (PetscInt)((max - first) / step + 1);
    
    TRY( ISStrideSetStride(is_test,n,first,step) );
    TRY( MatGetSubMatrix(Xt,is_test,NULL,MAT_INITIAL_MATRIX,&Xt_test) );
    TRY( VecGetSubVector(y,is_test,&y_test) );
    TRY( ISComplement(is_test,lo,hi,&is_train) );
    TRY( MatGetSubMatrix(Xt,is_train,NULL,MAT_INITIAL_MATRIX,&Xt_train) );
    TRY( VecGetSubVector(y,is_train,&y_train) );
    
    TRY( MatGetSize(Xt_test,&N,NULL) );
    TRY( PermonSVMCreate(PetscObjectComm((PetscObject)svm),&cross_svm) );
    TRY( PermonSVMSetTrainingSamples(cross_svm,Xt_train,y_train) );
    TRY( PermonSVMSetFromOptions(cross_svm) );
    for (j = 0; j < c_count; ++j) {
      C_i = C_min * PetscPowScalar(C_step, j);
      PetscPrintf(PETSC_COMM_WORLD, "C_min = %f, C_%d = %f\n", C_min, j, C_i);
      {
        //TODO this is just hotfix
        TRY( PermonSVMReset(cross_svm) );
        TRY( PermonSVMSetTrainingSamples(cross_svm,Xt_train,y_train) );
        TRY( PermonSVMSetFromOptions(cross_svm) );
      }
      TRY( PermonSVMSetPenalty(cross_svm,C_i) );
      TRY( PermonSVMSetUp(cross_svm) );
      TRY( PermonSVMSetFromOptions(cross_svm) );
      TRY( PermonSVMTrain(cross_svm) );
      TRY( PermonSVMTest(cross_svm,Xt_test,y_test,&N_all,&N_eq) );
      FLLOP_ASSERT(N_all==N,"N_all==N");
      array_rate[j] += ((PetscReal)N_eq) / ((PetscReal)N);
      PetscPrintf(PETSC_COMM_WORLD, "N_eq = %d, rate = %f, rate_acc = %f\n", N_eq, ((PetscReal)N_eq) / ((PetscReal)N), array_rate[j]);
    }
    TRY( PermonSVMDestroy(&cross_svm) );
    TRY( MatDestroy(&Xt_test) );
    TRY( VecRestoreSubVector(y,is_test,&y_test) );
    TRY( MatDestroy(&Xt_train) );
    TRY( VecRestoreSubVector(y,is_train,&y_train) );
    TRY( ISDestroy(&is_train) );
  }
  
  rate_max = 0.;
  for(i = 0; i < c_count; ++i) {
    rate = (PetscReal)(array_rate[i] / nfolds);
    if (rate > rate_max) {
      rate_max = rate;
      C = C_min * PetscPowScalar(C_step, i);
    }
  }
  TRY( PermonSVMSetPenalty(svm, C) );
  TRY( ISDestroy(&is_test) );
  TRY( PetscFree(array_rate) );
  PetscFunctionReturnI(0);
} 
