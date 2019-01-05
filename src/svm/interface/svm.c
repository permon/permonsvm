
#include <permon/private/svmimpl.h>

PetscClassId SVM_CLASSID;

const char *const ModelScores[]={"accuracy","precision","sensitivity","F1","mcc","ModelScore","model_",0};

#undef __FUNCT__
#define __FUNCT__ "SVMCreate"
/*@
  SVMCreate - Creates instance of Support Vector Machine classifier.

  Collective on MPI_Comm

  Input Parameter:
. comm - MPI comm

  Output Parameter:
. svm_out - pointer to created SVM

  Level: beginner

.seealso SVMDestroy(), SVMReset(), SVMSetup(), SVMSetFromOptions()
@*/
PetscErrorCode SVMCreate(MPI_Comm comm,SVM *svm_out)
{
  SVM svm;

  PetscFunctionBegin;
  PetscValidPointer(svm_out,2);

#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  TRY( SVMInitializePackage() );
#endif
  TRY( PetscHeaderCreate(svm,SVM_CLASSID,"SVM","SVM Classifier","SVM",comm,SVMDestroy,SVMView) );

  svm->C          = 1.;
  svm->C_old      = 1.;
  svm->LogCBase   = 2.;
  svm->LogCMin    = -2.;
  svm->LogCMax    = 2.;
  svm->loss_type  = SVM_L1;

  svm->cv_model_score = MODEL_ACCURACY;
  svm->nfolds         = 5;
  svm->warm_start     = PETSC_FALSE;

  svm->setupcalled          = PETSC_FALSE;
  svm->setfromoptionscalled = PETSC_FALSE;
  svm->autoposttrain        = PETSC_TRUE;
  svm->posttraincalled      = PETSC_FALSE;

  svm->Xt_test = NULL;
  svm->y_test  = NULL;

  *svm_out = svm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMReset"
/*@
  SVMReset - Resets a SVM context to the setupcalled = 0.

  Collective on SVM

  Input Parameter:
. svm - SVM context

  Level: beginner

.seealso SVMCreate(), SVMSetUp()
@*/
PetscErrorCode SVMReset(SVM svm)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  svm->C          = 1.;
  svm->C_old      = 1.;
  svm->LogCBase   = 2.;
  svm->LogCMin    = -2.;
  svm->LogCMax    = 2.;
  svm->loss_type  = SVM_L1;

  svm->nfolds     = 5;
  svm->warm_start = PETSC_FALSE;

  TRY( MatDestroy(&svm->Xt_test) );
  svm->Xt_test = NULL;
  TRY( VecDestroy(&svm->y_test) );
  svm->y_test  = NULL;

  TRY( (*svm->ops->reset)(svm) );

  svm->setupcalled          = PETSC_FALSE;
  svm->autoposttrain        = PETSC_TRUE;
  svm->posttraincalled      = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMDestroyDefault"
/*@
  SVMDestroyDefault - Destroys SVM context.

  Input parameter:
. svm - SVM context

  Developers Note: This is PETSC_EXTERN because it may be used by user written plugin SVM implementations

  Level: developer

.seealso SVMDestroy()
@*/
PetscErrorCode SVMDestroyDefault(SVM svm)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( PetscFree(svm->data) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMDestroy"
/*@
  SVMDestroy - Destroys SVM context.

  Collective on SVM

  Input Parameter:
. svm - SVM context

  Level: beginner

.seealso SVMCreate(), SVMSetUp()
@*/
PetscErrorCode SVMDestroy(SVM *svm)
{

  PetscFunctionBegin;
  if (!*svm) PetscFunctionReturn(0);

  PetscValidHeaderSpecific(*svm,SVM_CLASSID,1);
  if (--((PetscObject) (*svm))->refct > 0) {
    *svm = 0;
    PetscFunctionReturn(0);
  }

  TRY( SVMReset(*svm) );
  if ((*svm)->ops->destroy) {
    TRY( (*(*svm)->ops->destroy)(*svm) );
  }

  TRY( PetscHeaderDestroy(svm) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetFromOptions"
/*@
  SVMSetFromOptions - Sets SVM options from the options database.

  Logically Collective on SVM

  Input Parameter:
. svm - SVM context

  Level: beginner

.seealso SVMCreate(), SVMSetUp()
@*/
PetscErrorCode SVMSetFromOptions(SVM svm)
{
  PetscErrorCode ierr;

  PetscReal      C,logC_min,logC_max,logC_base;
  PetscBool      flg,warm_start;

  SVMLossType    loss_type;
  PetscInt       nfolds;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  ierr = PetscObjectOptionsBegin((PetscObject)svm);CHKERRQ(ierr);
  TRY( PetscOptionsReal("-svm_C","Set SVM C (C).","SVMSetC",svm->C,&C,&flg) );
  if (flg) {
    TRY( SVMSetC(svm,C) );
  }
  TRY( PetscOptionsReal("-svm_logC_min","Set SVM minimal C value (LogCMin).","SVMSetLogCMin",svm->LogCMin,&logC_min,&flg) );
  if (flg) {
    TRY( SVMSetLogCMin(svm,logC_min) );
  }
  TRY( PetscOptionsReal("-svm_logC_max","Set SVM maximal C value (LogCMax).","SVMSetLogCMax",svm->LogCMax,&logC_max,&flg) );
  if (flg) {
    TRY( SVMSetLogCMax(svm,logC_max) );
  }
  TRY( PetscOptionsReal("-svm_logC_base","Set power base of SVM parameter C (LogCBase).","SVMSetLogCBase",svm->LogCBase,&logC_base,&flg) );
  if (flg) {
    TRY( SVMSetLogCBase(svm,logC_base) );
  }
  TRY( PetscOptionsInt("-svm_nfolds","Set number of folds (nfolds).","SVMSetNfolds",svm->nfolds,&nfolds,&flg) );
  if (flg) {
    TRY( SVMSetNfolds(svm,nfolds) );
  }
  TRY( PetscOptionsEnum("-svm_loss_type","Specify the loss function for soft-margin SVM (non-separable samples).","SVMSetNfolds",SVMLossTypes,(PetscEnum)svm->loss_type,(PetscEnum*)&loss_type,&flg) );
  if (flg) {
    TRY( SVMSetLossType(svm,loss_type) );
  }
  TRY( PetscOptionsBool("-svm_warm_start","Specify whether warm start is used in cross-validation.","SVMSetWarmStart",svm->warm_start,&warm_start,&flg) );
  if (flg) {
    TRY( SVMSetWarmStart(svm,warm_start) );
  }

  if (svm->ops->setfromoptions) {
    TRY( svm->ops->setfromoptions(PetscOptionsObject,svm) );
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  svm->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetType"
/*@
  SVMSetType - Sets the type of SVM classifier.

  Collective on SVM

  Input Parameters:
+ svm - SVM context
- type - the type of SVM classifier

  Level: beginner

.seealso SVMCreate(), SVMType
@*/
PetscErrorCode SVMSetType(SVM svm,const SVMType type)
{
  PetscErrorCode (*create_svm)(SVM);
  PetscBool issame = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidCharPointer(type,2);

  TRY( PetscObjectTypeCompare((PetscObject) svm,type,&issame) );
  if (issame) PetscFunctionReturn(0);

  TRY( PetscFunctionListFind(SVMList,type,(void(**)(void))&create_svm) );
  if (!create_svm) FLLOP_SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested SVM type %s",type);

  /* Destroy the pre-existing private SVM context */
  if (svm->ops->destroy) svm->ops->destroy(svm);
  /* Reinitialize function pointers in SVMOps structure */
  TRY( PetscMemzero(svm->ops,sizeof(struct _SVMOps)) );

  TRY( (*create_svm)(svm) );
  TRY( PetscObjectChangeTypeName((PetscObject)svm,type) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetQPS"
/*@
  SVMSetQPS - Sets the QPS.

  Not Collective

  Input Parameters:
+ svm - SVM context
- qps - QPS context

  Level: advanced

.seealso SVMGetQPS()
@*/
PetscErrorCode SVMSetQPS(SVM svm,QPS qps)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(qps,QPS_CLASSID,2);

  TRY( PetscTryMethod(svm,"SVMSetQPS_C",(SVM,QPS),(svm,qps)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetQPS"
/*@
  SVMGetQPS - Returns the QPS.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. qps - QPS context

  Level: advanced

.seealso SVMSetQPS()
@*/
PetscErrorCode SVMGetQPS(SVM svm,QPS *qps)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidPointer(qps,2);

  TRY( PetscUseMethod(svm,"SVMGetQPS_C",(SVM,QPS *),(svm,qps)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetNfolds"
/*@
  SVMSetNfolds - Sets the number of folds.

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- nfolds - the number of folds

  Level: beginner

.seealso SVMSetNfolds(), SVMCrossValidation()
@*/
PetscErrorCode SVMSetNfolds(SVM svm,PetscInt nfolds)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveInt(svm,nfolds,2);

  if (nfolds < 2) FLLOP_SETERRQ(((PetscObject) svm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Argument must be greater than 1.");
  svm->nfolds = nfolds;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetNfolds"
/*@
  SVMGetNfolds - Returns the number of folds.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. nfolds - the number of folds

  Level: beginner

.seealso SVMSetNfolds(), SVMCrossValidation()
@*/
PetscErrorCode SVMGetNfolds(SVM svm,PetscInt *nfolds)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(nfolds,2);
  *nfolds = svm->nfolds;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetC"
/*@
  SVMSetC - Sets the value of penalty C.

  Collective on SVM

  Input Parameters:
+ svm - SVM context
- C - the value of penalty C

  Level: beginner

.seealso SVMGetC(), SVMGridSearch()
@*/
PetscErrorCode SVMSetC(SVM svm,PetscReal C)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,C,2);

  if (svm->C == C) PetscFunctionReturn(0);

  if (C <= 0 && C != PETSC_DECIDE && C != PETSC_DEFAULT) {
    FLLOP_SETERRQ(((PetscObject) svm)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");
  }

  svm->C_old = svm->C;
  svm->C     = C;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetC"
/*@
  SVMGetC - Returns the value of penalty C.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. C - the value of penalty C

  Level: beginner

.seealso SVMSetC(), SVMGridSearch()
@*/
PetscErrorCode SVMGetC(SVM svm,PetscReal *C)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(C,2);
  *C = svm->C;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetLogCBase"
/*@
  SVMSetLogCBase - Sets the value of penalty C step.

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- LogCBase - the value of penalty C step

  Level: beginner

.seealso SVMSetC(), SVMSetLogBase(), SVMSetLogCMin(), SVMSetLogCMax(), SVMGridSearch()
@*/
PetscErrorCode SVMSetLogCBase(SVM svm,PetscReal LogCBase)
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
  SVMGetLogCBase - Returns the value of penalty C step.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. LogCBase - the value of penalty C step

  Level: beginner

.seealso SVMGetC(), SVMGetLogCMin(), SVMGetLogCMax(), SVMGridSearch()
@*/
PetscErrorCode SVMGetLogCBase(SVM svm,PetscReal *LogCBase)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidRealPointer(LogCBase, 2);
  *LogCBase = svm->LogCBase;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetLogCMin"
/*@
  SVMSetLogCMin - Sets the minimum value of log C penalty.

  Logically Colective on SVM

  Input Parameter:
+ svm - SVM context
- LogCMin - the minimum value of log C penalty

  Level: beginner

.seealso SVMSetC(), SVMSetLogCBase(), SVMSetLogCMax(), SVMGridSearch()
@*/
PetscErrorCode SVMSetLogCMin(SVM svm,PetscReal LogCMin)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,LogCMin,2);
  svm->LogCMin = LogCMin;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetLogCMin"
/*@
  SVMGetLogCMin - Returns the minimum value of log C penalty.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. LogCMin - the minimum value of log C penalty

  Level: beginner

.seealso SVMGetC(), SVMGetLogCBase(), SVMGetLogCMax(), SVMGridSearch()
@*/
PetscErrorCode SVMGetLogCMin(SVM svm,PetscReal *LogCMin)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidRealPointer(LogCMin, 2);
  *LogCMin = svm->LogCMin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetLogCMax"
/*@
  SVMSetLogCMax - Sets the maximum value of log C penalty.

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- LogCMax - the maximum value of log C penalty

  Level: beginner

.seealso SVMSetC(), SVMSetLogCBase(), SVMSetLogCMin(), SVMGridSearch()
@*/
PetscErrorCode SVMSetLogCMax(SVM svm,PetscReal LogCMax)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,LogCMax,2);
  svm->LogCMax = LogCMax;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetLogCMax"
/*@
  SVMGetLogCMax - Returns the maximum value of log C penalty.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. LogCMax - the maximum value of log C penalty

  Level: beginner

.seealso SVMGetC(), SVMGetLogCBase(), SVMGetLogCMin(), SVMGridSearch()
@*/
PetscErrorCode SVMGetLogCMax(SVM svm,PetscReal *LogCMax)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(LogCMax,2);
  *LogCMax = svm->LogCMax;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetLossType"
/*@
   SVMSetLossType - Sets the type of the hinge loss function.

   Logically Collective on SVM

   Input Parameters:
+  svm - SVM context
-  type - the type of loss function

   Level: beginner

.seealso SVMLossType, SVMGetLossType()
@*/
PetscErrorCode SVMSetLossType(SVM svm,SVMLossType type)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(svm,type,2);
  if (type != svm->loss_type) {
    svm->loss_type = type;
    svm->setupcalled = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetLossType"
/*@
  SVMGetLossType - Returns the type of the loss function.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. type - the type of loss function

  Level: beginner

.seealso PermonSVMType, SVMSetLossType()
@*/
PetscErrorCode SVMGetLossType(SVM svm,SVMLossType *type)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidPointer(type,2);
  *type = svm->loss_type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetMod"
/*@
  SVMSetMod - Sets type of SVM formulation.

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- mod - type of SVM formulation

  Level: beginner

.seealso SVMGetMod()
@*/
PetscErrorCode SVMSetMod(SVM svm,PetscInt mod)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveInt(svm,mod,2);
  TRY( PetscTryMethod(svm,"SVMSetMod_C",(SVM,PetscInt),(svm,mod)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetMod"
/*@
  SVMGetMod - Returns type of SVM formulation.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. mod - the type of SVM formulation

  Level: beginner

.seealso SVMSetMod()
@*/
PetscErrorCode SVMGetMod(SVM svm,PetscInt *mod)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( PetscTryMethod(svm,"SVMGetMod_C",(SVM,PetscInt *),(svm,mod)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetOptionsPrefix"
/*@
  SVMSetOptionsPrefix - Sets the prefix used for searching for all options of the SVM classifier and the QPS solver in the database.

  Collective on SVM

  Input Parameters:
+ SVM - SVM context
- prefix - the prefix string

  Level: developer

.seealso SVMAppendOptionsPrefix(), SVMGetOptionsPrefix(), SVM, QPS
@*/
PetscErrorCode SVMSetOptionsPrefix(SVM svm,const char prefix[])
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( PetscTryMethod(svm,"SVMSetOptionsPrefix_C",(SVM,const char []),(svm,prefix)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMAppendOptionsPrefix"
/*@
  SVMAppendOptionsPrefix - Appends the prefix used for searching for all options of the SVM classifier and the QPS solver in the database.

  Collective on SVM

  Input Parameters:
+ SVM - SVM context
- prefix - the prefix string

  Level: developer

.seealso SVMSetOptionsPrefix(), SVMGetOptionsPrefix(), SVM, QPS
*/
PetscErrorCode SVMAppendOptionsPrefix(SVM svm,const char prefix[])
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( PetscTryMethod(svm,"SVMAppendOptionsPrefix_C",(SVM,const char []),(svm,prefix)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetOptionsPrefix"
/*@
  SVMGetOptionsPrefix - Returns the prefix of SVM classifier and QPS solver.

  Not Collective

  Input Parameters:
. SVM - SVM context

  Output Parameters:
. prefix - pointer to the prefix string

  Level: developer

.seealso SVMSetOptionsPrefix(), SVM, QPS
@*/
PetscErrorCode SVMGetOptionsPrefix(SVM svm,const char *prefix[])
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( PetscUseMethod(svm,"SVMGetOptionsPrefix_C",(SVM,const char *[]),(svm,prefix)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetWarmStart"
/*@
  SVMSetWarmStart - Set flag specifying whether warm start is used in cross-validation.
  It is set to PETSC_TRUE by default.

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- flg - use warm start in cross-validation

  Options Database Keys:
. -svm_warm_start - use warm start in cross-validation

  Level: advanced

.seealso PermonSVMType, SVMGetLossType()
@*/
PetscErrorCode SVMSetWarmStart(SVM svm,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveBool(svm,flg,2);
  svm->warm_start = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetUp"
/*@
  SVMSetUp - Sets up the internal data structures for the SVM.

  Collective on SVM

  Input Parameter:
. svm - SVM context

  Level: developer

.seealso SVMCreate(), SVMTrain(), SVMReset(), SVMDestroy(), SVM
@*/
PetscErrorCode SVMSetUp(SVM svm)
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  if (svm->setupcalled) PetscFunctionReturnI(0);

  TRY( svm->ops->setup(svm) );
  svm->setupcalled = PETSC_TRUE;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMView"
/*@
  SVMView - Views classification model details.

  Input Parameters:
+ svm - SVM context
- v - visualization context

  Level: beginner

  Options Database Keys:
. -svm_view - Prints info on classification model at conclusion of SVMTest()

.seealso PetscViewer
@*/
PetscErrorCode SVMView(SVM svm,PetscViewer v)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  if (svm->ops->view) {
    TRY( svm->ops->view(svm,v) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMViewScore"
/*@
  SVMView - Views performance score of model.

  Input Parameters:
+ svm - SVM context
- v - visualization context

  Level: beginner

  Options Database Keys:
. -svm_view_score - Prints info on classification model at conclusion of SVMTest()

.seealso PetscViewer
@*/
PetscErrorCode SVMViewScore(SVM svm,PetscViewer v)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  if (svm->ops->viewscore) {
    TRY( svm->ops->viewscore(svm,v) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetTrainingDataset"
/*@
  SVMSetTrainingDataset - Sets the training samples and labels.

  Not Collective

  Input Parameter:
+ svm - SVM context
. Xt_training - samples data
- y - known labels of training samples

  Level: beginner

.seealso SVMGetTrainingDataset(), SVMSetTestDataset(), SVMGetTestDataset()
@*/
PetscErrorCode SVMSetTrainingDataset(SVM svm,Mat Xt_training,Vec y_training)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  TRY( PetscTryMethod(svm,"SVMSetTrainingDataset_C",(SVM,Mat,Vec),(svm,Xt_training,y_training)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetTrainingDataset"
/*@
  SVMGetTrainingDataset - Returns the training samples and labels.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
+ Xt_training - training samples
- y_training - known labels of training samples

  Level: beginner

.seealso SVMSetTrainingDataset(), SVMSetTestDataset(), SVMGetTestDataset()
@*/
PetscErrorCode SVMGetTrainingDataset(SVM svm,Mat *Xt_training,Vec *y_training)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  TRY( PetscUseMethod(svm,"SVMGetTrainingDataset_C",(SVM,Mat *,Vec *),(svm,Xt_training,y_training)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetTestDataset"
/*@
  SVMSetTrainingDataset - Sets the test samples and labels.

  Not Collective

  Input Parameter:
+ svm - SVM context
. Xt_test - test samples
- y_test - known labels of test samples

  Level: beginner

.seealso SVMSetTrainingDataset(), SVMGetTrainingDataset(), SVMGetTestDataset()
@*/
PetscErrorCode SVMSetTestDataset(SVM svm,Mat Xt_test,Vec y_test)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(Xt_test,MAT_CLASSID,2);
  PetscCheckSameComm(svm,1,Xt_test,2);
  PetscValidHeaderSpecific(y_test,VEC_CLASSID,3);
  PetscCheckSameComm(svm,1,y_test,3);

  TRY( MatDestroy(&svm->Xt_test) );
  svm->Xt_test = Xt_test;
  TRY( PetscObjectReference((PetscObject) Xt_test) );

  TRY( VecDestroy(&svm->y_test) );
  svm->y_test = y_test;
  TRY( PetscObjectReference((PetscObject) y_test) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetTestDataset"
/*@
  SVMGetTestDataset - Returns the test samples and labels.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
+ Xt_test - test samples
- y_test - known labels of test samples

  Level: beginner

.seealso SVMSetTrainingDataset(), SVMGetTrainingDataset(), SVMSetTestDataset()
@*/
PetscErrorCode SVMGetTestDataset(SVM svm,Mat *Xt_test,Vec *y_test)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  if (Xt_test) {
    PetscValidPointer(Xt_test,2);
    *Xt_test = svm->Xt_test;
  }

  if (y_test) {
    PetscValidPointer(y_test,2);
    *y_test = svm->y_test;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SVMSetAutoPostTrain"
/*@
  SVMSetAutoPostTrain - Sets auto post train flag.

  Logically Collective on SVM

  Input Parameter:
. svm - SVM context

  Output Parameter:
. flg - flag

  Level: developer

.seealso SVMPostTrain()
@*/
PetscErrorCode SVMSetAutoPostTrain(SVM svm,PetscBool flg)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveBool(svm,flg,2);

  svm->autoposttrain = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMTrain"
/*@
  SVMTrain - Trains a classification model on the basis of training samples.

  Collective on SVM

  Input Parameters:
. svm - SVM context

  Level: beginner

.seealso SVMPrectict(), SVMTest(), SVMGetSeparatingHyperplane()
@*/
PetscErrorCode SVMTrain(SVM svm)
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->train(svm) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMPostTrain"
/*@
  SVMPostTrain - Applies post train function.

  Collective on SVM

  Input Parameter:
. svm - SVM context

  Level: advanced

.seealso SVMTrain()
@*/
PetscErrorCode SVMPostTrain(SVM svm)
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->posttrain(svm) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetSeparatingHyperplane"
/*@
  SVMSetSeparatingHyperplane - Sets the classifier (separator) <w,x> + b = 0.

  Not Collective

  Input Parameters:
+ svm - SVM context
. w - the normal vector to the separating hyperplane
- b - the offset of the hyperplane

  Level: beginner

.seealso SVMSetBias()
@*/
PetscErrorCode SVMSetSeparatingHyperplane(SVM svm,Vec w,PetscReal b)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  TRY( PetscTryMethod(svm,"SVMSetSeparatingHyperplane_C",(SVM,Vec,PetscReal),(svm,w,b)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetSeparatingHyperplane"
/*@
  SVMGetSeparatingHyperplane - Returns the linear classification model, i.e. <w,x> + b = 0, computed by PermonSVMTrain().

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameters:
+ w - the normal vector to the separating hyperplane
- b - the offset of the hyperplane

  Level: beginner

.seealso: SVMTrain(), SVMPredict(), SVMTest()
@*/
PetscErrorCode SVMGetSeparatingHyperplane(SVM svm,Vec *w,PetscReal *b)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  TRY( PetscUseMethod(svm,"SVMGetSeparatingHyperplane_C",(SVM,Vec *,PetscReal *),(svm,w,b)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetBias"
/*@
  SVMSetBias - Sets the bias (b) of the linear classification model, i.e. <w,x> + b.

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- bias - the bias (b) of the linear classification model

  Level: intermediate

.seealso SVMGetBias(), SVMSetMod()
@*/
PetscErrorCode SVMSetBias(SVM svm,PetscReal bias)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  TRY( PetscTryMethod(svm,"SVMSetBias_C",(SVM,PetscReal),(svm,bias)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__
/*@
  SVMSetBias - Returns the bias (b) of the linear classification model, i.e. <w,x> + b.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. bias - the bias (b) of the linear classification model

  Level: intermediate

.seealso SVMSetBias(), SVMSetMod()
@*/
PetscErrorCode SVMGetBias(SVM svm,PetscReal *bias)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  TRY( PetscUseMethod(svm,"SVMGetBias_C",(SVM,PetscReal *),(svm,bias)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMPredict"
/*@
  SVMPredict - Predicts labels of tested samples.

  Collective on SVM

  Input Parameters:
+ svm - SVM context
- Xt_test - matrix of tested samples

  Output Parameter:
. y - predicted labels of tested samples

  Level: beginner

.seealso: SVMTrain(), SVMTest()
@*/
PetscErrorCode SVMPredict(SVM svm,Mat Xt_pred,Vec *y_pred)
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->predict(svm,Xt_pred,y_pred) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMTest"
/*@
  SVMTest - Tests quality of classification model.

  Collective on SVM

  Input Parameters:
. svm - SVM context

  Level: beginner

.seealso SVMSetTestDataset(), SVMTrain(), SVMPredict()
@*/
PetscErrorCode SVMTest(SVM svm)
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->test(svm) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetModelScore"
/*@
  SVMGetModelScore - Returns the model performance score of specified score_type.

  Not Collective

  Input Parameters:
+ svm - SVM context
- score_type - type of model score

  Output Parameter:
. s - score value

.seealso ModelScore
@*/
PetscErrorCode SVMGetModelScore(SVM svm,ModelScore score_type,PetscReal *s)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  TRY( PetscUseMethod(svm,"SVMGetModelScore_C",(SVM,ModelScore,PetscReal *),(svm,score_type,s)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearch"
/*@
  SVMGridSearch - Chooses the best value of penalty C from manually specified set.

  Collective on SVM

  Input Parameter:
+ svm - SVM context

  Level: beginner

.seealso: SVMCrossValidation(), SVMSetLogCBase(), SVMSetLogCMin(), SVMSetLogCMax(), SVM
@*/
PetscErrorCode SVMGridSearch(SVM svm)
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->gridsearch(svm) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetCrossValidationScoreType"
/*@
  SVMSetCrossValidationScoreType - Sets score type for evaluating performance of model during cross validation.

  Logically Colective on SVM

  Input Parameters:
+ svm - SVM context
- type - type of model score

.seealso SVMGetCrossValidationScoreType(), ModelScore
@*/
PetscErrorCode SVMSetCrossValidationScoreType(SVM svm,ModelScore type)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(svm,type,2);
  svm->cv_model_score = type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetCrossValidationScoreType"
/*@
  SVMGetCrossValidationScoreType - Returns score type for evaluating performance of model during cross validation.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. type - type of model score

.seealso SVMGetCrossValidationScoreType(), ModelScore
@*/
PetscErrorCode SVMGetCrossValidationScoreType(SVM svm,ModelScore *type)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidPointer(type,2);
  *type = svm->cv_model_score;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMCrossValidation"
/*@
  SVMCrossValidation - Performs k-folds cross validation.

  Collective on SVM

  Input Parameters:
+ svm - SVM context
. c_arr - manually specified set of penalty C values
- m - size of c_arr

  Output Parameter:
. score - array of scores for each penalty C

  Level: beginner

.seealso: SVMGridSearch(), SVMSetNfolds(), SVM
@*/
PetscErrorCode SVMCrossValidation(SVM svm,PetscReal c_arr[],PetscInt m,PetscReal score[])
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->crossvalidation(svm,c_arr,m,score) );
  PetscFunctionReturnI(0);
}
