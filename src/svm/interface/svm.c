
#include <permon/private/svmimpl.h>

PetscClassId SVM_CLASSID;

#undef __FUNCT__
#define __FUNCT__ "SVMCreate"
/*@
  SVMCreate - Creates instance of Support Vector Machine classifier

  Input Parameter:
. comm - MPI comm

  Output Parameter:
. svm_out - pointer to created SVM

  Level: beginner
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

  svm->nfolds     = 5;
  svm->warm_start = PETSC_FALSE;

  svm->setupcalled          = PETSC_FALSE;
  svm->setfromoptionscalled = PETSC_FALSE;
  svm->autoposttrain        = PETSC_TRUE;
  svm->posttraincalled      = PETSC_FALSE;

  *svm_out = svm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMReset"
/*@
  SVMReset - Resets a SVM context

  Collective on SVM

  Input Parameter:
. svm - the SVM

  Level: beginner
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

  TRY( (*svm->ops->reset)(svm) );

  svm->setupcalled          = PETSC_FALSE;
  svm->setfromoptionscalled = PETSC_FALSE;
  svm->autoposttrain        = PETSC_TRUE;
  svm->posttraincalled      = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMDestroyDefault"
/*@
   SVMDestroyDefault - Destroys SVM content

   Input parameter:
.  svm - instance of SVM

   Developers Note: This is PETSC_EXTERN because it may be used by user written plugin SVM implementations

   Level: developer
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
   SVMDestroy - Destroys SVM context

   Collective on SVM

   Input Parameters:
.  svm - SVM context

   Level: beginner
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
  SVMSetFromOptions - Sets SVM options from the options database

  Input Parameter:
. svm - the SVM

  Level: beginner
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
    TRY(SVMSetWarmStart(svm,warm_start) );
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

  Collective on SVM

  Input parameters:
+ svm - instance of SVM
- qps - instance of QPS

  Level: advanced
@*/
PetscErrorCode SVMSetQPS(SVM svm,QPS qps)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(svm,QPS_CLASSID,2);

  TRY( PetscTryMethod(svm,"SVMSetQPS_C",(SVM,QPS),(svm,qps)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetQPS"
/*@
  SVMGetQPS - Gets the QPS.

  Input Parameter:
. svm - the SVM

  Output Parameter:
. qps - instance of QPS

  Level: advanced
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
   SVMSetNfolds - Sets the number of folds

  Input Parameters:
+ svm - the SVM
- C - C parameter

  Level: beginner
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
  SVMGetNfolds - Gets the number of folds

  Input Parameter:
. svm - the SVM

  Output Parameter:
. nfolds - number of folds

  Level: beginner
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
  SVMSetC - Sets the penalty C.

  Collective on SVM

  Input Parameters:
+ svm - the SVM
- C - penalty C

  Level: beginner
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
  SVMGetC - Gets the penalty C.

  Input Parameter:
. svm - the SVM

  Output Parameter:
. C - penalty C

  Level: beginner
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
  SVMSetLogCBase - Sets the step C penalty value

  Input Parameters:
+ svm - the SVM
- LogCBase - step C penalty value

  Level: beginner

.seealso SVMSetC(), SVMSetLogCMin(), SVMSetLogCMax()
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
  SVMGetC - Gets the step C penalty value

  Input Parameter:
. svm - the SVM

  Output Parameter:
. LogCBase - step C penalty value

  Level: beginner

.seealso SVMGetC(), SVMGetLogCMin(), SVMGetLogCMax()
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
  SVMSetLogCMin - Sets the minimal log C penalty value

  Input Parameter:
+ svm - the SVM
- LogCMin - minimal log C penalty value

  Level: beginner

.seealso SVMSetC(), SVMSetLogCBase(), SVMSetLogCMax()
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
  SVMGetLogCMin - Gets the minimal log C penalty value

  Input Parameter:
. svm - the SVM

  Output Parameter:
. LogCMin - minimal log C penalty value

  Level: beginner

.seealso SVMGetC(), SVMGetLogCBase(), SVMGetLogCMax()
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
  SVMSetLogCMax - Sets the maximal log C penalty value

  Input Parameters:
+ svm - the SVM
- LogCMax - maximal log C penalty value

  Level: beginner

.seealso SVMSetC(), SVMSetLogCMin()
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
  SVMGetLogCMax - Gets the maximal log C penalty value

  Input Parameter:
. svm - the SVM

  Output Parameter:
. LogCMax - maximal log C penalty value

  Level: beginner

.seealso SVMGetC(), SVMGetLogCMax()
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
   SVMSetLossType - Sets the type of the hinge loss function

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
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(svm,type,2);
  if (type != svm->loss_type) {
    svm->loss_type = type;
    svm->setupcalled = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetMod"
/*@
   SVMSetMod - Sets type of SVM formulation

   Logically Collective on SVM

   Input Parameters:
+  svm - the SVM
-  mod - type of SVM formulation
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
   SVMGetMod - Gets type of SVM formulation

   Input Parameter:
.  svm - the SVM

   Output Parameter:
.  mod - type of SVM formulation
@*/
PetscErrorCode SVMGetMod(SVM svm,PetscInt *mod)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( PetscTryMethod(svm,"SVMGetMod_C",(SVM,PetscInt *),(svm,mod)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetLossType"
/*@
   SVMGetLossType - Gets the type of the loss function

   Not Collective

   Input Parameter:
.  svm - the SVM

   Output Parameter:
.  type - type of loss function

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
#define __FUNCT__ "SVMSetOptionsPrefix"
/*@
  SVMSetOptionsPrefix - Sets the prefix used for searching for all options of SVM QPS solver in the database

  Collective on SVM

  Input Parameters:
+ SVM - the SVM
- prefix - the prefix string

  Level: developer

.seealso SVMGetOptionsPrefix(), SVM, QPS
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
  SVMAppendOptionsPrefix - Sets the prefix used for searching for all options of SVM QPS in the database

  Collective on SVM

  Input Parameters:
+ SVM - the SVM
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
  SVMGetOptionsPrefix - Gets the SVM QPS solver prefix

  Input Parameters:
. SVM - the SVM

  Output Parameters:
. prefix - pointer to the prefix string used is returned

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
+ svm - the SVM
- flg - use warm start in cross-validation

  Options Database Keys:
. -svm_warm_start - use warm start in cross-validation

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
#define __FUNCT__ "SVMSetUp"
/*@
  SVMSetUp - Sets up the internal data structures for the SVM.

  Collective on SVM

  Input parameter:
. svm - instance of SVM

  Level: developer

.seealso SVMCreate(), SVMTrain(), SVMReset(), SVMDestroy(), SVM
@*/
PetscErrorCode SVMSetUp(SVM svm)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  if (svm->setupcalled) PetscFunctionReturn(0);

  TRY( svm->ops->setup(svm) );
  svm->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMView"
/*@
   SVMView - Views classification model details

   Input Parameters:
+  svm - the SVM
-  v - visualization context

   Level: beginner
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
#define __FUNCT__ "SVMSetTrainingDataset"
/*@
   SVMSetTrainingDataset - Sets the training samples and labels.

   Input Parameter:
+  svm - the SVM
.  Xt_training - samples data
-  y - known labels of training samples

   Level: beginner
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
   SVMSetTrainingDataset - Sets the training samples and labels.

   Input Parameter:
.  svm - the SVM

   Output Parameter:
+  Xt_training - training samples
-  y_training - known labels of training samples

   Level: beginner
@*/
PetscErrorCode SVMGetTrainingDataset(SVM svm,Mat *Xt_training,Vec *y_training)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  TRY( PetscUseMethod(svm,"SVMGetTrainingDataset_C",(SVM,Mat *,Vec *),(svm,Xt_training,y_training)) );
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SVMSetAutoPostTrain"
/*@
  SVMSetAutoPostTrain - Sets post train flag.

  Collective on SVM

  Input Parameter:
. svm - the SVM

  Output Parameter:
. flg - flag

  Level: developer
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
  SVMTrain - Creates a classification model on the basis of training samples

  Input Parameters:
. svm - the SVM

  Level: beginner
@*/
PetscErrorCode SVMTrain(SVM svm)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->train(svm) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMPostTrain"
/*@
  SVMPostTrain - Applies post train function

  Collective on SVM

  Input Parameter:
. svm - the SVM

  Level: advanced
@*/
PetscErrorCode SVMPostTrain(SVM svm)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->posttrain(svm) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetSeparatingHyperplane"
/*@
  SVMSetSeparatingHyperplane - Sets the classifier (separator) w*x - b = 0

  Collective on SVM

  Input Parameters:
+ svm - the SVM
. w - the normal vector to the separating hyperplane
- b - the offset of the hyperplane

  Level: beginner
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
  SVMGetSeparatingHyperplane - Returns the classifier (separator) w*x - b = 0 computed by PermonSVMTrain()

  Not Collective

  Input Parameter:
. svm - the SVM context

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
PetscErrorCode SVMSetBias(SVM svm,PetscReal bias)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  TRY( PetscTryMethod(svm,"SVMSetBias_C",(SVM,PetscReal),(svm,bias)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__
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
  SVMPredict - Predicts labels of tested samples

  Input Parameter:
+ svm - the SVM
- Xt_test - matrix of tested samples

  Output Parameter:
. y - predicted labels of tested samples

  Level: beginner

.seealso: SVMTrain(), SVMTest()
@*/
PetscErrorCode SVMPredict(SVM svm,Mat Xt_pred,Vec *y_pred)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->predict(svm,Xt_pred,y_pred) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMTest"
/*@
  SVMTest - Tests quality of classification model

  Input Parameters:
+ svm - the SVM
. Xt_test - matrix of tested samples
- y_known - known labels of tested samples

  Output Parameters:
+ N_all - number of all tested samples
- N_eq  - number of right classified samples

  Level: beginner
@*/
PetscErrorCode SVMTest(SVM svm,Mat Xt_test,Vec y_known,PetscInt *N_all,PetscInt *N_eq)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->test(svm,Xt_test,y_known,N_all,N_eq) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearch"
/*@
  SVMGridSearch - Chooses best C penalty from manually specified set

  Input Parameter:
+ svm - the SVM

  Level: beginner

.seealso: SVMCrossValidation(), SVMSetLogCBase(), SVMSetLogCMin(), SVMSetLogCMax(), SVM
@*/
FLLOP_EXTERN PetscErrorCode SVMGridSearch(SVM svm)
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->gridsearch(svm) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMCrossValidation"
/*@
  SVMCrossValidation - Performs k-folds cross validation

  Input Parameter:
+ svm - the SVM

  Level: beginner

.seealso: SVMGridSearch(), SVMSetNfolds(), SVM
@*/
FLLOP_EXTERN PetscErrorCode SVMCrossValidation(SVM svm,PetscReal c_arr[],PetscInt m,PetscReal score[])
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->crossvalidation(svm,c_arr,m,score) );
  PetscFunctionReturnI(0);
}
