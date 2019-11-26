
#include <permon/private/svmimpl.h>
#include "../utils/io.h"

PetscClassId  SVM_CLASSID;
PetscLogEvent SVM_LoadDataset;

const char *const ModelScores[]={"accuracy","precision","sensitivity","F1","mcc","aucroc","G1","ModelScore","model_",0};
const char *const CrossValidationTypes[]={"kfold","stratified_kfold","CrossValidationType","cv_",0};

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
  svm->Cp         = 2.;
  svm->Cp_old     = 2.;
  svm->Cn         = 1.;
  svm->Cn_old     = 1.;
  svm->logC_base   = 2.;
  svm->logC_start    = -2.;
  svm->logC_end    = 2.;
  svm->logCp_base  = 2.;
  svm->logCp_start   = -2.;
  svm->logCp_end   = 2.;
  svm->logCn_base  = 2.;
  svm->logCn_start   = -2.;
  svm->logCn_end   = 2.;

  svm->loss_type  = SVM_L1;
  svm->svm_mod    = 2;

  TRY( PetscMemzero(svm->hopt_score_types,7 * sizeof(ModelScore)) );

  svm->penalty_type        = 1;
  svm->hyperoptset         = PETSC_FALSE;
  svm->hopt_score_types[0] = MODEL_ACCURACY;
  svm->hopt_nscore_types   = 1;
  svm->cv_type             = CROSS_VALIDATION_KFOLD;
  svm->nfolds              = 5;
  svm->warm_start          = PETSC_FALSE;

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

  TRY( MatDestroy(&svm->Xt_test) );
  svm->Xt_test = NULL;
  TRY( VecDestroy(&svm->y_test) );
  svm->y_test  = NULL;

  TRY( (*svm->ops->reset)(svm) );

  svm->setupcalled          = PETSC_FALSE;
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
  PetscErrorCode      ierr;


  PetscBool           hyperoptset;
  ModelScore          hyperopt_score_types[7];
  PetscInt            n;

  PetscInt            penalty_type;
  PetscReal           C,logC_min,logC_max,logC_base;
  PetscBool           flg,warm_start;

  SVMLossType         loss_type;
  PetscInt            svm_mod;

  CrossValidationType cv_type;
  PetscInt            nfolds;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  ierr = PetscObjectOptionsBegin((PetscObject)svm);CHKERRQ(ierr);
  TRY( PetscOptionsInt("-svm_binary_mod","","SVMSetMod",svm->svm_mod,&svm_mod,&flg) );
  if (flg) {
    TRY( SVMSetMod(svm,svm_mod) );
  }
  TRY( PetscOptionsInt("-svm_penalty_type","Set type of misclasification error penalty.","SVMSetPenaltyType",svm->penalty_type,&penalty_type,&flg) );
  if (flg) {
    TRY( SVMSetPenaltyType(svm,penalty_type) );
  }
  TRY( PetscOptionsBool("-svm_hyperopt","Specify whether hyperparameter optimization will be performed.","SVMSetHyperOpt",svm->hyperoptset,&hyperoptset,&flg) );
  if (flg) {
    TRY( SVMSetHyperOpt(svm,hyperoptset) );
  }
  TRY( PetscOptionsReal("-svm_C","Set SVM C (C).","SVMSetC",svm->C,&C,&flg) );
  if (flg) {
    TRY( SVMSetC(svm,C) );
  }
  TRY( PetscOptionsReal("-svm_Cp","Set SVM Cp (Cp).","SVMSetCp",svm->Cp,&C,&flg) );
  if (flg) {
    TRY( SVMSetCp(svm,C) );
  }
  TRY( PetscOptionsReal("-svm_Cn","Set SVM Cn (Cn).","SVMSetCn",svm->Cn,&C,&flg) );
  if (flg) {
    TRY( SVMSetCn(svm,C) );
  }
//  TRY( PetscOptionsReal("-svm_logC_min","Set SVM minimal C value (logC_start).","SVMSetLogCMin",svm->logC_start,&logC_min,&flg) );
//  if (flg) {
//    TRY( SVMSetLogCMin(svm,logC_min) );
//  }
//  TRY( PetscOptionsReal("-svm_logC_max","Set SVM maximal C value (logC_end).","SVMSetLogCMax",svm->logC_end,&logC_max,&flg) );
//  if (flg) {
//    TRY( SVMSetLogCMax(svm,logC_max) );
//  }
//  TRY( PetscOptionsReal("-svm_logC_base","Set power base of SVM parameter C (logC_base).","SVMSetLogCBase",svm->logC_base,&logC_base,&flg) );
//  if (flg) {
//    TRY( SVMSetLogCBase(svm,logC_base) );
//  }
//  TRY( PetscOptionsReal("-svm_logCp_min","Set SVM minimal Cp value (logCp_start).","SVMSetLogCpMin",svm->logCp_start,&logC_min,&flg) );
//  if (flg) {
//    TRY( SVMSetLogCpMin(svm,logC_min) );
//  }
//  TRY( PetscOptionsReal("-svm_logCp_max","Set SVM maximal Cp value (logCp_end).","SVMSetLogCpMax",svm->logCp_end,&logC_max,&flg) );
//  if (flg) {
//    TRY( SVMSetLogCpMax(svm,logC_max) );
//  }
//  TRY( PetscOptionsReal("-svm_logCp_base","Set power base of SVM parameter Cp (logCp_base).","SVMSetLogCpBase",svm->logCp_base,&logC_base,&flg) );
//  if (flg) {
//    TRY( SVMSetLogCpBase(svm,logC_base) );
//  }
//  TRY( PetscOptionsReal("-svm_logCn_min","Set SVM minimal Cn value (logCn_start).","SVMSetLogCnMin",svm->logCn_start,&logC_min,&flg) );
//  if (flg) {
//    TRY( SVMSetLogCnMin(svm,logC_min) );
//  }
//  TRY( PetscOptionsReal("-svm_logCn_max","Set SVM maximal Cn value (logCn_end).","SVMSetLogCnMax",svm->logCn_end,&logC_max,&flg) );
//  if (flg) {
//    TRY( SVMSetLogCnMax(svm,logC_max) );
//  }
//  TRY( PetscOptionsReal("-svm_logCn_base","Set power base of SVM parameter Cn (logCn_base).","SVMSetLogCnBase",svm->logCn_base,&logC_base,&flg) );
//  if (flg) {
//    TRY( SVMSetLogCnBase(svm,logC_base) );
//  }
  TRY( PetscOptionsInt("-svm_nfolds","Set number of folds (nfolds).","SVMSetNfolds",svm->nfolds,&nfolds,&flg) );
  if (flg) {
    TRY( SVMSetNfolds(svm,nfolds) );
  }
  TRY( PetscOptionsEnum("-svm_loss_type","Specify the loss function for soft-margin SVM (non-separable samples).","SVMSetNfolds",SVMLossTypes,(PetscEnum)svm->loss_type,(PetscEnum*)&loss_type,&flg) );
  if (flg) {
    TRY( SVMSetLossType(svm,loss_type) );
  }
  n = 7;
  TRY( PetscOptionsEnumArray("-svm_hyperopt_score_types","Specify the score types for evaluating performance of model during hyperparameter optimization.","SVMSetHyperOptScoreTypes",ModelScores,(PetscEnum *) hyperopt_score_types,&n,&flg) );
  if (flg) {
    TRY( SVMSetHyperOptScoreTypes(svm,n,hyperopt_score_types) );
  }
  TRY( PetscOptionsEnum("-svm_cv_type","Specify the type of cross validation.","SVMSetCrossValidationType",CrossValidationTypes,(PetscEnum)svm->cv_type,(PetscEnum*)&cv_type,&flg) );
  if (flg) {
    TRY( SVMSetCrossValidationType(svm,cv_type) );
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
#define __FUNCT__ "SVMSetPenaltyType"
/*@
  SVMSetPenaltyType - Sets type of penalty that penalizes misclassification error.

  Collective on SVM

  Input Parameters:
+ svm - SVM context
- type - penalty type

.seealso SVMSetC(), SVMSetCp(), SVMSetCn(), SVM
@*/
PetscErrorCode SVMSetPenaltyType(SVM svm,PetscInt type)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveInt(svm,type,2);
  if (svm->penalty_type == type) PetscFunctionReturn(0);

  if (type != 1 && type != 2) FLLOP_SETERRQ1(((PetscObject) svm)->comm,PETSC_ERR_SUP,"Type of penalty (%d) is not supported. It must be 1 or 2",type);

  svm->penalty_type = type;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetPenaltyType"
/*@
  SVMGetPenaltyType - Returns type of penalty that penalizes misclassification error.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. type - penalty type

.seealso SVMSetC(), SVMSetCp(), SVMSetCn(), SVM
@*/
PetscErrorCode SVMGetPenaltyType(SVM svm,PetscInt *type)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(type,2);
  *type = svm->penalty_type;
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

  if (svm->C == C) {
    svm->C_old = C;
    PetscFunctionReturn(0);
  }

  if (C <= 0) {
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
#define __FUNCT__ "SVMSetCp"
/*@
  SVMSetCp - Sets the value of penalty C for positive samples.

  Collective on SVM

  Input Parameters:
+ svm - SVM context
- Cp - the value of penalty C for positive samples

  Level: intermediate

.seealso SVMGetCp(), SVMSetCn(), SVMGetCn(), SVMGridSearch()
@*/
PetscErrorCode SVMSetCp(SVM svm,PetscReal Cp)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,Cp,2);

  if (svm->Cp == Cp) {
    svm->Cp_old = Cp;
    PetscFunctionReturn(0);
  }

  if (Cp <= 0) {
    FLLOP_SETERRQ(((PetscObject) svm)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");
  }

  svm->Cp_old = svm->Cp;
  svm->Cp     = Cp;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetCp"
/*@
  SVMGetCp - Returns the value of penalty C for positive samples.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. Cp - the value of penalty C for positive samples

  Level: intermediate

.seealso SVMSetCp(), SVMSetCn(), SVMGetCn(), SVMGridSearch()
@*/
PetscErrorCode SVMGetCp(SVM svm,PetscReal *Cp)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(Cp,2);
  *Cp = svm->Cp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetCn"
/*@
  SVMSetCn - Sets the value of penalty C for negative samples.

  Collective on SVM

  Input Parameters:
+ svm - SVM context
- Cp - the value of penalty C for negative samples

  Level: intermediate

.seealso SVMSetCp(), SVMGetCp(), SVMGetCn(), SVMGridSearch()
@*/
PetscErrorCode SVMSetCn(SVM svm,PetscReal Cn)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,Cn,2);

  if (svm->Cn == Cn) {
    svm->Cn_old = Cn;
    PetscFunctionReturn(0);
  }

  if (Cn <= 0) {
    FLLOP_SETERRQ(((PetscObject) svm)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");
  }

  svm->Cn_old = svm->Cn;
  svm->Cn     = Cn;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetCn"
/*@
  SVMGetCn - Returns the value of penalty C for negative samples.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. Cn - the value of penalty C for negative samples

  Level: intermediate

.seealso SVMSetCp(), SVMSetCn(), SVMSetCn(), SVMGridSearch()
@*/
PetscErrorCode SVMGetCn(SVM svm,PetscReal *Cn)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(Cn,2);
  *Cn = svm->Cn;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetPenalty"
/*@
  SVMSetPenalty - Sets C or Cp and Cn values.

  Collective on SVM

  Input Parameters:
+ svm - SVM context
. m - number of penalties
- p - values of penalties

  Level: intermediate

  Notes:
    m == 1: C = p[0] or Cp = p[0] and Cn = p[0]
    m == 2: Cp = p[0] and Cn = p[1]

.seealso SVMSetC(), SVMSetCp(), SVMSetCn(), SVM
@*/
PetscErrorCode SVMSetPenalty(SVM svm,PetscInt m,PetscReal p[])
{
  PetscInt penalty_type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  if (m > 2) FLLOP_SETERRQ(((PetscObject) svm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Argument must be 1 or 2");
  PetscValidLogicalCollectiveInt(svm,m,2);
  PetscValidLogicalCollectiveReal(svm,p[0],3);
  if (m == 2) PetscValidLogicalCollectiveReal(svm,p[1],3);

  TRY( SVMGetPenaltyType(svm,&penalty_type) );

  if (penalty_type == 1) {
    TRY( SVMSetC(svm,p[0]) );
  } else {
    if (m == 1) {
      TRY( SVMSetCp(svm,p[0]) );
      TRY( SVMSetCn(svm,p[0]) );
    } else {
      TRY( SVMSetCp(svm,p[0]) );
      TRY( SVMSetCn(svm,p[1]) );
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchSetBaseLogC"
/*@
  TODO edit and improve comment regarding to grid-search stride concept
  SVMGridSearchSetBaseLogC - Sets the value of penalty C step.

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- logC_base - the value of penalty C step

  Level: beginner

.seealso SVMSetC(), SVMSetLogBase(), SVMSetLogCMin(), SVMSetLogCMax(), SVMGridSearch()
@*/
PetscErrorCode SVMGridSearchSetBaseLogC(SVM svm,PetscReal logC_base)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logC_base,2);

  if (logC_base <= 0) FLLOP_SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");

  svm->logC_base = logC_base;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchGetBaseLogC"
/*@
  TODO edit and improve comment regarding to grid-search stride concept
  SVMGridSearchGetBaseLogC - Returns the value of penalty C step.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. logC_base - the value of penalty C step

  Level: beginner

.seealso SVMGetC(), SVMGetLogCMin(), SVMGetLogCMax(), SVMGridSearch()
@*/
PetscErrorCode SVMGridSearchGetBaseLogC(SVM svm,PetscReal *logC_base)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(logC_base,2);

  *logC_base = svm->logC_base;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetLogCMin"
/*@
  SVMSetLogCMin - Sets the minimum value of log C penalty.

  Logically Colective on SVM

  Input Parameter:
+ svm - SVM context
- logC_start - the minimum value of log C penalty

  Level: beginner

.seealso SVMSetC(), SVMSetLogCBase(), SVMSetLogCMax(), SVMGridSearch()
@*/
PetscErrorCode SVMSetLogCMin(SVM svm,PetscReal logC_start)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logC_start,2);
  svm->logC_start = logC_start;
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
. logC_start - the minimum value of log C penalty

  Level: beginner

.seealso SVMGetC(), SVMGetLogCBase(), SVMGetLogCMax(), SVMGridSearch()
@*/
PetscErrorCode SVMGetLogCMin(SVM svm,PetscReal *logC_start)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm, SVM_CLASSID, 1);
  PetscValidRealPointer(logC_start, 2);
  *logC_start = svm->logC_start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetLogCMax"
/*@
  SVMSetLogCMax - Sets the maximum value of log C penalty.

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- logC_end - the maximum value of log C penalty

  Level: beginner

.seealso SVMSetC(), SVMSetLogCBase(), SVMSetLogCMin(), SVMGridSearch()
@*/
PetscErrorCode SVMSetLogCMax(SVM svm,PetscReal logC_end)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logC_end,2);
  svm->logC_end = logC_end;
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
. logC_end - the maximum value of log C penalty

  Level: beginner

.seealso SVMGetC(), SVMGetLogCBase(), SVMGetLogCMin(), SVMGridSearch()
@*/
PetscErrorCode SVMGetLogCMax(SVM svm,PetscReal *logC_end)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(logC_end,2);
  *logC_end = svm->logC_end;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchSetPosBaseLogC"
/*@
  TODO edit and improve comment regarding to grid-search stride concept
  SVMGridSearchSetPosBaseLogC - Sets the value of penalty Cp step.

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- logCp_base - the value of penalty Cp step

  Level: beginner

.seealso SVMSetCp(), SVMGetLogCpBase(), SVMSetLogCpMin(), SVMSetLogCpMax(), SVMGridSearch()
@*/
PetscErrorCode SVMGridSearchSetPosBaseLogC(SVM svm,PetscReal logCp_base)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logCp_base,2);

  if (logCp_base <= 0) FLLOP_SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");

  svm->logCp_base = logCp_base;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchGetPosBaseLog"
/*@
  TODO edit and improve comment regarding to grid-search stride concept
  SVMGridSearchGetPosBaseLogC - Returns the value of penalty Cp step.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. logCp_base - the value of penalty Cp step

  Level: beginner

.seealso SVMGetCp(), SVMGetLogCpMin(), SVMGetLogCpMax(), SVMGridSearch()
@*/
PetscErrorCode SVMGridSearchGetPosBaseLogC(SVM svm,PetscReal *logCp_base)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(logCp_base,2);

  *logCp_base = svm->logCp_base;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetLogCpMin"
/*@
  SVMSetLogCpMin - Sets the minimum value of log Cp penalty.

  Logically Collective on SVM

  Input Parameter:
+ svm - SVM context
- logCp_start - the minimum value of log C penalty

  Level: beginner

.seealso SVMSetCp(), SVMSetLogCpBase(), SVMSetLogCpMax(), SVMGridSearch()
@*/
PetscErrorCode SVMSetLogCpMin(SVM svm,PetscReal logCp_start)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logCp_start,2);
  svm->logCp_start = logCp_start;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetLogCpMin"
/*@
  SVMGetLogCpMin - Returns the minimum value of log Cp penalty.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. logCp_start - the minimum value of log Cp penalty

  Level: beginner

.seealso SVMGetCp(), SVMGetLogCpBase(), SVMGetLogCpMax(), SVMGridSearch()
@*/
PetscErrorCode SVMGetLogCpMin(SVM svm,PetscReal *logCp_start)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(logCp_start,2);
  *logCp_start = svm->logCp_start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetLogCpMax"
/*@
  SVMSetLogCpMax - Sets the maximum value of log Cp penalty.

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- logCp_end - the maximum value of log Cp penalty

  Level: beginner

.seealso SVMSetCp(), SVMSetLogCpBase(), SVMSetLogCpMin(), SVMGridSearch()
@*/
PetscErrorCode SVMSetLogCpMax(SVM svm,PetscReal logCp_end)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logCp_end,2);
  svm->logCp_end = logCp_end;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetLogCpMax"
/*@
  SVMGetLogCpMax - Returns the maximum value of log Cp penalty.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. logC_end - the maximum value of log Cp penalty

  Level: beginner

.seealso SVMGetCp(), SVMGetLogCpBase(), SVMGetLogCpMin(), SVMGridSearch()
@*/
PetscErrorCode SVMGetLogCpMax(SVM svm,PetscReal *logCp_end)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(logCp_end,2);
  *logCp_end = svm->logCp_end;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchSetNegBaseLogC"
/*@
  TODO edit and improve comment regarding to grid-search stride concept
  SVMGridSearchSetNegBaseLogC - Sets the value of penalty Cn step.

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- logCn_base - the value of penalty Cn step

  Level: beginner

.seealso SVMSetCn(), SVMGetLogCnBase(), SVMSetLogCnMin(), SVMSetLogCnMax(), SVMGridSearch()
@*/
PetscErrorCode SVMGridSearchSetNegBaseLogC(SVM svm,PetscReal logCn_base)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logCn_base,2);

  if (logCn_base <= 0) FLLOP_SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");

  svm->logCn_base = logCn_base;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchGetNegBaseLogC"
/*@
  TODO edit and improve comment regarding to grid-search stride concept
  SVMGridSearchGetNegBaseLogC - Returns the value of penalty Cn step.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. logCn_base - the value of penalty Cn step

  Level: beginner

.seealso SVMGetCn(), SVMGetLogCnMin(), SVMGetLogCnMax(), SVMGridSearch()
@*/
PetscErrorCode SVMGridSearchGetNegBaseLogC(SVM svm,PetscReal *logCn_base)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(logCn_base,2);

  *logCn_base = svm->logCn_base;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetLogCnMin"
/*@
  SVMSetLogCnMin - Sets the minimum value of log Cn penalty.

  Logically Collective on SVM

  Input Parameter:
+ svm - SVM context
- logCn_start - the minimum value of log Cn penalty

  Level: beginner

.seealso SVMSetCn(), SVMSetLogCnBase(), SVMSetLogCnMax(), SVMGridSearch()
@*/
PetscErrorCode SVMSetLogCnMin(SVM svm,PetscReal logCn_start)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logCn_start,2);
  svm->logCn_start = logCn_start;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetLogCnMin"
/*@
  SVMGetLogCnMin - Returns the minimum value of log Cn penalty.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. logCn_start - the minimum value of log Cn penalty

  Level: beginner

.seealso SVMGetCn(), SVMGetLogCnBase(), SVMGetLogCnMax(), SVMGridSearch()
@*/
PetscErrorCode SVMGetLogCnMin(SVM svm,PetscReal *logCn_start)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(logCn_start,2);
  *logCn_start = svm->logCn_start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetLogCnMax"
/*@
  SVMSetLogCnMax - Sets the maximum value of log Cn penalty.

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- logCn_end - the maximum value of log Cn penalty

  Level: beginner

.seealso SVMSetCn(), SVMSetLogCnBase(), SVMSetLogCnMin(), SVMGridSearch()
@*/
PetscErrorCode SVMSetLogCnMax(SVM svm,PetscReal logCn_end)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logCn_end,2);
  svm->logCn_end = logCn_end;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetLogCnMax"
/*@
  SVMGetLogCnMax - Returns the maximum value of log Cn penalty.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. logCn_end - the maximum value of log Cn penalty

  Level: beginner

.seealso SVMGetCn(), SVMGetLogCnBase(), SVMGetLogCnMin(), SVMGridSearch()
@*/
PetscErrorCode SVMGetLogCnMax(SVM svm,PetscReal *logCn_end)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(logCn_end,2);
  *logCn_end = svm->logCn_end;
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
  if (svm->svm_mod != mod) {
    svm->svm_mod = mod;
    svm->setupcalled = PETSC_FALSE;
  }
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
  *mod = svm->svm_mod;
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
  PetscBool view;
  PetscViewer v;
  PetscViewerFormat format;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->posttrain(svm) );

  TRY( PetscOptionsGetViewer(((PetscObject) svm)->comm,NULL,((PetscObject) svm)->prefix,"-svm_view",&v,&format,&view) );
  if (view) {
    TRY( PetscViewerPushFormat(v,format) );
    TRY( SVMView(svm,v) );
    TRY( PetscViewerPopFormat(v) );
    TRY( PetscViewerDestroy(&v) );
  }
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
#define __FUNCT__ "SVMReconstructHyperplane"
/*@
  SVMReconstructHyperplane - Performs reconstruction from dual to primal for normal vector and bias.

  Collective on SVM

  Input Parameter:
. svm - SVM context

.seealso SVMTrain(), SVMPredict(), SVMTest()
@*/
PetscErrorCode SVMReconstructHyperplane(SVM svm)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  if (svm->ops->reconstructhyperplane) {
    TRY( svm->ops->reconstructhyperplane(svm) );
  }
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
  PetscViewer       v;
  PetscViewerFormat format;
  PetscBool         view_score;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->test(svm) );

  TRY( PetscOptionsGetViewer(((PetscObject)svm)->comm,NULL,((PetscObject)svm)->prefix,"-svm_view_score",&v,&format,&view_score) );
  if (view_score) {
    TRY( PetscViewerPushFormat(v,format) );
    TRY( SVMViewScore(svm,v) );
    TRY( PetscViewerPopFormat(v) );
    TRY( PetscViewerDestroy(&v) );
  }
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

.seealso ModelScore, SVMComputeModelScores()
@*/
PetscErrorCode SVMGetModelScore(SVM svm,ModelScore score_type,PetscReal *s)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  TRY( PetscUseMethod(svm,"SVMGetModelScore_C",(SVM,ModelScore,PetscReal *),(svm,score_type,s)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetHyperOpt"
/*@
  SVMSetHyperOpt - Set flag specifying whether optimization of hyperparameter will be performed.
  It is set to PETSC_FALSE by default.

  Collective on SVM

  InputParameters:
+ svm - SVM context
- flg - flg

.seealso SVMGridSearch(), SVMCrossValidation(), SVM
@*/
PetscErrorCode SVMSetHyperOpt(SVM svm,PetscBool flg)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveBool(svm,2,flg);
  svm->hyperoptset = flg;
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
  SVMSetHyperOptScoreTypes - Sets score types for evaluating performance of model during hyperparameter optimization.

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- type - type of model score

.seealso SVMGetHyperOptScoreTypes(), ModelScore
@*/
PetscErrorCode SVMSetHyperOptScoreTypes(SVM svm,PetscInt n,ModelScore types[])
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  for (i = 0; i < n; ++i) PetscValidLogicalCollectiveEnum(svm,types[i],2);
  TRY( PetscMemcpy(svm->hopt_score_types,types,n * sizeof(ModelScore)) );
  svm->hopt_nscore_types = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetHyperOptNScoreTypes"
/*@
  SVMGetHyperOptNScoreTypes - Returns count of score types specified for hyperparameter optimization.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. n - count of score types

seealso SVMSetHyperOptScoreTypes, ModelScore
@*/
PetscErrorCode SVMGetHyperOptNScoreTypes(SVM svm,PetscInt *n)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidPointer(n,2);
  *n = svm->hopt_nscore_types;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetHyperOptScoreTypes"
/*@
  SVMGetHyperOptScoreTypes - Returns array of score types specified for evaluating performance of model during hyperparameter optimization.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. types - types of model score

.seealso SVMSetHyperOptScoreTypes(), ModelScore
@*/
PetscErrorCode SVMGetHyperOptScoreTypes(SVM svm,const ModelScore *types[])
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidPointer(types,2);
  *types = svm->hopt_score_types;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetCrossValidationType"
/*@
  SVMSetCrossValidationType - Sets type of cross validation.

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- type - type of cross validation

.seealso SVMGetCrossValidationType(), CrossValidationType
@*/
PetscErrorCode SVMSetCrossValidationType(SVM svm,CrossValidationType type)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(svm,type,2);
  svm->cv_type = type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetCrossValidationType"
/*@
  SVMGetCrossValidationType - Returns type of cross validation.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. type - type of cross validation

.seealso SVMSetCrossValidationType(), CrossValidationType
@*/
PetscErrorCode SVMGetCrossValidationType(SVM svm,CrossValidationType *type)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidPointer(type,2);
  *type = svm->cv_type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMCrossValidation"
/*@
  SVMKFoldCrossValidation - Performs cross validation.

  Collective on SVM

  Input Parameters:
+ svm - SVM context
. c_arr - manually specified set of penalty C values
- m - size of c_arr

  Output Parameter:
. score - array of scores for each penalty C

  Level: beginner

.seealso: SVMKFoldCrossValidation(), SVMStratifiedKFoldCrossValidation(), SVMGridSearch(), SVMSetNfolds(), SVM
@*/
PetscErrorCode SVMCrossValidation(SVM svm,PetscReal c_arr[],PetscInt m,PetscReal score[])
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->crossvalidation(svm,c_arr,m,score) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMKFoldCrossValidation"
/*@
  SVMKFoldCrossValidation - Performs k-folds cross validation.

  Collective on SVM

  Input Parameters:
+ svm - SVM context
. c_arr - manually specified set of penalty C values
- m - size of c_arr

  Output Parameter:
. score - array of scores for each penalty C

  Level: beginner

.seealso: SVMStratifiedKFoldCrossValidation(), SVMGridSearch(), SVMSetNfolds(), SVM
@*/
PetscErrorCode SVMKFoldCrossValidation(SVM svm,PetscReal c_arr[],PetscInt m,PetscReal score[])
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( PetscTryMethod(svm,"SVMKFoldCrossValidation_C",(SVM,PetscReal [],PetscInt, PetscReal []),(svm,c_arr,m,score)) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMStratifiedKFoldCrossValidation"
/*@

@*/
PetscErrorCode SVMStratifiedKFoldCrossValidation(SVM svm,PetscReal c_arr[],PetscInt m,PetscReal score[])
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( PetscTryMethod(svm,"SVMStratifiedKFoldCrossValidation_C",(SVM,PetscReal [],PetscInt, PetscReal []),(svm,c_arr,m,score)) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMComputeModelScores"
/*@
  SVMComputeModelScores - Evaluates performance scores of model.

  Collective on SVM

  Input Parameters:
+ svm - SVM context
. y_pred - predicted labels of tested samples
- y_known - tested samples

  Level: intermediate

.seealso SVMTrain(), SVMTest(), SVMGetModelScores()
@*/
PetscErrorCode SVMComputeModelScores(SVM svm,Vec y_pred,Vec y_known)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  if (svm->ops->computemodelscores) {
    TRY( svm->ops->computemodelscores(svm,y_pred,y_known) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMComputeHingeLoss"
/*@
  SVMComputeHingeLoss - Computes hinge loss function.

  Collective on SVM

  Input Parameter:
. svm - SVM context

  Level: advanced

.seealso: SVMLossType, SVM_L1, SVM_L2
@*/
PetscErrorCode SVMComputeHingeLoss(SVM svm)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  if (svm->ops->computehingeloss) {
    TRY( svm->ops->computehingeloss(svm) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMComputeModelParams"
/*@
 SVMComputeModelParams - Computes parameters of model

  Collective on SVM

  Input Parameter:
. svm - SVM context

  Level: intermediate

.seealso SVMTrain(), SVMTest()
@*/
PetscErrorCode SVMComputeModelParams(SVM svm)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  if (svm->ops->computemodelparams) {
    TRY( svm->ops->computemodelparams(svm) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMLoadDataset"
/*@
  SVMLoadDataset - Loads dataset.

  Input Parameters:
+ svm - SVM context
- v - viewer

  Output Parameters:
+ Xt - matrix of samples
- y - known labels of samples

.seealso SVM
@*/
PetscErrorCode SVMLoadDataset(SVM svm,PetscViewer v,Mat Xt,Vec y)
{
  MPI_Comm   comm;
  const char *type_name = NULL;

  PetscBool  isascii,ishdf5,isbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscValidHeaderSpecific(Xt,MAT_CLASSID,3);
  PetscCheckSameComm(svm,1,Xt,3);
  PetscValidHeaderSpecific(y,VEC_CLASSID,4);
  PetscCheckSameComm(svm,1,y,4);

  TRY( PetscObjectTypeCompare((PetscObject) v,PETSCVIEWERASCII,&isascii) );
  TRY( PetscObjectTypeCompare((PetscObject) v,PETSCVIEWERHDF5,&ishdf5) );
  TRY( PetscObjectTypeCompare((PetscObject) v,PETSCVIEWERBINARY,&isbinary) );

  PetscLogEventBegin(SVM_LoadDataset,svm,0,0,0);
  if (isascii) {
    TRY( DatasetLoad_SVMLight(Xt,y,v) );
  } else if (ishdf5 || isbinary) {
    TRY( DatasetLoad_Binary(Xt,y,v) );
  } else {
    TRY( PetscObjectGetComm((PetscObject) v,&comm) );
    TRY( PetscObjectGetType((PetscObject) v,&type_name) );

    FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMLoadDataset",type_name);
  }
  PetscLogEventEnd(SVM_LoadDataset,svm,0,0,0);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMLoadTrainingDataset"
/*@
  SVMLoadTrainingDataset - Loads training dataset.

  Input Parameters:
+ svm - SVM context
- v - viewer

  Level: intermediate

.seealso SVMLoadTestDataset(), SVM
@*/
PetscErrorCode SVMLoadTrainingDataset(SVM svm,PetscViewer v)
{
  MPI_Comm   comm;

  const char *dataset_file;
  PetscBool  view_io,view_dataset;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);

  if (svm->ops->loadtrainingdataset) {
    TRY( svm->ops->loadtrainingdataset(svm,v) );
  }

  TRY( PetscViewerFileGetName(v,&dataset_file) );
  TRY( PetscStrcpy(svm->training_dataset_file,dataset_file) );

  TRY( PetscOptionsHasName(NULL,NULL,"-svm_view_io",&view_io) );
  TRY( PetscOptionsHasName(NULL,NULL,"-svm_view_training_dataset",&view_dataset) );
  if (view_io || view_dataset) {
    TRY( PetscObjectGetComm((PetscObject) v,&comm) );
    TRY( SVMViewTrainingDataset(svm,PETSC_VIEWER_STDOUT_(comm)) );
  }
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMViewTrainingDataset"
PetscErrorCode SVMViewTrainingDataset(SVM svm,PetscViewer v)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);

  if (svm->ops->viewtrainingdataset) {
    TRY(svm->ops->viewtrainingdataset(svm,v) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMLoadTestDataset"
/*@
  SVMLoadTestDataset - Loads test dataset.

  Input Parameters:
+ svm - SVM context
- v - viewer

  Level: intermediate

.seealso SVMLoadTrainingDataset(), SVM
@*/
PetscErrorCode SVMLoadTestDataset(SVM svm,PetscViewer v)
{
  MPI_Comm   comm;

  Mat        Xt_test;
  Mat        Xt_biased;
  Vec        y_test;

  PetscReal  bias;
  PetscInt   mod;

  const char *dataset_file;
  PetscBool  view_io,view_test_dataset;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);

  TRY( PetscObjectGetComm((PetscObject) svm,&comm) );
  TRY( MatCreate(comm,&Xt_test) );
  TRY( PetscObjectSetName((PetscObject) Xt_test,"Xt_test") );
  TRY( VecCreate(comm,&y_test) );
  TRY( PetscObjectSetName((PetscObject) y_test,"y_test") );

  TRY( SVMLoadDataset(svm,v,Xt_test,y_test) );
  TRY( SVMGetMod(svm,&mod) );
  if (mod == 2) {
    TRY( SVMGetBias(svm,&bias) );
    TRY( MatBiasedCreate(Xt_test,bias,&Xt_biased) );
    Xt_test = Xt_biased;
  }
  TRY( SVMSetTestDataset(svm,Xt_test,y_test) );

  TRY( PetscViewerFileGetName(v,&dataset_file) );
  TRY( PetscStrcpy(svm->test_dataset_file,dataset_file) );

  TRY( PetscOptionsHasName(NULL,NULL,"-svm_view_io",&view_io) );
  TRY( PetscOptionsHasName(NULL,NULL,"-svm_view_test_dataset",&view_test_dataset) );
  if (view_io || view_test_dataset) {
    TRY( PetscObjectGetComm((PetscObject) v,&comm) );
    TRY( SVMViewTestDataset(svm,PETSC_VIEWER_STDOUT_(comm)) );
  }

  /* Free memory */
  TRY( MatDestroy(&Xt_test) );
  TRY( VecDestroy(&y_test) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMViewTestDataset"
PetscErrorCode SVMViewTestDataset(SVM svm,PetscViewer v)
{
  MPI_Comm   comm;
  const char *type_name = NULL;

  Mat        Xt_test;
  Vec        y_test;

  PetscBool  isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);

  TRY( SVMGetTestDataset(svm,&Xt_test,&y_test) );
  if (!Xt_test || !y_test) {
    TRY( PetscObjectGetComm((PetscObject) v,&comm) );
    FLLOP_SETERRQ(comm,PETSC_ERR_ARG_NULL,"Test dataset is not set");
  }

  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&isascii) );
  if (isascii) {
    TRY( PetscViewerASCIIPrintf(v, "=====================\n") );
    TRY( PetscObjectPrintClassNamePrefixType((PetscObject) svm,v) );

    TRY( PetscViewerASCIIPushTab(v) );
    TRY( PetscViewerASCIIPrintf(v,"Test dataset from file \"%s\" was loaded successfully!\n",svm->test_dataset_file) );
    TRY( PetscViewerASCIIPrintf(v,"Dataset contains:\n") );

    TRY( PetscViewerASCIIPushTab(v) );
    TRY( SVMDatasetInfo(svm,Xt_test,y_test,v) );
    TRY( PetscViewerASCIIPopTab(v) );

    TRY(PetscViewerASCIIPopTab(v));
    TRY(PetscViewerASCIIPrintf(v,"=====================\n"));
  } else {
    TRY( PetscObjectGetComm((PetscObject) v,&comm) );
    TRY( PetscObjectGetType((PetscObject) v,&type_name) );

    FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewTestDataset",type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMDatasetInfo"
PetscErrorCode SVMDatasetInfo(SVM svm,Mat Xt,Vec y,PetscViewer v)
{
  MPI_Comm   comm;
  const char *type_name = NULL;

  PetscInt   M,M_plus,M_minus,N;
  PetscReal  per_plus,per_minus;

  PetscInt   svm_mod;

  Vec        y_max;
  PetscReal  max;

  IS         is_plus;

  PetscBool  isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(Xt,MAT_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,4);

  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&isascii) );
  if (!isascii) {
    TRY( PetscObjectGetComm((PetscObject) v,&comm) );
    TRY( PetscObjectGetType((PetscObject) v,&type_name) );

    FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMDatasetInfo",type_name);
  }

  TRY( MatGetSize(Xt,&M,&N) );

  TRY( VecDuplicate(y,&y_max) );
  TRY( VecMax(y,NULL,&max) );
  TRY( VecSet(y_max,max) );

  TRY( VecWhichEqual(y,y_max,&is_plus) );
  TRY( ISGetSize(is_plus,&M_plus) );

  per_plus  = ((PetscReal) M_plus / (PetscReal) M) * 100.;
  M_minus   = M - M_plus;
  per_minus = 100 - per_plus;

  TRY( SVMGetMod(svm,&svm_mod) );
  if (svm_mod == 2) N -= 1;

  TRY( PetscViewerASCIIPushTab(v) );
  TRY( PetscViewerASCIIPrintf(v,"samples\t%5D\n",M) );
  TRY( PetscViewerASCIIPrintf(v,"samples+\t%5D (%.2f%%)\n",M_plus,per_plus) );
  TRY( PetscViewerASCIIPrintf(v,"samples-\t%5D (%.2f%%)\n",M_minus,per_minus) );
  TRY( PetscViewerASCIIPrintf(v,"features\t%5D\n",N) );
  TRY( PetscViewerASCIIPopTab(v) );

  TRY( VecDestroy(&y_max) );
  TRY( ISDestroy(&is_plus) );
  PetscFunctionReturn(0);
}
