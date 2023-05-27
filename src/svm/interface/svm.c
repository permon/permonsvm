
#include <permon/private/svmimpl.h>
#include "../utils/io.h"

PetscClassId  SVM_CLASSID;
PetscLogEvent SVM_LoadDataset,SVM_LoadGramian;

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
  PetscCall(SVMInitializePackage());
#endif
  PetscCall(PetscHeaderCreate(svm,SVM_CLASSID,"SVM","SVM Classifier","SVM",comm,SVMDestroy,SVMView));

  svm->C      = 1.;
  svm->C_old  = 1.;
  svm->Cp     = 2.;
  svm->Cp_old = 2.;
  svm->Cn     = 1.;
  svm->Cn_old = 1.;

  svm->logC_base   =  2.;
  svm->logC_start  = -2.;
  svm->logC_step   =  1.;
  svm->logC_end    =  2.;
  svm->logCp_base  =  2.;
  svm->logCp_start = -2.;
  svm->logCp_end   =  2.;
  svm->logCp_step  =  1.;
  svm->logCn_base  =  2.;
  svm->logCn_start = -2.;
  svm->logCn_end   =  2.;
  svm->logCn_step  =  1.;

  svm->loss_type  = SVM_L1;
  svm->user_bias  = 1.;
  svm->svm_mod    = 2;

  PetscCall(PetscMemzero(svm->hopt_score_types,7 * sizeof(ModelScore)));

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
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscCall(MatDestroy(&svm->Xt_test));
  svm->Xt_test = NULL;
  PetscCall(VecDestroy(&svm->y_test));
  svm->y_test  = NULL;

  PetscCall((*svm->ops->reset)(svm));

  svm->setupcalled          = PETSC_FALSE;
  svm->posttraincalled      = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(PetscFree(svm->data));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  QPS  qps;
  void *cctx;

  PetscFunctionBegin;
  if (!*svm) PetscFunctionReturn(PETSC_SUCCESS);

  PetscValidHeaderSpecific(*svm,SVM_CLASSID,1);
  --((PetscObject)(*svm))->refct;
  if (((PetscObject)(*svm))->refct == 1) {
    PetscCall(SVMGetQPS(*svm,&qps));
    PetscCall(QPSGetConvergenceContext(qps,&cctx));
    if (*svm != cctx) {
      *svm = 0;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  } else if (((PetscObject)(*svm))->refct > 1) {
    *svm = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (((PetscObject)(*svm))->refct < 0) PetscFunctionReturn(PETSC_SUCCESS);
  ((PetscObject)(*svm))->refct = 0;

  PetscCall(SVMReset(*svm));
  if ((*svm)->ops->destroy) {
    PetscCall((*(*svm)->ops->destroy)(*svm));
  }

  PetscCall(PetscHeaderDestroy(svm));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscInt            svm_mod;
  SVMLossType         loss_type;
  PetscInt            penalty_type;
  PetscReal           bias;
  PetscReal           C;

  PetscBool           hyperoptset;
  ModelScore          hyperopt_score_types[7];
  PetscReal           logC_base,logC_stride[3];

  CrossValidationType cv_type;
  PetscInt            nfolds;

  PetscBool           warm_start;

  PetscInt            n;
  PetscBool           flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  PetscObjectOptionsBegin((PetscObject)svm);
  // TODO move to SetFromOptions_Binary
  PetscCall(PetscOptionsInt("-svm_binary_mod","","SVMSetMod",svm->svm_mod,&svm_mod,&flg));
  if (flg) {
    PetscCall(SVMSetMod(svm,svm_mod));
  }

  PetscCall(PetscOptionsEnum("-svm_loss_type","Specify the loss function for soft-margin SVM (non-separable samples).","SVMSetLossType",SVMLossTypes,(PetscEnum)svm->loss_type,(PetscEnum*)&loss_type,&flg));
  if (flg) {
    PetscCall(SVMSetLossType(svm,loss_type));
  }

  PetscCall(PetscOptionsInt("-svm_penalty_type","Set type of misclasification error penalty.","SVMSetPenaltyType",svm->penalty_type,&penalty_type,&flg));
  if (flg) {
    PetscCall(SVMSetPenaltyType(svm,penalty_type));
  }

  PetscCall(PetscOptionsReal("-svm_user_bias","Set a user defined bias for a relaxed bias formulation.","SVMSetUserBias",svm->user_bias,&bias,&flg));
  if (flg) {
    PetscCall(SVMSetUserBias(svm,bias));
  }

  PetscCall(PetscOptionsReal("-svm_C","Set SVM C (C).","SVMSetC",svm->C,&C,&flg));
  if (flg) {
    PetscCall(SVMSetC(svm,C));
  }

  PetscCall(PetscOptionsReal("-svm_Cp","Set SVM Cp (Cp).","SVMSetCp",svm->Cp,&C,&flg));
  if (flg) {
    PetscCall(SVMSetCp(svm,C));
  }

  PetscCall(PetscOptionsReal("-svm_Cn","Set SVM Cn (Cn).","SVMSetCn",svm->Cn,&C,&flg));
  if (flg) {
    PetscCall(SVMSetCn(svm,C));
  }

  /* Hyperparameter optimization options */
  PetscCall(PetscOptionsBool("-svm_hyperopt","Specify whether hyperparameter optimization will be performed.","SVMSetHyperOpt",svm->hyperoptset,&hyperoptset,&flg));
  if (flg) {
    PetscCall(SVMSetHyperOpt(svm,hyperoptset));
  }

  n = 7;
  PetscCall(PetscOptionsEnumArray("-svm_hyperopt_score_types","Specify the score types for evaluating performance of model during hyperparameter optimization.","SVMSetHyperOptScoreTypes",ModelScores,(PetscEnum *) hyperopt_score_types,&n,&flg));
  if (flg) {
    PetscCall(SVMSetHyperOptScoreTypes(svm,n,hyperopt_score_types));
  }

  /* Grid search options (penalty type 1) */
  PetscCall(PetscOptionsReal("-svm_gs_logC_base","Base of log of C values that specify grid (penalty type 1).","SVMGridSearchSetBaseLogC",svm->logC_base,&logC_base,&flg));
  if (flg) {
    PetscCall(SVMGridSearchSetBaseLogC(svm,logC_base));
  }

  n = 3;
  logC_stride[1] =  2.;
  logC_stride[2] =  1.;
  PetscCall(PetscOptionsRealArray("-svm_gs_logC_stride","Stride log C values that specify grid (penalty type 1)","SVMGridSearchSetStrideLogC",logC_stride,&n,&flg));
  if (flg) {
    PetscCall(SVMGridSearchSetStrideLogC(svm,logC_stride[0],logC_stride[1],logC_stride[2]));
  }

  /* Grid search options (penalty type 2) */
  PetscCall(PetscOptionsReal("-svm_gs_logCp_base","Base of log of C+ values that specify grid (penalty type 2).","SVMGridSearchSetPositiveBaseLogC",svm->logCp_base,&logC_base,&flg));
  if (flg) {
    PetscCall(SVMGridSearchSetPositiveBaseLogC(svm,logC_base));
  }

  n = 3;
  logC_stride[1] = 2.;
  logC_stride[2] = 1.;
  PetscCall(PetscOptionsRealArray("-svm_gs_logCp_stride","Stride log C+ values that specify grid (penalty type 2).","SVMGridSearchSetPositiveStrideLogC",logC_stride,&n,&flg));
  if (flg) {
    PetscCall(SVMGridSearchSetPositiveStrideLogC(svm,logC_stride[0],logC_stride[1],logC_stride[2]));
  }

  PetscCall(PetscOptionsReal("-svm_gs_logCn_base","Base of log of C- values that specify grid (penalty type 2).","SVMGridSearchSetNegativeBaseLogC",svm->logCn_base,&logC_base,&flg));
  if (flg) {
    PetscCall(SVMGridSearchSetNegativeBaseLogC(svm,logC_base));
  }

  n = 3;
  logC_stride[1] = 2.;
  logC_stride[2] = 1.;
  PetscCall(PetscOptionsRealArray("-svm_gs_logCn_stride","Stride log C- values that specify grid (penalty type 2).","SVMGridSearchSetNegativeStrideLogC",logC_stride,&n,&flg));
  if (flg) {
    PetscCall(SVMGridSearchSetNegativeStrideLogC(svm,logC_stride[0],logC_stride[1],logC_stride[2]));
  }

  /* Cross validation options */
  PetscCall(PetscOptionsEnum("-svm_cv_type","Specify the type of cross validation.","SVMSetCrossValidationType",CrossValidationTypes,(PetscEnum)svm->cv_type,(PetscEnum*)&cv_type,&flg));
  if (flg) {
    PetscCall(SVMSetCrossValidationType(svm,cv_type));
  }

  PetscCall(PetscOptionsInt("-svm_nfolds","Set number of folds (nfolds).","SVMSetNfolds",svm->nfolds,&nfolds,&flg));
  if (flg) {
    PetscCall(SVMSetNfolds(svm,nfolds));
  }

  /* Warm start */
  PetscCall(PetscOptionsBool("-svm_warm_start","Specify whether warm start is used in cross-validation.","SVMSetWarmStart",svm->warm_start,&warm_start,&flg));
  if (flg) {
    PetscCall(SVMSetWarmStart(svm,warm_start));
  }

  if (svm->ops->setfromoptions) {
    PetscCall(svm->ops->setfromoptions(PetscOptionsObject,svm));
  }
  PetscOptionsEnd();

  svm->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscCall(PetscObjectTypeCompare((PetscObject) svm,type,&issame));
  if (issame) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(SVMList,type,(void(**)(void))&create_svm));
  if (!create_svm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested SVM type %s",type);

  /* Destroy the pre-existing private SVM context */
  if (svm->ops->destroy) PetscCall(svm->ops->destroy(svm));
  /* Reinitialize function pointers in SVMOps structure */
  PetscCall(PetscMemzero(svm->ops,sizeof(struct _SVMOps)));

  PetscCall((*create_svm)(svm));
  PetscCall(PetscObjectChangeTypeName((PetscObject)svm,type));
  PetscFunctionReturn(PETSC_SUCCESS);
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

.seealso SVMGetQPS(), SVMGetQP()
@*/
PetscErrorCode SVMSetQPS(SVM svm,QPS qps)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(qps,QPS_CLASSID,2);

  PetscTryMethod(svm,"SVMSetQPS_C",(SVM,QPS),(svm,qps));
  PetscFunctionReturn(PETSC_SUCCESS);
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

.seealso SVMSetQPS(), SVMGetQP()
@*/
PetscErrorCode SVMGetQPS(SVM svm,QPS *qps)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidPointer(qps,2);

  PetscUseMethod(svm,"SVMGetQPS_C",(SVM,QPS *),(svm,qps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetQP"
/*@
  SVMGetQP - Returns QP context.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. qp - QP context

  Level: beginner

.seealso SVMSetQPS(), SVMGetQPS()
@*/
PetscErrorCode SVMGetQP(SVM svm,QP *qp)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidPointer(qp,2);

  PetscUseMethod(svm,"SVMGetQP_C",(SVM,QP *),(svm,qp));
  PetscFunctionReturn(PETSC_SUCCESS);
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

  if (nfolds < 2) SETERRQ(((PetscObject) svm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Argument must be greater than 1.");
  svm->nfolds = nfolds;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  if (svm->penalty_type == type) PetscFunctionReturn(PETSC_SUCCESS);

  if (type != 1 && type != 2) SETERRQ(((PetscObject) svm)->comm,PETSC_ERR_SUP,"Type of penalty (%" PetscInt_FMT ") is not supported. It must be 1 or 2",type);

  svm->penalty_type = type;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (C <= 0) {
    SETERRQ(((PetscObject) svm)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");
  }

  svm->C_old = svm->C;
  svm->C     = C;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (Cp <= 0) {
    SETERRQ(((PetscObject) svm)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");
  }

  svm->Cp_old = svm->Cp;
  svm->Cp     = Cp;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (Cn <= 0) {
    SETERRQ(((PetscObject) svm)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");
  }

  svm->Cn_old = svm->Cn;
  svm->Cn     = Cn;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  if (m > 2) SETERRQ(((PetscObject) svm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Argument must be 1 or 2");
  PetscValidLogicalCollectiveInt(svm,m,2);
  PetscValidLogicalCollectiveReal(svm,p[0],3);
  if (m == 2) PetscValidLogicalCollectiveReal(svm,p[1],3);

  PetscCall(SVMGetPenaltyType(svm,&penalty_type));

  if (penalty_type == 1) {
    PetscCall(SVMSetC(svm,p[0]));
  } else {
    if (m == 1) {
      PetscCall(SVMSetCp(svm,p[0]));
      PetscCall(SVMSetCn(svm,p[0]));
    } else {
      PetscCall(SVMSetCp(svm,p[0]));
      PetscCall(SVMSetCn(svm,p[1]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchSetBaseLogC"
/*@
  SVMGridSearchSetBaseLogC - Sets the base of log of C values that specify grid (penalty type 1).

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- logC_base - base of log of C

  Level: beginner

.seealso SVMGridSearch(), SVMGridSearchGetBaseLogC(), SVMGridSearchSetStrideLogC(), SVMSetPenaltyType()
@*/
PetscErrorCode SVMGridSearchSetBaseLogC(SVM svm,PetscReal logC_base)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logC_base,2);

  if (logC_base <= 0) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");

  svm->logC_base = logC_base;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchGetBaseLogC"
/*@
  SVMGridSearchGetBaseLogC - Returns the base of log of C values that specify grid (penalty type 1).

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. logC_base - base of log of C

  Level: beginner

.seealso SVMGridSearch(), SVMGridSearchSetBaseLogC(), SVMGridSearchSetStrideLogC(), SVMSetPenaltyType()
@*/
PetscErrorCode SVMGridSearchGetBaseLogC(SVM svm,PetscReal *logC_base)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(logC_base,2);

  *logC_base = svm->logC_base;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchSetStrideLogC"
/*@
  SVMGridSearchSetStrideLogC - Sets stride of log C values that specify grid used in hyperparameter optimization based on grid-searching (penalty type 1).

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
. logC_start - first value of log C stride
. logC_end - last value of log C stride
- logC_step - step of log C stride

  Level: beginner

.seealso SVMGridSearch(), SVMGridSearchSetBaseLogC(), SVMGridSearchGetStrideLogC(), SVMSetPenaltyType()
@*/
PetscErrorCode SVMGridSearchSetStrideLogC(SVM svm,PetscReal logC_start,PetscReal logC_end,PetscReal logC_step)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logC_start,2);
  PetscValidLogicalCollectiveReal(svm,logC_end,3);
  PetscValidLogicalCollectiveReal(svm,logC_step,4);

  /* Validating values of input parameters */
  if (logC_step == 0) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Step (logC_step) cannot be 0");
  if (logC_start == logC_end) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Start (logC_start) and end (logC_end) cannot be same");
  if (logC_start > logC_end && logC_step > 0) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Step (logC_step) cannot be greater than 0 if start (logC_start) is greater than end (logC_end)");
  if (logC_start < logC_end && logC_step < 0) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Step (logC_step) cannot be less than 0 if start (logC_start) is less than end (logC_end)");

  svm->logC_start = logC_start;
  svm->logC_end  = logC_end;
  svm->logC_step  = logC_step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchGetStrideLogC"
/*@
  SVMGridSearchGetStrideLogC - Returns stride of log C values used for specifying grid in hyperparameter optimization based on grid-searching (penalty type 1).

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameters:
+ logC_start - first value of log C stride
. logC_end - last value of log C stride
- logC_step - step of log C stride

  Level: beginner

.seealso SVMGridSearch(), SVMGridSearchSetBaseLogC(), SVMGridSearchSetStrideLogC(), SVMSetPenaltyType()
@*/
PetscErrorCode SVMGridSearchGetStrideLogC(SVM svm,PetscReal *logC_start,PetscReal *logC_end,PetscReal *logC_step)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(logC_start,2);
  PetscValidRealPointer(logC_end,3);
  PetscValidRealPointer(logC_step,4);

  if (logC_start) {
    *logC_start = svm->logC_start;
  }
  if (logC_end) {
    *logC_end = svm->logC_end;
  }
  if (logC_step) {
    *logC_step = svm->logC_step;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchSetPositiveBaseLogC"
/*@
  SVMGridSearchSetPositiveBaseLogC - Returns the base of log of C+ values that specify grid (penalty type 2).

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- logCp_base - base of log of C+

  Level: beginner

.seealso SVMGridSearch(), SVMGridSearchGetPositiveBaseLogC(), SVMGridSearchSetPositiveStrideLogC(), SVMGridSearchSetNegativeBaseLogC(), SVMGridSearchSetNegativeStrideLogC(), SVMSetPenaltyType()
@*/
PetscErrorCode SVMGridSearchSetPositiveBaseLogC(SVM svm,PetscReal logCp_base)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logCp_base,2);

  if (logCp_base <= 0) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");

  svm->logCp_base = logCp_base;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchGetPositiveBaseLogC"
/*@
  SVMGridSearchGetPosBaseLogC - Returns the base of log of C+ values that specify grid (penalty type 2).

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. logCp_base - base of log of C+

  Level: beginner

.seealso SVMGridSearch(), SVMGridSearchSetPositiveBaseLogC(), SVMGridSearchSetPositiveStrideLogC(), SVMGridSearchSetNegativeBaseLogC(), SVMGridSearchSetNegativeStrideLogC(), SVMSetPenaltyType()
@*/
PetscErrorCode SVMGridSearchGetPositiveBaseLogC(SVM svm,PetscReal *logCp_base)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(logCp_base,2);

  *logCp_base = svm->logCp_base;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchSetPositiveStrideLogC"
/*@
  SVMGridSearchSetPositiveStrideLogC - Sets stride of log C+ values that specify grid used in hyperparameter optimization based on grid-searching (penalty type 2).

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
. logC_start - first value of log C+ stride
. logC_end - last value of log C+ stride
- logC_step - step of log C+ stride

  Level: beginner

.seealso SVMGridSearch(), SVMGridSearchSetPositiveBaseLogC(), SVMGridSearchGetPositiveStrideLogC(), SVMGridSearchSetNegativeBaseLogC(), SVMGridSearchSetNegativeStrideLogC(), SVMSetPenaltyType()
@*/
PetscErrorCode SVMGridSearchSetPositiveStrideLogC(SVM svm,PetscReal logC_start,PetscReal logC_end,PetscReal logC_step)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logC_start,2);
  PetscValidLogicalCollectiveReal(svm,logC_end,3);
  PetscValidLogicalCollectiveReal(svm,logC_step,4);

  /* Validating values of input parameters */
  if (logC_step == 0) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Step (logC_step) cannot be 0");
  if (logC_start == logC_end) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Start (logC_start) and end (logC_end) cannot be same");
  if (logC_start > logC_end && logC_step > 0) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Step (logC_step) cannot be greater than 0 if start (logC_start) is greater than end (logC_end)");
  if (logC_start < logC_end && logC_step < 0) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Step (logC_step) cannot be less than 0 if start (logC_start) is less than end (logC_end)");

  svm->logCp_start = logC_start;
  svm->logCp_end   = logC_end;
  svm->logCp_step  = logC_step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchGetPositiveStrideLogC"
/*@
  SVMGridSearchGetStrideLogC - Returns stride of log C+ values used for specifying grid in hyperparameter optimization based on grid-searching (penalty type 2).

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameters:
+ logC_start - first value of log C+ stride
. logC_end - last value of log C+ stride
- logC_step - step of log C+ stride

  Level: beginner

.seealso SVMGridSearch(), SVMGridSearchSetPositiveBaseLogC(), SVMGridSearchSetPositiveStrideLogC(), SVMGridSearchSetNegativeBaseLogC(), SVMGridSearchSetNegativeStrideLogC(), SVMSetPenaltyType()
@*/
PetscErrorCode SVMGridSearchGetPositiveStrideLogC(SVM svm,PetscReal *logC_start,PetscReal *logC_end,PetscReal *logC_step)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(logC_start,2);
  PetscValidRealPointer(logC_end,3);
  PetscValidRealPointer(logC_step,4);

  if (logC_start) {
    *logC_start = svm->logCp_start;
  }
  if (logC_end) {
    *logC_end = svm->logCp_end;
  }
  if (logC_step) {
    *logC_step = svm->logCp_step;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchSetNegativeBaseLogC"
/*@
  SVMGridSearchSetNegativeBaseLogC - Sets the base of log of C- values that specify grid (penalty type 2).

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
- logCn_base - base of log of C-

  Level: beginner

.seealso SVMGridSearch(), SVMGridSearchGetNegativeBaseLogC(), SVMGridSearchSetNegativeStrideLogC(), SVMGridSearchSetPositiveBaseLogC(), SVMGridSearchSetPositiveStrideLogC(), SVMSetPenaltyType()
@*/
PetscErrorCode SVMGridSearchSetNegativeBaseLogC(SVM svm,PetscReal logCn_base)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logCn_base,2);

  if (logCn_base <= 0) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");

  svm->logCn_base = logCn_base;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchGetNegativeBaseLogC"
/*@
  SVMGridSearchGetNegativeBaseLogC - Returns base of log of C- values that specify grid (penalty type 2).

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. logCn_base - base of log of C-

  Level: beginner

.seealso SVMGridSearch(), SVMGridSearchSetNegativeBaseLogC(), SVMGridSearchSetNegativeStrideLogC(), SVMGridSearchSetPositiveBaseLogC(), SVMGridSearchSetPositiveStrideLogC(), SVMSetPenaltyType()
@*/
PetscErrorCode SVMGridSearchGetNegativeBaseLogC(SVM svm,PetscReal *logCn_base)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(logCn_base,2);

  *logCn_base = svm->logCn_base;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchSetNegativeStrideLogC"
/*@
  SVMGridSearchSetNegativeStrideLogC - Sets stride of log C- values that specify grid used in hyperparameter optimization based on grid-searching (penalty type 2).

  Logically Collective on SVM

  Input Parameters:
+ svm - SVM context
. logC_start - first value of log C- stride
. logC_end - last value of log C- stride
- logC_step - step of log C- stride

  Level: beginner

.seealso SVMGridSearch(), SVMGridSearchSetNegativeBaseLogC(), SVMGridSearchGetNegativeStrideLogC(), SVMGridSearchSetPositiveBaseLogC(), SVMGridSearchSetPositiveStrideLogC(), SVMSetPenaltyType()
@*/
PetscErrorCode SVMGridSearchSetNegativeStrideLogC(SVM svm,PetscReal logC_start,PetscReal logC_end,PetscReal logC_step)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,logC_start,2);
  PetscValidLogicalCollectiveReal(svm,logC_end,3);
  PetscValidLogicalCollectiveReal(svm,logC_step,4);

  /* Validating values of input parameters */
  if (logC_step == 0) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Step (logC_step) cannot be 0");
  if (logC_start == logC_end) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Start (logC_start) and end (logC_end) cannot be same");
  if (logC_start > logC_end && logC_step > 0) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Step (logC_step) cannot be greater than 0 if start (logC_start) is greater than end (logC_end)");
  if (logC_start < logC_end && logC_step < 0) SETERRQ(PetscObjectComm((PetscObject) svm),PETSC_ERR_ARG_OUTOFRANGE,"Step (logC_step) cannot be less than 0 if start (logC_start) is less than end (logC_end)");

  svm->logCn_start = logC_start;
  svm->logCn_end   = logC_end;
  svm->logCn_step  = logC_step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGridSearchGetNegativeStrideLogC"
/*@
  SVMGridSearchGetStrideLogC - Gets stride of log C- values used for specifying grid in hyperparameter optimization based on grid-searching (penalty type 2).

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameters:
+ logC_start - the first value of log C- stride
. logC_end - the last value of log C- stride
- logC_step - step of log C- stride

  Level: beginner

.seealso SVMGridSearch(), SVMGridSearchSetNegativeBaseLogC(), SVMGridSearchSetNegativeStrideLogC(), SVMGridSearchSetPositiveBaseLogC(), SVMGridSearchSetPositiveStrideLogC(), SVMSetPenaltyType()
@*/
PetscErrorCode SVMGridSearchGetNegativeStrideLogC(SVM svm,PetscReal *logC_start,PetscReal *logC_end,PetscReal *logC_step)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(logC_start,2);
  PetscValidRealPointer(logC_end,3);
  PetscValidRealPointer(logC_step,4);

  if (logC_start) {
    *logC_start = svm->logCn_start;
  }
  if (logC_end) {
    *logC_end = svm->logCn_end;
  }
  if (logC_step) {
    *logC_step = svm->logCn_step;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscTryMethod(svm,"SVMSetOptionsPrefix_C",(SVM,const char []),(svm,prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscTryMethod(svm,"SVMAppendOptionsPrefix_C",(SVM,const char []),(svm,prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscUseMethod(svm,"SVMGetOptionsPrefix_C",(SVM,const char *[]),(svm,prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@

@*/
PetscErrorCode SVMGetInnerSVM(SVM svm,SVM *out)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscUseMethod(svm,"SVMGetInnerSVM_C",(SVM,SVM *),(svm,out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@

@*/
PetscErrorCode SVMGetTao(SVM svm,Tao *tao)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscUseMethod(svm,"SVMGetTao_C",(SVM,Tao *),(svm,tao));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  if (svm->setupcalled) PetscFunctionReturnI(PETSC_SUCCESS);

  PetscCall(svm->ops->setup(svm));
  svm->setupcalled = PETSC_TRUE;
  PetscFunctionReturnI(PETSC_SUCCESS);
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
    PetscCall(svm->ops->view(svm,v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SVMViewTrainingPredictions - Views predictions on training samples using trained model.

  Input Parameters:
+ svm - SVM context
- v - visualization context

  Level: beginner

.seealso SVMLoadTrainingDataset, SVMPredict, PetscViewer
@*/
PetscErrorCode SVMViewTrainingPredictions(SVM svm,PetscViewer v)
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  if (v) PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  if (svm->ops->viewtrainingpredictions) {
    PetscCall(svm->ops->viewtrainingpredictions(svm,v));
  }
  PetscFunctionReturnI(PETSC_SUCCESS);
}

/*@
  SVMViewTestPredictions - Views predictions on test samples using trained model.

  Input Parameters:
+ svm - SVM context
- v - visualization context

  Level: beginner

.seealso SVMLoadTestDataset, SVMPredict, PetscViewer
@*/
PetscErrorCode SVMViewTestPredictions(SVM svm,PetscViewer v)
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  if (v) {PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);}
  if (svm->ops->viewtestpredictions) {
    PetscCall(svm->ops->viewtestpredictions(svm,v));
  }
  PetscFunctionReturnI(PETSC_SUCCESS);
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
    PetscCall(svm->ops->viewscore(svm,v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetGramian"
/*@
  SVMSetGramian - Set precomputed Gramian (kernel) matrix.

  Input Parameters:
+ svm - SVM context
- G - precomputed Gramian matrix

  Level: intermediate

.seealso SVM, SVMGetGramian(), SVMLoadGramian()
@*/
PetscErrorCode SVMSetGramian(SVM svm,Mat G)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(G,MAT_CLASSID,2);

  PetscTryMethod(svm,"SVMSetGramian_C",(SVM,Mat),(svm,G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetGramian"
/*@
  SVMGetGramian - Get precomputed Gramian (kernel) matrix.

  Input Parameter:
. svm - SVM context

  Output Parameter:
. G - precomputed Gramian matrix

  Level: intermediate

.seealso SVM, SVMSetGramian(), SVMLoadGramian()
@*/
PetscErrorCode SVMGetGramian(SVM svm,Mat *G)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidPointer(G,2);

  PetscUseMethod(svm,"SVMGetGramian_C",(SVM,Mat *),(svm,G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetOperator"
/*@
  SVMSetOperator - Sets the Hessian matrix associated with underlying QP.

  Collective on SVM

  Input Parameters:
+ svm - SVM context
- A - Hessian matrix

  Level: intermediate

.seealso SVM, QP, SVMGetOperator()
@*/
PetscErrorCode SVMSetOperator(SVM svm,Mat A)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);

  PetscTryMethod(svm,"SVMSetOperator_C",(SVM,Mat),(svm,A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMGetOperator"
/*@
  SVMSetOperator - Gets the Hessian matrix associated with underlying QP.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. A - Hessian matrix

  Level: intermediate

.seealso SVM, QP, SVMSetOperator()
@*/
PetscErrorCode SVMGetOperator(SVM svm,Mat *A)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidPointer(A,2);

  PetscUseMethod(svm,"SVMGetOperator_C",(SVM,Mat *),(svm,A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMComputeOperator"
/*@
  SVMComputeOperator - Computes implicit Hessian matrix associated with underlying QP problem.

  Collective on SVM

  Level: advanced

  Input parameter:
. svm - SVM context

  Output Parameter:
. A - Hessian matrix

.seealso SVM, SVMSetOperator(), SVMGetOperator(), SVMSetGramian(), SVMGetGramian()
@*/
PetscErrorCode SVMComputeOperator(SVM svm,Mat *A)
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  PetscTryMethod(svm,"SVMComputeOperator_C",(SVM,Mat *),(svm,A));
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetTrainingDataset"
/*@
  SVMSetTrainingDataset - Sets the training samples and labels.

  Not Collective

  Input Parameter:
+ svm - SVM context
. Xt_training - samples data
- y_training - known labels of training samples

  Level: beginner

.seealso SVMGetTrainingDataset(), SVMSetTestDataset(), SVMGetTestDataset()
@*/
PetscErrorCode SVMSetTrainingDataset(SVM svm,Mat Xt_training,Vec y_training)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(Xt_training,MAT_CLASSID,2);
  PetscCheckSameComm(svm,1,Xt_training,2);
  PetscValidHeaderSpecific(y_training,VEC_CLASSID,3);
  PetscCheckSameComm(svm,1,y_training,3);

  PetscTryMethod(svm,"SVMSetTrainingDataset_C",(SVM,Mat,Vec),(svm,Xt_training,y_training));
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscUseMethod(svm,"SVMGetTrainingDataset_C",(SVM,Mat *,Vec *),(svm,Xt_training,y_training));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SVMSetCalibrationDataset - Sets the calibration samples and labels.

  Not Collective

  Input Parameter:
+ svm - SVM context
. Xt_calib - calibration samples data
- y_calib - known labels of calibration samples

  Level: beginner

.seealso SVMSetTrainingDataset(), SVMGetTrainingDataset(), SVMSetTestDataset(), SVMGetTestDataset()
@*/
PetscErrorCode SVMSetCalibrationDataset(SVM svm,Mat Xt_calib,Vec y_calib)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  PetscTryMethod(svm,"SVMSetCalibrationDataset_C",(SVM,Mat,Vec),(svm,Xt_calib,y_calib));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SVMGetTrainingDataset - Returns the calibration samples and labels.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
+ Xt_calib- calibration samples
- y_calib - known labels of calibration samples

  Level: beginner

.seealso SVMSetTrainingDataset(), SVMGetTrainingDataset(), SVMSetTestDataset(), SVMGetTestDataset()
@*/
PetscErrorCode SVMGetCalibrationDataset(SVM svm,Mat *Xt_calib,Vec *y_calib)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  PetscUseMethod(svm,"SVMGetCalibrationDataset_C",(SVM,Mat *,Vec *),(svm,Xt_calib,y_calib));
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscCall(MatDestroy(&svm->Xt_test));
  svm->Xt_test = Xt_test;
  PetscCall(PetscObjectReference((PetscObject) Xt_test));

  PetscCall(VecDestroy(&svm->y_test));
  svm->y_test = y_test;
  PetscCall(PetscObjectReference((PetscObject) y_test));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SVMGetLabels - Returns original labels related to each category.

  Not Collective

  Input Parameters:
. svm - SVM context
. labels - labels related to each category

  Level: intermediate

.seealso SVMSetTrainingDataset, SVMGetTrainingDataset
@*/
PetscErrorCode SVMGetLabels(SVM svm,const PetscReal *labels[])
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscUseMethod(svm,"SVMGetLabels_C",(SVM,const PetscReal *[]),(svm,labels));
  PetscFunctionReturn(PETSC_SUCCESS);
}


#undef __FUNCT__
#define __FUNCT__ "SVMSetAutoPostTrain"
/*@
  SVMSetAutoPostTrain - Sets auto post train flag.

  Logically Collective on SVM

  Input Parameters:
. svm - SVM context
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SVMGetAutoPostTrain - Returns auto post train flag.

  Not Collective

  Input Parameter:
. svm - SVM context

  Output Parameter:
. flg - flag

  Level: developer
@*/
PetscErrorCode SVMGetAutoPostTrain(SVM svm,PetscBool *flg)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidPointer(flg,2);
  *flg = svm->autoposttrain;
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
  PetscCall(svm->ops->train(svm));
  PetscFunctionReturnI(PETSC_SUCCESS);
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
  PetscCall(svm->ops->posttrain(svm));

  PetscCall(PetscOptionsGetViewer(((PetscObject) svm)->comm,NULL,((PetscObject) svm)->prefix,"-svm_view",&v,&format,&view));
  if (view) {
    PetscCall(PetscViewerPushFormat(v,format));
    PetscCall(SVMView(svm,v));
    PetscCall(PetscViewerPopFormat(v));
    PetscCall(PetscViewerDestroy(&v));
  }

  PetscCall(PetscOptionsGetViewer(((PetscObject) svm)->comm,NULL,((PetscObject) svm)->prefix,"-svm_view_training_predictions",&v,&format,&view));
  if (view) {
    PetscCall(PetscViewerPushFormat(v,format));
    PetscCall(SVMViewTrainingPredictions(svm,v));
    PetscCall(PetscViewerPopFormat(v));
    PetscCall(PetscViewerDestroy(&v));
  }
  PetscFunctionReturnI(PETSC_SUCCESS);
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

  PetscTryMethod(svm,"SVMSetSeparatingHyperplane_C",(SVM,Vec,PetscReal),(svm,w,b));
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscUseMethod(svm,"SVMGetSeparatingHyperplane_C",(SVM,Vec *,PetscReal *),(svm,w,b));
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscTryMethod(svm,"SVMSetBias_C",(SVM,PetscReal),(svm,bias));
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscUseMethod(svm,"SVMGetBias_C",(SVM,PetscReal *),(svm,bias));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@

@*/
PetscErrorCode SVMSetUserBias(SVM svm,PetscReal bias)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,bias,2);

  if (svm->user_bias != bias) {
    svm->user_bias = bias;
    svm->setupcalled = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@

@*/
PetscErrorCode SVMGetUserBias(SVM svm,PetscReal *bias)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  *bias = svm->user_bias;
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
    PetscCall(svm->ops->reconstructhyperplane(svm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SVMGetDistancesFromHyperplane - Gets distances of samples from a separating hyperplane.

  Collected on SVM

  Input Parameters:
+ svm - SVM context
- Xt - matrix of samples

  Output Parameter:
. dist - distances of samples from hyperplane

  Level: intermediate

.seealso: SVMPredict()
@*/
PetscErrorCode SVMGetDistancesFromHyperplane(SVM svm,Mat Xt,Vec *dist)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  PetscUseMethod(svm,"SVMGetDistancesFromHyperplane_C",(SVM,Mat,Vec *),(svm,Xt,dist));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(svm->ops->predict(svm,Xt_pred,y_pred));
  PetscFunctionReturnI(PETSC_SUCCESS);
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
  PetscBool         view;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscCall(svm->ops->test(svm));

  PetscCall(PetscOptionsGetViewer(((PetscObject)svm)->comm,NULL,((PetscObject)svm)->prefix,"-svm_view_score",&v,&format,&view));
  if (view) {
    PetscCall(PetscViewerPushFormat(v,format));
    PetscCall(SVMViewScore(svm,v));
    PetscCall(PetscViewerPopFormat(v));
    PetscCall(PetscViewerDestroy(&v));
  }

  PetscCall(PetscOptionsGetViewer(((PetscObject)svm)->comm,NULL,((PetscObject)svm)->prefix,"-svm_view_test_predictions",&v,&format,&view));
  if (view) {
    PetscCall(PetscViewerPushFormat(v,format));
    PetscCall(SVMViewTestPredictions(svm,v));
    PetscCall(PetscViewerPopFormat(v));
    PetscCall(PetscViewerDestroy(&v));
  }
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMConvergedSetUp"
/*@

@*/
PetscErrorCode SVMConvergedSetUp(SVM svm)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  // call specific implementation of setting convergence test
  if (svm->ops->convergedsetup) PetscCall(svm->ops->convergedsetup(svm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMDefaultConvergedCreate"
/*@

@*/
PetscErrorCode SVMDefaultConvergedCreate(SVM svm, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  *ctx = svm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMDefaultConvergedDestroy"
/*@

@*/
PetscErrorCode SVMDefaultConvergedDestroy(void *ctx)
{
  PetscFunctionBegin;
  PetscCall(SVMDestroy((SVM*)&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscUseMethod(svm,"SVMGetModelScore_C",(SVM,ModelScore,PetscReal *),(svm,score_type,s));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(svm->ops->gridsearch(svm));
  PetscFunctionReturnI(PETSC_SUCCESS);
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
  PetscCall(PetscMemcpy(svm->hopt_score_types,types,n * sizeof(ModelScore)));
  svm->hopt_nscore_types = n;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(svm->ops->crossvalidation(svm,c_arr,m,score));
  PetscFunctionReturnI(PETSC_SUCCESS);
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
  PetscTryMethod(svm,"SVMKFoldCrossValidation_C",(SVM,PetscReal [],PetscInt, PetscReal []),(svm,c_arr,m,score));
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMStratifiedKFoldCrossValidation"
/*@

@*/
PetscErrorCode SVMStratifiedKFoldCrossValidation(SVM svm,PetscReal c_arr[],PetscInt m,PetscReal score[])
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscTryMethod(svm,"SVMStratifiedKFoldCrossValidation_C",(SVM,PetscReal [],PetscInt, PetscReal []),(svm,c_arr,m,score));
  PetscFunctionReturnI(PETSC_SUCCESS);
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
    PetscCall(svm->ops->computemodelscores(svm,y_pred,y_known));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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
    PetscCall(svm->ops->computehingeloss(svm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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
    PetscCall(svm->ops->computemodelparams(svm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMLoadGramian"
/*@
  SVMLoadGramian - Loads precomputed Gramian (kernel) matrix.

  Collective on PetscViewer

  Input Parameters:
+ svm - SVM context
- v - viewer

 Level: intermediate

.seealso SVM, SVMSetGramian(), SVMGetGramian(), SVMViewGramian(), SVMLoadTrainingDataset()
@*/
PetscErrorCode SVMLoadGramian(SVM svm,PetscViewer v)
{
  MPI_Comm   comm;
  PetscBool  view_io,view_dataset;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);

  PetscLogEventBegin(SVM_LoadGramian,svm,0,0,0);
  if (svm->ops->loadgramian) {
    PetscCall(svm->ops->loadgramian(svm,v));
  }
  PetscLogEventEnd(SVM_LoadGramian,svm,0,0,0);

  PetscCall(PetscOptionsHasName(NULL,NULL,"-svm_view_io",&view_io));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-svm_view_gramian",&view_dataset));
  if (view_io || view_dataset) {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    PetscCall(SVMViewGramian(svm,PETSC_VIEWER_STDOUT_(comm)));
  }
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMViewGramian"
/*@
  SVMViewGramian - Visualizes (precomputed) Gramian matrix.

  Collective on PetscViewer

  Input Parameters:
+ svm - SVM context
- v - viewer

  Level: intermediate

.seealso SVM, SVMSetGramian(), SVMGetGramian(), SVMLoadGramian(), SVMViewTrainingDataset()
@*/
PetscErrorCode SVMViewGramian(SVM svm,PetscViewer v)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);

  if (svm->ops->viewgramian) {
    PetscCall(svm->ops->viewgramian(svm,v) );
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscCall(PetscObjectTypeCompare((PetscObject) v,PETSCVIEWERASCII,&isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject) v,PETSCVIEWERHDF5,&ishdf5));
  PetscCall(PetscObjectTypeCompare((PetscObject) v,PETSCVIEWERBINARY,&isbinary));

  PetscLogEventBegin(SVM_LoadDataset,svm,0,0,0);
  if (isascii) {
    PetscCall(DatasetLoad_SVMLight(Xt,y,v));
  } else if (ishdf5 || isbinary) {
    PetscCall(DatasetLoad_Binary(Xt,y,v));
  } else {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    PetscCall(PetscObjectGetType((PetscObject) v,&type_name));

    SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMLoadDataset",type_name);
  }
  PetscLogEventEnd(SVM_LoadDataset,svm,0,0,0);

  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMLoadTrainingDataset"
/*@
  SVMLoadTrainingDataset - Loads training dataset.

  Input Parameters:
+ svm - SVM context
- v - viewer

  Level: intermediate

.seealso SVMLoadCalibrationDataset(), SVMLoadTestDataset(), SVM
@*/
PetscErrorCode SVMLoadTrainingDataset(SVM svm,PetscViewer v)
{
  MPI_Comm  comm;

  Mat       Xt_training;
  Mat       Xt_biased;
  Vec       y_training;

  PetscReal user_bias;
  PetscInt  mod;

  PetscBool view_io,view_dataset;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);

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
    PetscCall(SVMGetUserBias(svm,&user_bias));
    PetscCall(MatBiasedCreate(Xt_training,user_bias,&Xt_biased));
    Xt_training = Xt_biased;
  }
  PetscCall(SVMSetTrainingDataset(svm,Xt_training,y_training));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-svm_view_io",&view_io));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-svm_view_training_dataset",&view_dataset));
  if (view_io || view_dataset) {
    PetscCall(SVMViewTrainingDataset(svm,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject) v))));
  }

  /* Free memory */
  PetscCall(MatDestroy(&Xt_training));
  PetscCall(VecDestroy(&y_training));

  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMViewTrainingDataset"
/*@
  SVMViewTrainingDataset - Views details associated with training dataset such as number of positive and negative samples, features, etc.

  Input Parameters:
+ svm - SVM context
- v - visualization context

  Level: beginner

  Options Database Keys:
+ -svm_view_io - Prints info on training dataset (as well as test dataset) at the end of SVMLoadTrainingDataset() and/or SVMLoadTestDataset, respectively.
- -svm_view_training_dataset - Prints info just on test dataset at the end of SVMLoadTrainingDataset().

.seealso SVM, PetscViewer, SVMViewDataset(), SVMViewTestDataset()
@*/
PetscErrorCode SVMViewTrainingDataset(SVM svm,PetscViewer v)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMLoadTestDataset"
/*@
  SVMLoadTestDataset - Loads test dataset.

  Input Parameters:
+ svm - SVM context
- v - viewer

  Level: intermediate

.seealso SVMLoadTrainingDataset(), SVMLoadCalibrationDataset(), SVM
@*/
PetscErrorCode SVMLoadTestDataset(SVM svm,PetscViewer v)
{
  MPI_Comm    comm;

  Mat         Xt_test;
  Mat         Xt_biased;
  Vec         y_test;

  PetscReal   user_bias;
  PetscInt    mod;

  PetscBool  view_io,view_test_dataset;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);

  PetscCall(PetscObjectGetComm((PetscObject) svm,&comm));
  PetscCall(MatCreate(comm,&Xt_test));
  /* Create matrix of test samples */
  PetscCall(PetscObjectSetName((PetscObject) Xt_test,"Xt_test"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) Xt_test,"Xt_test_"));
  PetscCall(MatSetFromOptions(Xt_test));
  /* Create label vector of test samples */
  PetscCall(VecCreate(comm,&y_test));
  PetscCall(VecSetFromOptions(y_test));
  PetscCall(PetscObjectSetName((PetscObject) y_test,"y_test"));

  PetscCall(PetscLogEventBegin(SVM_LoadDataset,svm,0,0,0));
  PetscCall(PetscViewerLoadSVMDataset(Xt_test,y_test,v));
  PetscCall(PetscLogEventEnd(SVM_LoadDataset,svm,0,0,0));

  PetscCall(SVMGetMod(svm,&mod));
  if (mod == 2) {
    PetscCall(SVMGetUserBias(svm,&user_bias));
    PetscCall(MatBiasedCreate(Xt_test,user_bias,&Xt_biased));
    Xt_test = Xt_biased;
  }
  PetscCall(SVMSetTestDataset(svm,Xt_test,y_test));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-svm_view_io",&view_io));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-svm_view_test_dataset",&view_test_dataset));
  if (view_io || view_test_dataset) {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    PetscCall(SVMViewTestDataset(svm,PETSC_VIEWER_STDOUT_(comm)));
  }

  /* Free memory */
  PetscCall(MatDestroy(&Xt_test));
  PetscCall(VecDestroy(&y_test));
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMViewTestDataset"
/*@
  SVMViewTestDataset - Views details associated with test dataset such as number of positive and negative samples, features, etc.

  Input Parameters:
+ svm - SVM context
- v - visualization context

  Level: beginner

  Options Database Keys:
+ -svm_view_io - Prints info on test dataset (as well as training dataset) at the end of SVMLoadTestDataset() and/or SVMLoadTrainingDataset, respectively.
- -svm_view_test_dataset - Prints info just on test dataset at the end of SVMLoadTestDataset().

.seealso SVM, PetscViewer, SVMViewDataset(), SVMViewTrainingDataset()
@*/
PetscErrorCode SVMViewTestDataset(SVM svm,PetscViewer v)
{
  MPI_Comm   comm;

  Mat        Xt;
  Vec        y;

  PetscBool  isascii;
  const char *type_name = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);

  PetscCall(SVMGetTestDataset(svm,&Xt,&y));
  if (!Xt || !y) {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    SETERRQ(comm,PETSC_ERR_ARG_NULL,"Test dataset is not set");
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

    SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewTestDataset",type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMLoadCalibrationDataset"
/*@
  SVMLoadCalibrationDataset - Loads calibration dataset.

  Input Parameters:
+ svm - SVM context
- v - viewer

  Level: intermediate

.seealso SVMLoadTrainingDataset(), SVMLoadTestDataset(), SVM
@*/
PetscErrorCode SVMLoadCalibrationDataset(SVM svm,PetscViewer v)
{
  MPI_Comm  comm;
  PetscBool view_io,view_calibration_dataset;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscTryMethod(svm,"SVMLoadCalibrationDataset_C",(SVM,PetscViewer),(svm,v));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-svm_view_io",&view_io));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-svm_view_calibration_dataset",&view_calibration_dataset));
  if (view_io || view_calibration_dataset) {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    PetscCall(SVMViewCalibrationDataset(svm,PETSC_VIEWER_STDOUT_(comm)));
  }
  PetscFunctionReturnI(0);
}

/*@
  SVMViewCalibrationDataset - Views details associated with calibration dataset such as number of positive and negative samples, features, etc.

  Input Parameters:
+ svm - SVM context
- v - visualization context

  Level: beginner

  Options Database Keys:
+ -svm_view_io - Prints info on calibration dataset (as well as training/Test dataset) at the end of SVMLoadCalibrationDataset(), and SVMLoadTrainingDataset()/SVMLoadTestDataset(), respectively.
- -svm_view_calibration_dataset - Prints info just on calibration dataset at the end of SVMLoadCalibrationDataset().

.seealso SVM, PetscViewer, SVMViewDataset(), SVMViewTrainingDataset(), SVMViewTestDataset()
@*/
PetscErrorCode SVMViewCalibrationDataset(SVM svm,PetscViewer v)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscTryMethod(svm,"SVMViewCalibrationDataset_C",(SVM,PetscViewer),(svm,v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMViewDataset"
/*@
  SVMViewDataset - Views details associated with dataset such as number of positive and negative samples, features, etc.

  Input Parameters:
+ svm - SVM context
- v - visualization context

  Level: beginner

.seealso SVM, PetscViewer, SVMViewTestDataset(), SVMViewTrainingDataset()
@*/
PetscErrorCode SVMViewDataset(SVM svm,Mat Xt,Vec y,PetscViewer v)
{
  MPI_Comm   comm;
  const char *type_name = NULL;

  PetscInt   M,M_plus,M_minus,N;
  PetscReal  per_plus,per_minus;

  PetscInt   svm_mod;

  Mat        Xt_inner;
  PetscReal  bias;

  Vec        y_max = NULL;
  IS         is_plus = NULL;
  PetscReal  max;

  PetscBool  isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(Xt,MAT_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,4);

  PetscCall(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(MatGetSize(Xt,&M,&N));

    PetscCall(VecDuplicate(y,&y_max));
    PetscCall(VecMax(y,NULL,&max));
    PetscCall(VecSet(y_max,max));

    PetscCall(VecWhichEqual(y,y_max,&is_plus));
    PetscCall(ISGetSize(is_plus,&M_plus));

    per_plus  = ((PetscReal) M_plus / (PetscReal) M) * 100.;
    M_minus   = M - M_plus;
    per_minus = 100. - per_plus;

    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject) Xt,v));
    PetscCall(SVMGetMod(svm,&svm_mod));
    if (svm_mod == 2) {
      N -= 1;

      PetscCall(MatBiasedGetInnerMat(Xt,&Xt_inner));
      PetscCall(MatBiasedGetBias(Xt,&bias));

      PetscCall(PetscViewerASCIIPushTab(v));
      PetscCall(PetscViewerASCIIPrintf(v,"Samples are augmented with additional dimension by means of bias %.2f\n",bias));
      PetscCall(PetscViewerASCIIPrintf(v,"inner"));
      PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject) Xt_inner,v));
      PetscCall(PetscViewerASCIIPopTab(v));

      PetscCall(PetscViewerASCIIPushTab(v));
    }

    PetscCall(PetscViewerASCIIPushTab(v));
    PetscCall(PetscViewerASCIIPrintf(v,"samples\t%5" PetscInt_FMT "\n",M));
    PetscCall(PetscViewerASCIIPrintf(v,"samples+\t%5" PetscInt_FMT " (%.2f%%)\n",M_plus,(double)per_plus));
    PetscCall(PetscViewerASCIIPrintf(v,"samples-\t%5" PetscInt_FMT " (%.2f%%)\n",M_minus,(double)per_minus));
    if (svm_mod == 2) {
      PetscCall(PetscViewerASCIIPrintf(v,"features\t%5" PetscInt_FMT " (%" PetscInt_FMT ")\n",N,N + 1));
      PetscCall(PetscViewerASCIIPopTab(v));
    } else {
      PetscCall(PetscViewerASCIIPrintf(v,"features\t%5" PetscInt_FMT "\n",N));
    }
    PetscCall(PetscViewerASCIIPopTab(v));

    /* Memory deallocation */
    PetscCall(VecDestroy(&y_max));
    PetscCall(ISDestroy(&is_plus));
  } else {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    PetscCall(PetscObjectGetType((PetscObject) v,&type_name));

    SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewDataset",type_name);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
