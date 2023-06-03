
#include "probimpl.h"
#include "../binary/binaryimpl.h"

#include "../../utils/report.h"


PetscErrorCode SVMReset_Probability(SVM svm)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&svm_prob->Xt_training));
  PetscCall(VecDestroy(&svm_prob->y_training));
  PetscCall(MatDestroy(&svm_prob->Xt_calib));
  PetscCall(VecDestroy(&svm_prob->y_calib));

  PetscCall(SVMDestroy(&svm_prob->inner));
  PetscCall(TaoDestroy(&svm_prob->tao));

  PetscCall(VecDestroyVecs(2,&svm_prob->work_vecs));
  PetscCall(VecDestroy(&svm_prob->vec_dist));
  PetscCall(VecDestroy(&svm_prob->vec_targets));

  svm_prob->Xt_training = NULL;
  svm_prob->y_training  = NULL;

  svm_prob->Xt_calib    = NULL;
  svm_prob->y_calib     = NULL;

  svm_prob->inner       = NULL;
  svm_prob->tao         = NULL;

  svm_prob->work_vecs   = NULL;
  svm_prob->vec_dist    = NULL;
  svm_prob->vec_targets = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMDestroy_Probability(SVM svm)
{

  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetInnerSVM_C"           ,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetTao_C"                ,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetTrainingDataset_C"    ,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetTrainingDataset_C"    ,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetCalibrationDataset_C" ,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetCalibrationDataset_C" ,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetLabels_C"             ,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMLoadCalibrationDataset_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMViewCalibrationDataset_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetOptionsPrefix_C"      ,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetOptionsPrefix_C"      ,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMAppendOptionsPrefix_C"   ,NULL));

  PetscCall(SVMDestroyDefault(svm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMView_Probability(SVM svm,PetscViewer v)
{

  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMViewScore_Probability(SVM svm,PetscViewer v)
{
  MPI_Comm        comm;

  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;
  SVM             svm_uncalibrated;

  PetscReal       threshold;
  PetscBool       isascii;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject) svm);
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);

  PetscCall(PetscObjectTypeCompare((PetscObject) v,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(SVMGetInnerSVM(svm,&svm_uncalibrated));
    PetscCall(SVMViewScore(svm_uncalibrated,v));

    PetscCall(SVMProbGetThreshold(svm,&threshold));

    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject) svm,v));
    PetscCall(PetscViewerASCIIPushTab(v));
    PetscCall(PetscViewerASCIIPrintf(v,"model performance scores with threshold=%.2f\n",(double) threshold));
    PetscCall(PetscViewerASCIIPushTab(v));
    PetscCall(SVMPrintBinaryClassificationReport(svm,svm_prob->confusion_matrix,svm_prob->model_scores,v));
    PetscCall(PetscViewerASCIIPopTab(v));
    PetscCall(PetscViewerASCIIPopTab(v));

    /* Clean up */
    PetscCall(SVMDestroy(&svm_uncalibrated));
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewScore", ((PetscObject)v)->type_name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMSetTrainingDataset_Probability(SVM svm,Mat Xt_training,Vec y_training)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  Mat             Xt_calib;
  PetscInt        k,l,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(Xt_training,MAT_CLASSID,2);
  PetscCheckSameComm(svm,1,Xt_training,2);
  PetscValidHeaderSpecific(y_training,VEC_CLASSID,3);
  PetscCheckSameComm(svm,1,y_training,3);

  PetscCall(MatGetSize(Xt_training,&k,NULL));
  PetscCall(VecGetSize(y_training,&l));
  if (k != l) {
    SETERRQ(PetscObjectComm((PetscObject) Xt_training),PETSC_ERR_ARG_SIZ,"Dimensions are incompatible, "
                                                                         "X_training(%" PetscInt_FMT ",) !="
                                                                         "y_training(%" PetscInt_FMT ")",k,l);
  }

  PetscCall(SVMGetCalibrationDataset(svm,&Xt_calib,NULL));
  if (Xt_calib != NULL) {
    PetscCall(MatGetSize(Xt_training,NULL,&l));
    PetscCall(MatGetSize(Xt_calib,NULL,&n));

    if (l != n) {
      SETERRQ(PetscObjectComm((PetscObject) Xt_calib),PETSC_ERR_ARG_SIZ,"Dimensions are incompatible, "
                                                                        "X_training(,%" PetscInt_FMT ") !="
                                                                        "X_calib(,%" PetscInt_FMT ")",l,n);
    }
  }

  PetscCall(MatDestroy(&svm_prob->Xt_training));
  svm_prob->Xt_training = Xt_training;
  PetscCall(PetscObjectReference((PetscObject) Xt_training));

  PetscCall(VecDestroy(&svm_prob->y_training));
  svm_prob->y_training = y_training;
  PetscCall(PetscObjectReference((PetscObject) y_training));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMGetTrainingDataset_Probability(SVM svm,Mat *Xt_training,Vec *y_training)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscFunctionBegin;
  if (Xt_training) {
    PetscValidPointer(Xt_training,2);
    *Xt_training = svm_prob->Xt_training;
  }
  if (y_training) {
    PetscValidPointer(y_training,3);
    *y_training = svm_prob->y_training;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SVMGetNumberPositiveNegativeSamples_Probability_Private(Vec y,PetscInt *ntarget_lo,PetscInt *ntarget_hi)
{
  Vec       tmp;
  IS        is_p,is_n;

  PetscInt  lo,hi,n;
  PetscReal max;

  PetscFunctionBegin;
  /* Determine index sets of positive and negative samples */
  PetscCall(VecGetOwnershipRange(y,&lo,&hi));
  PetscCall(VecMax(y,NULL,&max));
  PetscCall(VecDuplicate(y,&tmp));
  PetscCall(VecSet(tmp,max));

  /* Index set for positive and negative samples */
  PetscCall(VecWhichEqual(y,tmp,&is_p));
  PetscCall(ISComplement(is_p,lo,hi,&is_n));

  PetscCall(ISGetSize(is_n,&n));
  *ntarget_lo = n;
  PetscCall(ISGetSize(is_p,&n));
  *ntarget_hi = n;

  /* Clean up */
  PetscCall(VecDestroy(&tmp));
  PetscCall(ISDestroy(&is_p));
  PetscCall(ISDestroy(&is_n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMSetCalibrationDataset_Probability(SVM svm,Mat Xt_calib,Vec y_calib)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  Mat             Xt_training;

  PetscInt        k,l;
  PetscInt        n;

  PetscInt        ntargets_lo,ntargets_hi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(Xt_calib,MAT_CLASSID,2);
  PetscCheckSameComm(svm,1,Xt_calib,2);
  PetscValidHeaderSpecific(y_calib,VEC_CLASSID,3);
  PetscCheckSameComm(svm,1,y_calib,3);

  PetscCall(MatGetSize(Xt_calib,&k,NULL));
  PetscCall(VecGetSize(y_calib,&l));
  if (k != l) {
    SETERRQ(PetscObjectComm((PetscObject) Xt_calib),PETSC_ERR_ARG_SIZ,"Dimensions are incompatible, "
                                                                      "X_calib(%" PetscInt_FMT ",) != "
                                                                      "y_calib(%" PetscInt_FMT ")",k,l);
  }

  PetscCall(SVMGetTrainingDataset(svm,&Xt_training,NULL));
  if (Xt_training != NULL) {
    PetscCall(MatGetSize(Xt_calib,NULL,&l));
    PetscCall(MatGetSize(Xt_training,NULL,&n));

    if (l != n) {
      SETERRQ(PetscObjectComm((PetscObject) Xt_calib),PETSC_ERR_ARG_SIZ,"Dimensions are incompatible, "
                                                                        "X_calib(,%" PetscInt_FMT ") !="
                                                                        "X_training(,%" PetscInt_FMT ")",l,n);
    }
  }

  PetscCall(MatDestroy(&svm_prob->Xt_calib));
  svm_prob->Xt_calib = Xt_calib;
  PetscCall(PetscObjectReference((PetscObject) Xt_calib));

  PetscCall(VecDestroy(&svm_prob->y_calib));
  svm_prob->y_calib = y_calib;
  PetscCall(PetscObjectReference((PetscObject) y_calib));

  /* Determine a number of positive and negative samples in calibration dataset */
  PetscCall(SVMGetNumberPositiveNegativeSamples_Probability_Private(y_calib,&ntargets_lo,&ntargets_hi));
  svm_prob->Nn_calib = ntargets_lo;
  svm_prob->Np_calib = ntargets_hi;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMGetCalibrationDataset_Probability(SVM svm,Mat *Xt_calib,Vec *y_calib)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscFunctionBegin;
  if (Xt_calib) {
    PetscValidPointer(Xt_calib,2);
    *Xt_calib = svm_prob->Xt_calib;
  }

  if (y_calib) {
    PetscValidPointer(y_calib,3);
    *y_calib = svm_prob->y_calib;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMGetLabels_Probability(SVM svm,const PetscReal *labels[])
{
  SVM             svm_uncalibrated;
  const PetscReal *labels_inner;

  PetscFunctionBegin;
  PetscCall(SVMGetInnerSVM(svm,&svm_uncalibrated));
  PetscCall(SVMGetLabels(svm_uncalibrated,&labels_inner));
  *labels = labels_inner;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SVMCreateTAO_Probability_Private(SVM svm,Tao *tao)
{
  Tao        tao_inner;

  KSP        ksp;
  PC         pc;

  const char *prefix = NULL;

  PetscFunctionBegin;
  PetscCall(TaoCreate(MPI_COMM_SELF,&tao_inner));
  PetscCall(TaoSetType(tao_inner,TAONLS));

  /* Disable preconditioning */
  PetscCall(TaoGetKSP(tao_inner,&ksp));
  PetscCall(PCCreate(MPI_COMM_SELF,&pc));
  PetscCall(PCSetType(pc,PCNONE));
  PetscCall(KSPSetPC(ksp,pc));

  /* Set prefix */
  PetscCall(SVMGetOptionsPrefix(svm,&prefix));
  PetscCall(TaoAppendOptionsPrefix(tao_inner,prefix));

  *tao = tao_inner;

  /* Clean up */
  PetscCall(PCDestroy(&pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMGetTao_Probability(SVM svm,Tao *tao)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;
  Tao             tao_inner;

  PetscFunctionBegin;
  if (!svm_prob->tao) {
    PetscCall(SVMCreateTAO_Probability_Private(svm,&tao_inner));
    svm_prob->tao = tao_inner;
  }

  PetscCall(PetscObjectReference((PetscObject) svm_prob->tao));
  *tao = svm_prob->tao;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SVMTransformUncalibratedPredictions_Probability_Private(SVM svm)
{
  SVM_Probability   *svm_prob = (SVM_Probability *) svm->data;
  SVM               svm_uncalibrated;

  Mat               Xt_calib;
  Vec               y_calib;

  VecScatter        scatter;
  Vec               dist;

  Vec               vec_labels;
  const PetscReal   *labels = NULL;
  IS                is_labels;

  Vec               vec_targets,vec_targets_sub;
  PetscReal         target_prob[2];

  PetscBool         label_to_target_prob;

  PetscInt          Np,Nn;
  PetscInt          i;

  PetscFunctionBegin;
  PetscCall(SVMGetInnerSVM(svm,&svm_uncalibrated));
  PetscCall(SVMGetCalibrationDataset(svm,&Xt_calib,&y_calib));

  /* Get distance samples from a separating hyperplane */
  PetscCall(VecDestroy(&svm_prob->vec_dist));

  PetscCall(SVMGetDistancesFromHyperplane(svm_uncalibrated,Xt_calib,&dist));
  PetscCall(VecScatterCreateToZero(dist,&scatter,&svm_prob->vec_dist));
  PetscCall(VecScatterBegin(scatter,dist,svm_prob->vec_dist,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter,dist,svm_prob->vec_dist,INSERT_VALUES,SCATTER_FORWARD));

  PetscCall(SVMProbGetConvertLabelsToTargetProbability(svm,&label_to_target_prob));
  if (label_to_target_prob) {
    /*
     Transform labels to target probabilities as it proposed in
     http://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf
    */
    PetscCall(SVMGetNumberPositiveNegativeSamples_Probability_Private(y_calib,&Np,&Nn));
    target_prob[0] = 1. / (Nn + 2);
    target_prob[1] = (Np + 1.) / (Np + 2.);
  } else {
    target_prob[0] = 0.;
    target_prob[1] = 1.;
  }

  PetscCall(SVMGetLabels(svm,&labels));
  PetscCall(VecDuplicate(y_calib,&vec_labels));
  PetscCall(VecDuplicate(y_calib,&vec_targets));
  PetscCall(VecSetFromOptions(vec_labels));
  PetscCall(VecSetFromOptions(vec_targets));

  for (i = 0; i < 2; ++i) {
    PetscCall(VecSet(vec_labels,labels[i]));
    PetscCall(VecWhichEqual(y_calib,vec_labels,&is_labels));
    PetscCall(VecGetSubVector(vec_targets,is_labels,&vec_targets_sub));
    PetscCall(VecSet(vec_targets_sub,target_prob[i]));
    PetscCall(VecRestoreSubVector(vec_targets,is_labels,&vec_targets_sub));
    PetscCall(ISDestroy(&is_labels));
  }

  /* Scatter targets to a root process */
  PetscCall(VecScatterCreateToZero(vec_targets,&scatter,&svm_prob->vec_targets));
  PetscCall(VecScatterBegin(scatter,vec_targets,svm_prob->vec_targets,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter,vec_targets,svm_prob->vec_targets,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&scatter));

  /* Clean up */
  PetscCall(VecDestroy(&dist));
  PetscCall(VecDestroy(&vec_labels));
  PetscCall(VecDestroy(&vec_targets));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoFormFunctionGradient_Probability_Private(Tao tao,Vec params_sigmoid,PetscReal *fnc,Vec g,void *ctx)
{
  SVM             svm       = (SVM) ctx;
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscReal       g_arr[2]  = {0.,0.};
  PetscInt        idx[2]    = {0,1};

  PetscReal       fnc_inner = 0.;
  PetscReal       tmp;

  Vec             *work_vecs = NULL;
  Vec             *work_sub  = NULL;
  PetscInt        N_works    = 2;

  IS              is_p,is_n;
  PetscInt        i,N;

  PetscReal       A,B;
  const PetscReal *params_sigmoid_arr = NULL;

  PetscFunctionBegin;
  PetscCall(VecGetSize(svm_prob->vec_dist,&N));

  if (!svm_prob->work_vecs) {
    PetscCall(VecDuplicateVecs(svm_prob->vec_dist,N_works,&work_vecs));
    for (i = 0; i < N_works; ++i) {
      PetscCall(VecSetFromOptions(work_vecs[i]));
    }
    svm_prob->work_vecs = work_vecs;
  } else {
    work_vecs = svm_prob->work_vecs;
  }

  work_sub = svm_prob->work_sub;

  PetscCall(VecSet(work_vecs[0],1.));
  PetscCall(VecSet(work_vecs[1],0.));

  /* Actual parameters of sigmoid */
  PetscCall(VecGetArrayRead(params_sigmoid,&params_sigmoid_arr));
  A = params_sigmoid_arr[0]; B = params_sigmoid_arr[1];
  PetscCall(VecRestoreArrayRead(params_sigmoid,&params_sigmoid_arr));

  PetscCall(VecAXPBY(work_vecs[0],A,B,svm_prob->vec_dist));        /* work[0] <- A * dist + B  (AdpB) */
  PetscCall(VecDot(svm_prob->vec_targets,work_vecs[0],&tmp));      /* target^T * AdpB */
  fnc_inner = tmp;

  PetscCall(VecWhichGreaterThan(work_vecs[0],work_vecs[1],&is_p));
  PetscCall(ISComplement(is_p,0,N,&is_n));

  /*
    CASE: A * dist + B >= 0
   */

  PetscCall(VecGetSubVector(work_vecs[0],is_p,&work_sub[0]));
  PetscCall(VecGetSubVector(work_vecs[1],is_p,&work_sub[1]));

  PetscCall(VecScale(work_sub[0],-1.));                               /* work[0,is+] <- -1. * AdpB[is+]                             */
  PetscCall(VecExp(work_sub[0]));                                     /* work[0,is+] <- exp(-1. * AdpB[is+])                        */
  PetscCall(VecCopy(work_sub[0],work_sub[1]));                        /* work[1,is+] <- work[0,is+]                                 */
  PetscCall(VecShift(work_sub[0],1.));                                /* work[0,is+] <- 1. + work[0,is+]                            */

  PetscCall(VecPointwiseDivide(work_sub[1],work_sub[1],work_sub[0])); /* work[1,is+] <- exp(-work[0,is+]) / (1 + exp(-work[0,is+])) */

  PetscCall(VecRestoreSubVector(work_vecs[0],is_p,&work_sub[0]));
  PetscCall(VecRestoreSubVector(work_vecs[1],is_p,&work_sub[1]));

  /*
    CASE: A * dist + B < 0
   */

  PetscCall(VecGetSubVector(work_vecs[0],is_n,&work_sub[0]));
  PetscCall(VecGetSubVector(work_vecs[1],is_n,&work_sub[1]));

  PetscCall(VecSum(work_sub[0],&tmp));
  fnc_inner -= tmp;

  PetscCall(VecExp(work_sub[0]));                                   /* work[0,is-] <- exp(AdpB[is-])                        */
  PetscCall(VecShift(work_sub[0],1.));                              /* work[0,is-] <- 1. + exp(AdpB[is-])                   */
  PetscCall(VecCopy(work_sub[0],work_sub[1]));                      /* work[1,is-] <- work[0,is-]                           */

  PetscCall(VecReciprocal(work_sub[1]));                            /* work[1,is-] <- 1. / (1. + exp(AdpB[is-])             */

  PetscCall(VecRestoreSubVector(work_vecs[0],is_n,&work_sub[0]));
  PetscCall(VecRestoreSubVector(work_vecs[1],is_n,&work_sub[1]));

  /*
    Compute value of objective function
   */

  PetscCall(VecLog(work_vecs[0]));
  PetscCall(VecSum(work_vecs[0],&tmp));
  fnc_inner += tmp;

  *fnc = fnc_inner; /* Update a value of objective function */

  /*
    Compute gradient
   */

  PetscCall(VecAYPX(work_vecs[1],-1.,svm_prob->vec_targets));
  PetscCall(VecSum(work_vecs[1],&g_arr[1]));
  PetscCall(VecDot(work_vecs[1],svm_prob->vec_dist,&g_arr[0]));

  PetscCall(VecSetValues(g,2,idx,g_arr,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(g));
  PetscCall(VecAssemblyEnd(g));

  /* Clean up */
  PetscCall(ISDestroy(&is_p));
  PetscCall(ISDestroy(&is_n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoFormHessian_Probability_Private(Tao tao,Vec params_sigmoid,Mat H,Mat Hpre,void *ctx)
{
  SVM             svm       = (SVM) ctx;
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscReal       H_arr[4]  = {0.,0.,0.,0.};
  PetscInt        idxm[2]   = {0,1};
  PetscInt        idxn[2]   = {0,1};

  Vec             *work_vecs = NULL;
  Vec             *work_sub  = NULL;
  PetscInt        N_works    = 2;

  IS              is_p,is_n;
  PetscInt        i,N;

  PetscReal       A,B;
  const PetscReal *params_sigmoid_arr = NULL;

  PetscFunctionBegin;
  PetscCall(VecGetSize(svm_prob->vec_dist,&N));

  if (!svm_prob->work_vecs) {
    PetscCall(VecDuplicateVecs(svm_prob->vec_dist,N_works,&work_vecs));
    for (i = 0; i < N_works; ++i) {
      PetscCall(VecSetFromOptions(work_vecs[i]));
    }
    svm_prob->work_vecs = work_vecs;
  } else {
    work_vecs = svm_prob->work_vecs;
  }

  work_sub = svm_prob->work_sub;

  PetscCall(VecSet(work_vecs[0],1.));
  PetscCall(VecSet(work_vecs[1],0.));

  /* Actual parameters of sigmoid */
  PetscCall(VecGetArrayRead(params_sigmoid,&params_sigmoid_arr));
  A = params_sigmoid_arr[0]; B = params_sigmoid_arr[1];
  PetscCall(VecRestoreArrayRead(params_sigmoid,&params_sigmoid_arr));

  PetscCall(VecAXPBY(work_vecs[0],A,B,svm_prob->vec_dist));           /* work[0] <- A * dist + B  (AdpB) */

  PetscCall(VecWhichGreaterThan(work_vecs[0],work_vecs[1],&is_p));
  PetscCall(ISComplement(is_p,0,N,&is_n));

  /*
   CASE: A * dist + B >= 0
   */

  PetscCall(VecGetSubVector(work_vecs[0],is_p,&work_sub[0]));
  PetscCall(VecGetSubVector(work_vecs[1],is_p,&work_sub[1]));

  PetscCall(VecScale(work_sub[0],-1.));                               /* work[0,is+] <- -1. * AdpB[is+]           */
  PetscCall(VecExp(work_sub[0]));                                     /* work[0,is+] <- exp(-1. * AdpB[is+])      */
  PetscCall(VecCopy(work_sub[0],work_sub[1]));                        /* work[1,is+] <- work[0,is+]               */

  PetscCall(VecShift(work_sub[1],1.));                                /* work[1,is+] <- 1. + work[1,is+]          */

  PetscCall(VecPointwiseDivide(work_sub[0],work_sub[0],work_sub[1])); /* work[0,is+] <- work[0,is+] / work[1,is+] */
  PetscCall(VecReciprocal(work_sub[1]));                              /* work[1,is+] <- 1. / work[1,is+]          */

  PetscCall(VecRestoreSubVector(work_vecs[1],is_p,&work_sub[1]));
  PetscCall(VecRestoreSubVector(work_vecs[0],is_p,&work_sub[0]));

  /*
   CASE: A * dist + B < 0
   */

  PetscCall(VecGetSubVector(work_vecs[0],is_n,&work_sub[0]));
  PetscCall(VecGetSubVector(work_vecs[1],is_n,&work_sub[1]));

  PetscCall(VecExp(work_sub[0]));                                     /* work[0,is-] <- exp(AdpB[is-])           */
  PetscCall(VecCopy(work_sub[0],work_sub[1]));                        /* work[1,is-] <- work[0,is-]              */
  PetscCall(VecShift(work_sub[0],1.));                                /* work[0,is-] <- 1. + exp(AdpB[is-])      */

  PetscCall(VecPointwiseDivide(work_sub[1],work_sub[1],work_sub[0])); /* work[1,is-] <- work[1,is-] / work[0,is-] */
  PetscCall(VecReciprocal(work_sub[0]));                              /* work[0,is-] <- 1. / work[0,is-]          */

  PetscCall(VecRestoreSubVector(work_vecs[0],is_n,&work_sub[0]));
  PetscCall(VecRestoreSubVector(work_vecs[1],is_n,&work_sub[1]));

  /* Form Hessian */
  PetscCall(VecPointwiseMult(work_vecs[0],work_vecs[1],work_vecs[0]));
  PetscCall(VecPointwiseMult(work_vecs[1],svm_prob->vec_dist,svm_prob->vec_dist));

  PetscCall(VecDot(work_vecs[0],work_vecs[1],&H_arr[0]));
  PetscCall(VecDot(svm_prob->vec_dist,work_vecs[0],&H_arr[1]));
  H_arr[2] = H_arr[1];
  PetscCall(VecSum(work_vecs[0],&H_arr[3]));

  H_arr[0] += 1e-12;
  H_arr[3] += 1e-12;

  /* Update Hessian */
  PetscCall(MatSetValues(H,2,idxm,2,idxn,H_arr,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));

  /* Clean up */
  PetscCall(ISDestroy(&is_p));
  PetscCall(ISDestroy(&is_n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SVMSetUp_Tao_Private(SVM svm)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;
  Tao             tao;

  Mat             H;
  Vec             g;
  Vec             x_init;

  PetscInt        Nn,Np;
  PetscReal       v;

  PetscFunctionBegin;
  /* Create Hessian */
  PetscCall(MatCreateSeqDense(MPI_COMM_SELF,2,2,NULL,&H));
  PetscCall(PetscObjectSetName((PetscObject) H,"H_bce"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) H,"H_bce_"));
  PetscCall(MatSetFromOptions(H));

  PetscCall(VecCreateSeq(MPI_COMM_SELF,2,&g));
  PetscCall(VecSetFromOptions(g));

  PetscCall(VecCreateSeq(MPI_COMM_SELF,2,&x_init));
  PetscCall(VecSetFromOptions(x_init));

  PetscCall(SVMGetTao(svm,&tao));

  /* Set initial guess */
  Nn = svm_prob->Nn_calib;
  Np = svm_prob->Np_calib;

  /* Set a function for forming and updating gradient and Hessian */
  PetscCall(TaoSetObjectiveAndGradient(tao,g,TaoFormFunctionGradient_Probability_Private,svm));
  PetscCall(TaoSetHessian(tao,H,H,TaoFormHessian_Probability_Private,svm));

  /* Set initial guess */
  PetscCall(VecSetValue(x_init,0,0.,INSERT_VALUES));
  v = PetscLogReal((Nn + 1.) / (Np + 1.));
  PetscCall(VecSetValue(x_init,1,v,INSERT_VALUES));

  PetscCall(VecAssemblyBegin(x_init));
  PetscCall(VecAssemblyEnd(x_init));
  PetscCall(TaoSetSolution(tao,x_init));

  PetscCall(TaoSetTolerances(tao,1e-5,1e-5,1e-5));

  /* Set tao solver from cli options */
  if (svm->setfromoptionscalled) {
    PetscCall(TaoSetFromOptions(tao));
  }

  /* Clean up */
  PetscCall(TaoDestroy(&tao));
  PetscCall(MatDestroy(&H));
  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&x_init));

  svm->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMSetUp_Probability(SVM svm)
{
  MPI_Comm    comm;
  PetscMPIInt rank;

  SVM         svm_uncalibrated;

  Mat         Xt_training;
  Vec         y_training;

  Mat         Xt_calib;
  Vec         y_calib;

  Mat         Xt_test;
  Vec         y_test;

  PetscFunctionBegin;
  if (svm->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(SVMGetInnerSVM(svm,&svm_uncalibrated));
  if (svm->setfromoptionscalled) {
    PetscCall(SVMSetFromOptions(svm_uncalibrated));
  }

  PetscCall(SVMGetCalibrationDataset(svm,&Xt_calib,&y_calib));
  if (Xt_calib != NULL) {
    PetscCall(SVMGetTrainingDataset(svm,&Xt_training,&y_training));
  } else {
    Xt_training = Xt_calib;
    y_training  = y_calib;
  }

  PetscCall(SVMSetTrainingDataset(svm_uncalibrated,Xt_training,y_training));

  PetscCall(SVMGetTestDataset(svm,&Xt_test,&y_test));
  if (Xt_test && y_test) {
    PetscCall(SVMSetTestDataset(svm_uncalibrated,Xt_test,y_test));
  }

  PetscCall(SVMSetUp(svm_uncalibrated));

  /* Set up TAO solver. Since problem is small (dim=2), it is being solved on a root process. */
  PetscCall(PetscObjectGetComm((PetscObject)svm,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  if (rank == 0) {
    PetscCall(SVMSetUp_Tao_Private(svm));
  }

  /* Clean up */
  PetscCall(SVMDestroy(&svm_uncalibrated));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMSetOptionsPrefix_Probability(SVM svm,const char prefix[])
{
  SVM inner;

  PetscFunctionBegin;
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) svm,prefix));
  PetscCall(SVMGetInnerSVM(svm,&inner));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) inner,prefix));
  PetscCall(SVMDestroy(&inner));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMAppendOptionsPrefix_Probability(SVM svm,const char prefix[])
{
  SVM inner;

  PetscFunctionBegin;
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject) svm,prefix));
  PetscCall(SVMGetInnerSVM(svm,&inner));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject) inner,prefix));
  PetscCall(SVMDestroy(&inner));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMGetOptionsPrefix_Probability(SVM svm,const char *prefix[])
{
  PetscFunctionBegin;
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) svm,prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMSetFromOptions_Probability(PetscOptionItems *PetscOptionsObject,SVM svm)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscBool       to_target_probs;
  PetscReal       threshold;
  PetscBool       flg;

  PetscFunctionBegin;
  PetscObjectOptionsBegin((PetscObject)svm);
  PetscCall(PetscOptionsBool("-svm_convert_labels_to_target_probs",
                             "Convert sample labels to target probability as suggested by Platt. Default (true).",
                             "SVMProbSetConvertLabelsToTargetProbability",
                             svm_prob->labels_to_target_probs,&to_target_probs,&flg));
  if (flg) {
    PetscCall(SVMProbSetConvertLabelsToTargetProbability(svm,to_target_probs));
  }
  PetscCall(PetscOptionsReal("-svm_threshold",
                             "Convert sample labels to target probability as suggested by Platt. Default (true).",
                             "SVMProbSetConvertLabelsToTargetProbability",
                             svm_prob->threshold,&threshold,&flg));
  if (flg) {
    PetscCall(SVMProbSetThreshold(svm,threshold));
  }
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMTrain_Probability(SVM svm)
{
  MPI_Comm    comm;
  PetscMPIInt rank;

  SVM         svm_uncalibrated;
  Tao         tao;

  PetscBool   post_train;

  PetscFunctionBegin;
  PetscCall(SVMSetUp(svm));
  PetscCall(SVMGetInnerSVM(svm,&svm_uncalibrated));

  /* Train uncalibrated svm model */
  PetscCall(SVMGetAutoPostTrain(svm_uncalibrated,&post_train));
  PetscCall(SVMTrain(svm_uncalibrated));

  /* Train logistic regression over uncalibrated model (known as Platt's scaling) */
  PetscCall(SVMTransformUncalibratedPredictions_Probability_Private(svm));
  PetscCall(SVMGetTao(svm,&tao));

  /* Solve underlying unconstrained problem on root */
  PetscCall(PetscObjectGetComm((PetscObject)svm,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  if (rank == 0) {
    PetscCall(TaoSolve(tao));
  }

  /* Run post-processing (training) of a trained probability model */
  PetscCall(SVMGetAutoPostTrain(svm,&post_train));
  if (post_train) {
    PetscCall(SVMPostTrain(svm));
  }

  /* Clean up */
  PetscCall(SVMDestroy(&svm_uncalibrated));
  PetscCall(TaoDestroy(&tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMPostTrain_Probability(SVM svm)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  MPI_Comm        comm;
  PetscMPIInt     rank;

  Tao             tao;
  Vec             x;
  const PetscReal *x_arr = NULL;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)svm,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  if (rank == 0) {
    PetscCall(SVMGetTao(svm,&tao));
    PetscCall(TaoGetSolution(tao,&x));

    PetscCall(VecGetArrayRead(x,&x_arr));
    PetscCall(PetscMemcpy(svm_prob->sigmoid_params,x_arr,2 * sizeof(PetscReal)));
    PetscCall(VecRestoreArrayRead(x,&x_arr));
    /* Clean up */
    PetscCall(TaoDestroy(&tao));
  }

  /* Broadcast solution */
  PetscCallMPI(MPI_Bcast(svm_prob->sigmoid_params,2,MPIU_REAL,0,comm));

  /* Clean up */
  svm->posttraincalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMPredict_Probability(SVM svm,Mat Xt,Vec *y_out)
{
  SVM       svm_uncalibrated;

  Vec       pred;

  PetscInt  N_work_vec = 2;
  Vec       *work_vecs = NULL;
  Vec       work_sub[N_work_vec];

  IS        is_p,is_n;
  PetscInt  lo,hi;

  PetscReal A,B;

  PetscFunctionBegin;
  if (!svm->posttraincalled) {
    PetscCall(SVMPostTrain(svm));
  }

  PetscCall(SVMGetInnerSVM(svm,&svm_uncalibrated));

  PetscCall(SVMGetDistancesFromHyperplane(svm_uncalibrated,Xt,&pred));  /* Get distances from hyperplane */
  PetscCall(SVMProbGetSigmoidParams(svm,&A,&B));

  /* Create working vectors */
  PetscCall(VecDuplicateVecs(pred,N_work_vec,&work_vecs));
  PetscCall(VecSet(work_vecs[0],1.));
  PetscCall(VecSet(work_vecs[1],0.));

  PetscCall(VecWhichGreaterThan(pred,work_vecs[1],&is_p));
  PetscCall(VecGetOwnershipRange(pred,&lo,&hi));
  PetscCall(ISComplement(is_p,lo,hi,&is_n));

  PetscCall(VecAXPBY(pred,B,A,work_vecs[0])); /* pred <- A * dist + B (AdpB) */

  PetscCall(VecWhichGreaterThan(pred,work_vecs[1],&is_p));
  PetscCall(VecGetOwnershipRange(pred,&lo,&hi));
  PetscCall(ISComplement(is_p,lo,hi,&is_n));

  /*
    Compute posterior probability
   */

  /*
    CASE: A * dist + B >= 0
   */
  PetscCall(VecGetSubVector(pred        ,is_p,&work_sub[0]));
  PetscCall(VecGetSubVector(work_vecs[0],is_p,&work_sub[1]));

  PetscCall(VecScale(work_sub[0],-1.));
  PetscCall(VecExp(work_sub[0]));
  PetscCall(VecAXPY(work_sub[1],1.,work_sub[0]));
  PetscCall(VecPointwiseDivide(work_sub[0],work_sub[0],work_sub[1]));

  PetscCall(VecRestoreSubVector(pred        ,is_p,&work_sub[0]));
  PetscCall(VecRestoreSubVector(work_vecs[0],is_p,&work_sub[1]));

  /*
    CASE: A * dist + B < 0
   */
  PetscCall(VecGetSubVector(pred,is_n,&work_sub[0]));

  PetscCall(VecExp(work_sub[0]));
  PetscCall(VecShift(work_sub[0],1.));
  PetscCall(VecReciprocal(work_sub[0]));

  PetscCall(VecRestoreSubVector(pred,is_n,&work_sub[0]));

  *y_out = pred;

  /* Clean up */
  PetscCall(ISDestroy(&is_n));
  PetscCall(ISDestroy(&is_p));
  PetscCall(VecDestroyVecs(N_work_vec,&work_vecs));
  PetscCall(SVMDestroy(&svm_uncalibrated));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMComputeModelScores_Probability(SVM svm,Vec y,Vec y_known)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscFunctionBegin;
  PetscCall(SVMGetBinaryClassificationReport(svm,y,y_known,svm_prob->confusion_matrix,svm_prob->model_scores));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMTest_Probability(SVM svm)
{
  SVM svm_uncalibrated;

  Mat Xt_test;
  Vec y_test;

  Vec y_pred_label;
  Vec y_pred_prob;

  PetscFunctionBegin;
  PetscCall(SVMGetTestDataset(svm,&Xt_test,&y_test));

  PetscCall(SVMGetInnerSVM(svm,&svm_uncalibrated));
  PetscCall(SVMPredict(svm_uncalibrated,Xt_test,&y_pred_label));
  PetscCall(SVMComputeModelScores(svm_uncalibrated,y_pred_label,y_test));

  PetscCall(SVMPredict(svm,Xt_test,&y_pred_prob));
  PetscCall(SVMProbConvertProbabilityToLabels(svm,y_pred_prob));
  PetscCall(SVMComputeModelScores(svm,y_pred_prob,y_test));

  /* Clean up */
  PetscCall(SVMDestroy(&svm_uncalibrated));
  PetscCall(VecDestroy(&y_pred_prob));
  PetscCall(VecDestroy(&y_pred_label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMGetInnerSVM_Probability(SVM svm,SVM *inner_out)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  MPI_Comm        comm;
  SVM             inner;

  const char      *prefix = NULL;

  PetscFunctionBegin;
  if (!svm_prob->inner) {
    PetscCall(PetscObjectGetComm((PetscObject) svm,&comm));
    PetscCall(SVMCreate(comm,&inner));
    PetscCall(SVMSetType(inner,SVM_BINARY));

    PetscCall(SVMSetOptionsPrefix(inner,"uncalibrated_"));
    PetscCall(SVMGetOptionsPrefix(svm,&prefix));
    PetscCall(SVMAppendOptionsPrefix(inner,prefix));

    svm_prob->inner = inner;
  }

  PetscCall(PetscObjectReference((PetscObject) svm_prob->inner));
  *inner_out = svm_prob->inner;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMLoadCalibrationDataset_Probability(SVM svm,PetscViewer v)
{
  MPI_Comm  comm;

  Mat       Xt_calib,Xt_biased;
  Vec       y_calib;

  PetscReal user_bias;
  PetscInt  mod;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) svm,&comm));
  PetscCall(MatCreate(comm,&Xt_calib));
  /* Create matrix of calibration samples */
  PetscCall(PetscObjectSetName((PetscObject) Xt_calib,"Xt_calib"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) Xt_calib,"Xt_calib_"));
  PetscCall(MatSetFromOptions(Xt_calib));
  /* Create label vector of calibration samples */
  PetscCall(VecCreate(comm,&y_calib));
  PetscCall(PetscObjectSetName((PetscObject) y_calib,"Xt_calib_"));
  PetscCall(VecSetFromOptions(y_calib));

  PetscCall(PetscLogEventBegin(SVM_LoadDataset,svm,0,0,0));
  PetscCall(PetscViewerLoadSVMDataset(Xt_calib,y_calib,v));
  PetscCall(PetscLogEventEnd(SVM_LoadDataset,svm,0,0,0));

  PetscCall(SVMGetMod(svm,&mod));
  if (mod == 2) {
    PetscCall(SVMGetUserBias(svm,&user_bias));
    PetscCall(MatBiasedCreate(Xt_calib,user_bias,&Xt_biased));
    Xt_calib = Xt_biased;
  }
  PetscCall(SVMSetCalibrationDataset(svm,Xt_calib,y_calib));

  /* Clean up */
  PetscCall(MatDestroy(&Xt_calib));
  PetscCall(VecDestroy(&y_calib));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMViewCalibrationDataset_Probability(SVM svm,PetscViewer v)
{
  MPI_Comm   comm;

  Mat        Xt;
  Vec        y;

  PetscBool  isascii;
  const char *type_name = NULL;

  PetscFunctionBegin;
  PetscCall(SVMGetCalibrationDataset(svm,&Xt,&y));
  if (!Xt || !y) {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    SETERRQ(comm,PETSC_ERR_ARG_NULL,"Calibration dataset is not set");
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

    SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported for SVMViewCalibrationDataset",type_name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMViewTrainingPredictions_Probability(SVM svm,PetscViewer v)
{
  MPI_Comm   comm;

  Mat        Xt_training;
  Vec        y_pred;

  PetscFunctionBegin;
  PetscCall(SVMGetTrainingDataset(svm,&Xt_training,NULL));
  if (!Xt_training) {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    SETERRQ(comm,PETSC_ERR_ARG_NULL,"Training dataset is not set");
  }

  /* View predictions on training samples */
  PetscCall(SVMPredict(svm,Xt_training,&y_pred));
  PetscCall(PetscObjectSetName((PetscObject) y_pred,"y_predictions"));
  PetscCall(VecView(y_pred,v));

  /* Free memory */
  PetscCall(VecDestroy(&y_pred));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMViewTestPredictions_Probability(SVM svm,PetscViewer v)
{
  MPI_Comm   comm;

  Mat        Xt_test;
  Vec        y_pred;

  PetscFunctionBegin;
  PetscCall(SVMGetTestDataset(svm,&Xt_test,NULL));
  if (!Xt_test) {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    SETERRQ(comm,PETSC_ERR_ARG_NULL,"Test dataset is not set");
  }

  /* View predictions on test samples */
  PetscCall(SVMPredict(svm,Xt_test,&y_pred));
  PetscCall(PetscObjectSetName((PetscObject) y_pred,"y_predictions"));
  PetscCall(VecView(y_pred,v));

  /* Free memory */
  PetscCall(VecDestroy(&y_pred));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMCreate_Probability(SVM svm)
{
  SVM_Probability *svm_prob = NULL;

  PetscFunctionBegin;
  PetscCall(PetscNew(&svm_prob));
  svm->data = (void *) svm_prob;

  svm_prob->inner       = NULL;
  svm_prob->tao         = NULL;
  svm_prob->work_vecs   = NULL;

  svm_prob->Xt_training = NULL;
  svm_prob->y_training  = NULL;
  svm_prob->Xt_calib    = NULL;
  svm_prob->y_calib     = NULL;

  svm_prob->Np_calib    = -1;
  svm_prob->Nn_calib    = -1;
  svm_prob->labels_to_target_probs = true;

  svm_prob->threshold   = .5;

  svm->ops->setup                   = SVMSetUp_Probability;
  svm->ops->reset                   = SVMReset_Probability;
  svm->ops->destroy                 = SVMDestroy_Probability;
  svm->ops->setfromoptions          = SVMSetFromOptions_Probability;
  svm->ops->train                   = SVMTrain_Probability;
  svm->ops->posttrain               = SVMPostTrain_Probability;
  svm->ops->predict                 = SVMPredict_Probability;
  svm->ops->test                    = SVMTest_Probability;
  svm->ops->view                    = SVMView_Probability;
  svm->ops->viewscore               = SVMViewScore_Probability;
  svm->ops->computemodelscores      = SVMComputeModelScores_Probability;
  svm->ops->viewtrainingpredictions = SVMViewTrainingPredictions_Probability;
  svm->ops->viewtestpredictions     = SVMViewTestPredictions_Probability;

  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetInnerSVM_C"           ,SVMGetInnerSVM_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetTao_C"                ,SVMGetTao_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetTrainingDataset_C"    ,SVMSetTrainingDataset_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetTrainingDataset_C"    ,SVMGetTrainingDataset_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetCalibrationDataset_C" ,SVMSetCalibrationDataset_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetCalibrationDataset_C" ,SVMGetCalibrationDataset_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetLabels_C"             ,SVMGetLabels_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMLoadCalibrationDataset_C",SVMLoadCalibrationDataset_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMViewCalibrationDataset_C",SVMViewCalibrationDataset_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetOptionsPrefix_C"      ,SVMSetOptionsPrefix_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetOptionsPrefix_C"      ,SVMGetOptionsPrefix_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMAppendOptionsPrefix_C"   ,SVMAppendOptionsPrefix_Probability));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMProbSetConvertLabelsToTargetProbability(SVM svm,PetscBool flg)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  if (svm_prob->labels_to_target_probs == flg) {
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  svm_prob->labels_to_target_probs = flg;
  svm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMProbGetConvertLabelsToTargetProbability(SVM svm,PetscBool *flg)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  *flg = svm_prob->labels_to_target_probs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMProbConvertProbabilityToLabels(SVM svm,Vec y)
{
  Vec            vec_threshold;
  PetscReal      threshold;

  Vec             y_sub;

  IS              is_p,is_n;
  PetscInt        lo,hi;

  const PetscReal *labels = NULL;

  PetscFunctionBegin;
  PetscCall(SVMProbGetThreshold(svm,&threshold));
  PetscCall(SVMGetLabels(svm,&labels));

  PetscCall(VecDuplicate(y,&vec_threshold));
  PetscCall(VecSet(vec_threshold,threshold));

  PetscCall(VecWhichGreaterThan(y,vec_threshold,&is_p));
  PetscCall(VecGetOwnershipRange(y,&lo,&hi));
  PetscCall(ISComplement(is_p,lo,hi,&is_n));

  PetscCall(VecGetSubVector(y,is_n,&y_sub));
  PetscCall(VecSet(y_sub,labels[0]));
  PetscCall(VecRestoreSubVector(y,is_n,&y_sub));

  PetscCall(VecGetSubVector(y,is_p,&y_sub));
  PetscCall(VecSet(y_sub,labels[1]));
  PetscCall(VecRestoreSubVector(y,is_p,&y_sub));

  /* Clean up */
  PetscCall(ISDestroy(&is_p));
  PetscCall(ISDestroy(&is_n));
  PetscCall(VecDestroy(&vec_threshold));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMProbGetSigmoidParams(SVM svm,PetscReal *A,PetscReal *B)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscFunctionBegin;
  if (A) {
    PetscValidPointer(A,1);
    *A = svm_prob->sigmoid_params[0];
  }
  if (B) {
    PetscValidPointer(B,2);
    *B = svm_prob->sigmoid_params[1];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMProbSetThreshold(SVM svm,PetscReal v)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveReal(svm,v,2);
  svm_prob->threshold = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMProbGetThreshold(SVM svm,PetscReal *v)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidRealPointer(v,2);
  *v = svm_prob->threshold;
  PetscFunctionReturn(PETSC_SUCCESS);
}
