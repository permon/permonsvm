
#include "probimpl.h"

PetscErrorCode SVMReset_Probability(SVM svm)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&svm_prob->Xt_training));
  PetscCall(VecDestroy(&svm_prob->y_training));
  PetscCall(MatDestroy(&svm_prob->Xt_calib));
  PetscCall(VecDestroy(&svm_prob->y_calib));
  PetscCall(SVMDestroy(&svm_prob->inner));

  PetscCall(PetscFree(svm_prob->deci));
  PetscCall(PetscFree(svm_prob->target));
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

  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMSetTrainingDataset_Probability(SVM svm,Mat Xt_training,Vec y_training)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  Mat      Xt_calib;
  PetscInt k,l,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(Xt_training,MAT_CLASSID,2);
  PetscCheckSameComm(svm,1,Xt_training,2);
  PetscValidHeaderSpecific(y_training,VEC_CLASSID,3);
  PetscCheckSameComm(svm,1,y_training,3);

  PetscCall(MatGetSize(Xt_training,&k,NULL));
  PetscCall(VecGetSize(y_training,&l));
  if (k != l) {
    SETERRQ(PetscObjectComm((PetscObject) Xt_training),PETSC_ERR_ARG_SIZ,"Dimensions are incompatible, X_training(%" PetscInt_FMT
                                                                         ",) != y_training(%" PetscInt_FMT ")",k,l);
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
  Vec tmp;
  IS  is_p,is_n;

  PetscInt lo,hi,n;
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

  Mat      Xt_training;

  PetscInt k,l;
  PetscInt n;

  PetscInt ntargets_lo,ntargets_hi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidHeaderSpecific(Xt_calib,MAT_CLASSID,2);
  PetscCheckSameComm(svm,1,Xt_calib,2);
  PetscValidHeaderSpecific(y_calib,VEC_CLASSID,3);
  PetscCheckSameComm(svm,1,y_calib,3);

  PetscCall(MatGetSize(Xt_calib,&k,NULL));
  PetscCall(VecGetSize(y_calib,&l));
  if (k != l) {
    SETERRQ(PetscObjectComm((PetscObject) Xt_calib),PETSC_ERR_ARG_SIZ,"Dimensions are incompatible, X_calib(%" PetscInt_FMT
                                                                      ",) != y_calib(%" PetscInt_FMT ")",k,l);
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

static PetscErrorCode SVMCreateTAO_Probability_Private(SVM svm,Tao *tao)
{
  Tao tao_inner;

  KSP ksp;
  PC  pc;

  const char *prefix = NULL;

  PetscFunctionBegin;
  PetscCall(TaoCreate(MPI_COMM_SELF,&tao_inner));
  PetscCall(TaoSetType(tao_inner,TAONLS)); // TODO possible to also set TAONTL, TAONLS, TAONTR

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

  Tao tao_inner;

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
    SVM_Probability *svm_prob = (SVM_Probability *) svm->data;
    SVM         svm_inner;

    Mat         Xt_calib;
    Vec         y_calib;

    Vec         w;
    PetscReal   b;

    Vec         Xtw,Xtw_seq;
    VecScatter  scatter;

    PetscBool   label_to_target_prob;
    PetscReal   hi_target;
    PetscReal   lo_target;

    PetscInt    i;
    PetscInt    N,Np,Nn;

    const PetscScalar *Xtw_arr = NULL;
    const PetscScalar *y_arr   = NULL;

    PetscFunctionBegin;
    PetscCall(SVMGetInnerSVM(svm,&svm_inner));
    PetscCall(SVMGetCalibrationDataset(svm,&Xt_calib,&y_calib));

    /* Get uncalibrated prediction (based on distance of sample from separating hyperplane) */
    PetscCall(SVMReconstructHyperplane(svm_inner));
    PetscCall(SVMGetSeparatingHyperplane(svm_inner,&w,&b));
    PetscCall(MatCreateVecs(Xt_calib,NULL,&Xtw));
    PetscCall(MatMult(Xt_calib,w,Xtw));

    /* Scatter Xtw to root process */
    PetscCall(VecScatterCreateToZero(Xtw,&scatter,&Xtw_seq));
    PetscCall(VecScatterBegin(scatter,Xtw,Xtw_seq,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter,Xtw,Xtw_seq,INSERT_VALUES,SCATTER_FORWARD));

    /* Clean up previous helper arrays */
    PetscCall(PetscFree(svm_prob->deci));
    PetscCall(PetscFree(svm_prob->target));

    /* Alloc helper arrays again (distance sample from hyperplane and target) */
    PetscCall(VecGetSize(Xtw_seq,&N));
    PetscCall(PetscMalloc(N * sizeof(PetscReal),&svm_prob->deci));
    PetscCall(PetscMalloc(N * sizeof(PetscReal),&svm_prob->target));

    PetscCall(SVMProbGetConvertLabelsToTargetProbability(svm,&label_to_target_prob));
    if (label_to_target_prob) {
        /*
         Transform labels to target probabilities proposed in
         http://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf
        */
        Nn = svm_prob->Nn_calib;
        Np = svm_prob->Np_calib;

        PetscCall(SVMGetNumberPositiveNegativeSamples_Probability_Private(y_calib,&Np,&Nn));
        lo_target = 1. / (Nn + 2);
        hi_target = (Np + 1.) / (Np + 2.);
    } else {
        lo_target = 0.;
        hi_target = 1.;
    }

    /* Transform labels to target probabilities */
    PetscCall(VecGetArrayRead(Xtw_seq,&Xtw_arr));
    PetscCall(VecGetArrayRead(y_calib,&y_arr));

    for (i = 0; i < N; ++i) {
        svm_prob->deci[i]   = Xtw_arr[i]; // distance from hyperplane
        svm_prob->target[i] = (y_arr[i] == -1) ? lo_target : hi_target; // labels converted to probability
    }

    PetscCall(VecRestoreArrayRead(Xtw_seq,&Xtw_arr));
    PetscCall(VecRestoreArrayRead(Xtw_seq,&y_arr));

    /* Clean up */
    PetscCall(SVMDestroy(&svm_inner));
    PetscCall(VecScatterDestroy(&scatter));
    PetscCall(VecDestroy(&Xtw));
    PetscCall(VecDestroy(&Xtw_seq));

    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoFormFunctionGradient_Probability_Private(Tao tao,Vec x,PetscReal *fnc,Vec g,void *ctx)
{
  SVM             svm   = (SVM) ctx;
  SVM_Probability *prob = (SVM_Probability *) svm->data;

  PetscReal fnc_inner = 0.;

  PetscReal g_arr[2] = {0.,0.};
  PetscInt  idx[2] = {0,1};

  PetscReal ApB;
  PetscReal p;

  PetscInt  i,N;

  const PetscReal *x_arr = NULL;
  Vec   y;

  PetscFunctionBegin;

  if (!prob->deci && !prob->target) {
    PetscCall(SVMTransformUncalibratedPredictions_Probability_Private(svm));
  }

  PetscCall(SVMGetCalibrationDataset(svm,NULL,&y));
  PetscCall(VecGetSize(y,&N));
  PetscCall(VecGetArrayRead(x,&x_arr));

  /*
    Gradient and objective function evaluation. Implementation proposed in https://www.csie.ntu.edu.tw/~cjlin/papers/plattprob.pdf.
   */
  for (i = 0; i < N; ++i) {

    ApB = x_arr[0] * prob->deci[i] + x_arr[1];

    if (ApB >= 0.) {
      fnc_inner += prob->target[i] * ApB + PetscLogReal(1. + PetscExpReal(-ApB));
      p = PetscExpReal(-ApB) / (1. + PetscExpReal(-ApB));
    } else {
      fnc_inner += (prob->target[i] - 1.) * ApB + PetscLogReal(1. + PetscExpReal(ApB));
      p = 1. / (1. + PetscExpReal(ApB) );
    }

    g_arr[0] += prob->deci[i] * (prob->target[i] - p);
    g_arr[1] += prob->target[i] - p;
  }

  PetscCall(VecRestoreArrayRead(x,&x_arr));

  *fnc = fnc_inner;

  /* update gradient */
  PetscCall(VecSetValues(g,2,idx,g_arr,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(g));
  PetscCall(VecAssemblyEnd(g));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoFormHessian_Probability_Private(Tao tao,Vec x,Mat H,Mat Hpre,void *ctx)
{
  SVM             svm   = (SVM) ctx;
  SVM_Probability *prob = (SVM_Probability *) svm->data;

  PetscReal       H_arr[4] = {0.,0.,0.,0.};
  PetscInt        idxm[2]  = {0,1};
  PetscInt        idxn[2]  = {0,1};

  PetscReal       d,p,q;
  PetscReal       AdpB;

  PetscInt        i,N;

  const PetscReal *x_arr = NULL;
  Vec   y;

  PetscFunctionBegin;

  if (!prob->deci && !prob->target) {
      PetscCall(SVMTransformUncalibratedPredictions_Probability_Private(svm));
  }

  PetscCall(SVMGetCalibrationDataset(svm,NULL,&y));
  PetscCall(VecGetSize(y,&N));
  PetscCall(VecGetArrayRead(x,&x_arr));

  /*
    Hessian evaluation. Implementation proposed in https://www.csie.ntu.edu.tw/~cjlin/papers/plattprob.pdf.
   */
  for (i = 0; i < N; ++i) {

    AdpB = prob->deci[i] * x_arr[0] + x_arr[1];

    if (AdpB >= 0.) {
      p = PetscExpReal(-AdpB) / (1. + PetscExpReal(-AdpB));
      q =  1. / (1. + PetscExpReal(-AdpB));
    } else {
      p = 1. / (1. + PetscExpReal(AdpB));
      q =  PetscExpReal(AdpB) / (1. + PetscExpReal(AdpB));
    }

    d = p * q;

    H_arr[0] += prob->deci[i] * prob->deci[i] * d;
    H_arr[1] += prob->deci[i] * d;
    H_arr[2] += prob->deci[i] * d;
    H_arr[3] += d;
  }

  PetscCall(VecRestoreArrayRead(x,&x_arr));

  H_arr[0] += 1e-12;
  H_arr[3] += 1e-12;

  /* Update Hessian */
  PetscCall(MatSetValues(H,2,idxm,2,idxn,H_arr,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SVMSetUp_Tao_Private(SVM svm)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;
  Tao tao;

  Mat H;
  Vec x_init;
  Vec g;

  PetscInt  Nn,Np;
  PetscReal v;

  PetscFunctionBegin;

  /* Create Hessian */
  PetscCall(MatCreateSeqDense(MPI_COMM_SELF,2,2,NULL,&H));
  PetscCall(VecCreateSeq(MPI_COMM_SELF,2,&g));
  PetscCall(VecCreateSeq(MPI_COMM_SELF,2,&x_init));

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
  MPI_Comm comm;

  SVM      svm_binary;

  Mat      Xt_training;
  Vec      y_training;

  Mat      Xt_calib;
  Vec      y_calib;

  PetscMPIInt rank;

  PetscFunctionBegin;
  if (svm->setupcalled) PetscFunctionReturn(0);

  PetscCall(SVMGetInnerSVM(svm,&svm_binary));
  if (svm->setfromoptionscalled) {
    PetscCall(SVMSetFromOptions(svm_binary));
  }

  PetscCall(SVMGetCalibrationDataset(svm,&Xt_calib,&y_calib));
  if (Xt_calib != NULL) {
    PetscCall(SVMGetTrainingDataset(svm,&Xt_training,&y_training));
  } else {
    // TODO split training data set into training and calibration
    Xt_training = Xt_calib;
    y_training  = y_calib;
  }

  PetscCall(SVMSetTrainingDataset(svm_binary,Xt_training,y_training));
  PetscCall(SVMSetUp(svm_binary));

  PetscCall(PetscObjectGetComm((PetscObject)svm,&comm));
  MPI_Comm_rank(comm,&rank);
  // Set up TAO solver for master
  if (rank == 0) {
    PetscCall(SVMSetUp_Tao_Private(svm));
  }

  /* Clean up */
  PetscCall(SVMDestroy(&svm_binary));

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

  PetscBool to_target_probs;
  PetscBool flg;

  PetscFunctionBegin;
  PetscObjectOptionsBegin((PetscObject)svm);
  PetscCall(PetscOptionsBool("-svm_convert_labels_to_target_probs",
                             "Convert sample labels to target probability as suggested by Platt",
                             "SVMProbSetConvertLabelsToTargetProbability",
                             svm_prob->labels_to_target_probs,&to_target_probs,&flg));
  if (flg) {
    PetscCall(SVMProbSetConvertLabelsToTargetProbability(svm,to_target_probs));
  }
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMTrain_Probability(SVM svm)
{
  SVM svm_binary;
  Tao tao;

  PetscBool post_train;

  PetscFunctionBegin;
  PetscCall(SVMSetUp(svm));
  PetscCall(SVMGetInnerSVM(svm,&svm_binary));

  // Train uncalibrated svm model
  PetscCall(SVMGetAutoPostTrain(svm_binary,&post_train));
  PetscCall(SVMTrain(svm_binary));

  // Train logistic regression over uncalibrated model (Platt's scaling)
  PetscCall(SVMTransformUncalibratedPredictions_Probability_Private(svm));
  PetscCall(SVMGetTao(svm,&tao));
  PetscCall(TaoSolve(tao));
  // Run post-processing (training) of a trained probability model
  PetscCall(SVMGetAutoPostTrain(svm,&post_train));
  if (post_train) {
    PetscCall(SVMPostTrain(svm));
  }

  /* Clean up */
  PetscCall(SVMDestroy(&svm_binary));
  PetscCall(TaoDestroy(&tao));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMPostTrain_Probability(SVM svm)
{

  PetscFunctionBegin;
  // TODO implement
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMPredict_Probability(SVM svm,Mat Xt_pred,Vec *y_out)
{

  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMComputeModelScores_Probability(SVM svm,Vec y_pred,Vec y_known)
{

  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMTest_Probability(SVM svm)
{

  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMGetInnerSVM_Probability(SVM svm,SVM *inner_out)
{
  SVM_Probability *svm_prob = (SVM_Probability *) svm->data;

  MPI_Comm   comm;
  const char *prefix = NULL;

  SVM        inner;

  PetscFunctionBegin;
  if (!svm_prob->inner) {
    PetscCall(PetscObjectGetComm((PetscObject) svm,&comm));
    PetscCall(SVMCreate(comm,&inner));
    PetscCall(SVMSetType(inner,SVM_BINARY));

    // Set prefix
    PetscCall(SVMSetOptionsPrefix(inner,"prob_binary_"));
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

PetscErrorCode SVMCreate_Probability(SVM svm)
{
  SVM_Probability *svm_prob;

  PetscFunctionBegin;
  PetscCall(PetscNew(&svm_prob));
  svm->data = (void *) svm_prob;

  svm_prob->inner       = NULL;
  svm_prob->tao         = NULL;

  svm_prob->Xt_training = NULL;
  svm_prob->y_training  = NULL;
  svm_prob->Xt_calib    = NULL;
  svm_prob->y_calib     = NULL;

  svm_prob->deci        = NULL;
  svm_prob->target      = NULL;

  svm_prob->Np_calib    = -1;
  svm_prob->Nn_calib    = -1;
  svm_prob->labels_to_target_probs = true;

  svm->ops->setup              = SVMSetUp_Probability;
  svm->ops->reset              = SVMReset_Probability;
  svm->ops->destroy            = SVMDestroy_Probability;
  svm->ops->setfromoptions     = SVMSetFromOptions_Probability;
  svm->ops->train              = SVMTrain_Probability;
  svm->ops->posttrain          = SVMPostTrain_Probability;
  svm->ops->predict            = SVMPredict_Probability;
  svm->ops->test               = SVMTest_Probability;
  svm->ops->view               = SVMView_Probability;
  svm->ops->viewscore          = SVMViewScore_Probability;
  svm->ops->computemodelscores = SVMComputeModelScores_Probability;

  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetInnerSVM_C"           ,SVMGetInnerSVM_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetTao_C"                ,SVMGetTao_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetTrainingDataset_C"    ,SVMSetTrainingDataset_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetTrainingDataset_C"    ,SVMGetTrainingDataset_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMSetCalibrationDataset_C" ,SVMSetCalibrationDataset_Probability));
  PetscCall(PetscObjectComposeFunction((PetscObject) svm,"SVMGetCalibrationDataset_C" ,SVMGetCalibrationDataset_Probability));
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