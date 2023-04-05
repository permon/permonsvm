
#include "binaryimpl.h"

#undef __FUNCT__
#define __FUNCT__ "SVMCrossValidation_Binary"
PetscErrorCode SVMCrossValidation_Binary(SVM svm,PetscReal c_arr[],PetscInt m,PetscReal score[])
{
  CrossValidationType cv_type;

  PetscFunctionBegin;
  PetscCall(SVMGetCrossValidationType(svm,&cv_type));
  if (cv_type == CROSS_VALIDATION_KFOLD) {
    PetscCall(SVMKFoldCrossValidation(svm,c_arr,m,score));
  } else {
    PetscCall(SVMStratifiedKFoldCrossValidation(svm,c_arr,m,score));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMKFoldCrossValidation_Binary"
PetscErrorCode SVMKFoldCrossValidation_Binary(SVM svm,PetscReal c_arr[],PetscInt m,PetscReal score[])
{
  MPI_Comm          comm;
  SVM               cross_svm;

  QPS               qps;

  Mat               Xt,Xt_training,Xt_test;
  Vec               y,y_training,y_test;

  PetscInt          lo,hi,first,n;
  PetscInt          i,j,k,l,nfolds;

  IS                is_training,is_test;

  PetscInt          p;       /* penalty type */
  PetscInt          svm_mod;
  SVMLossType       svm_loss;

  PetscReal         s;
  const ModelScore  *model_scores;
  PetscInt          nscores;

  PetscBool         info_set;

  const char        *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) svm,&comm));
  PetscCall(SVMGetLossType(svm,&svm_loss));
  PetscCall(SVMGetMod(svm,&svm_mod));
  PetscCall(SVMGetPenaltyType(svm,&p));

  /* Create SVM used in cross validation */
  PetscCall(SVMCreate(comm,&cross_svm));
  PetscCall(SVMSetType(cross_svm,SVM_BINARY));

  /* Set options of SVM used in cross validation */
  PetscCall(SVMSetMod(cross_svm,svm_mod));
  PetscCall(SVMSetLossType(cross_svm,svm_loss));
  PetscCall(SVMSetPenaltyType(cross_svm,p));
  PetscCall(SVMAppendOptionsPrefix(cross_svm,"cross_"));
  PetscCall(SVMSetFromOptions(cross_svm));

  PetscCall(SVMGetOptionsPrefix(cross_svm,&prefix));
  PetscCall(PetscOptionsHasName(NULL,prefix,"-svm_info",&info_set));

  PetscCall(SVMGetTrainingDataset(svm,&Xt,&y));
  PetscCall(MatGetOwnershipRange(Xt,&lo,&hi));

  PetscCall(SVMGetHyperOptNScoreTypes(svm,&nscores));
  PetscCall(SVMGetHyperOptScoreTypes(svm,&model_scores));
  PetscCall(SVMGetNfolds(svm,&nfolds));
  for (i = 0; i < nfolds; ++i) {
    if (info_set) {
      PetscCall(PetscPrintf(comm,"SVM: fold %" PetscInt_FMT " of %" PetscInt_FMT "\n",i+1,nfolds));
    }

    first = lo + i - lo % nfolds;
    if (first < lo) first += nfolds;
    n = (hi + nfolds - first - 1) / nfolds;

    /* Create test dataset */
    PetscCall(ISCreateStride(comm,n,first,nfolds,&is_test));
    PetscCall(MatCreateSubMatrix(Xt,is_test,NULL,MAT_INITIAL_MATRIX,&Xt_test));
    PetscCall(VecGetSubVector(y,is_test,&y_test));

    /* Create training dataset */
    PetscCall(ISComplement(is_test,lo,hi,&is_training));
    PetscCall(MatCreateSubMatrix(Xt,is_training,NULL,MAT_INITIAL_MATRIX,&Xt_training));
    PetscCall(VecGetSubVector(y,is_training,&y_training));

    PetscCall(SVMSetTrainingDataset(cross_svm,Xt_training,y_training));
    PetscCall(SVMSetTestDataset(cross_svm,Xt_test,y_test));

    /* Test C values on fold */
    PetscCall(SVMGetQPS(cross_svm,&qps));
    qps->max_it = 1e7;
    k = 0;
    for (j = 0; j < m; j += p) {
      PetscCall(SVMSetPenalty(cross_svm,p,&c_arr[j]));
      PetscCall(SVMTrain(cross_svm));
      PetscCall(SVMTest(cross_svm));
      for (l = 0; l < nscores; ++l) {
        PetscCall(SVMGetModelScore(cross_svm,model_scores[l],&s));
        score[k] += s;
      }
      ++k;
    }
    PetscCall(SVMReset(cross_svm));
    PetscCall(SVMSetOptionsPrefix(cross_svm,prefix));

    PetscCall(VecRestoreSubVector(y,is_training,&y_training));
    PetscCall(MatDestroy(&Xt_training));
    PetscCall(VecRestoreSubVector(y,is_test,&y_test));
    PetscCall(MatDestroy(&Xt_test));
    PetscCall(ISDestroy(&is_training));
    PetscCall(ISDestroy(&is_test));
  }

  n = m / p;
  for (i = 0; i < n; ++i) score[i] /= (PetscReal) nscores * nfolds;

  PetscCall(SVMDestroy(&cross_svm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMFoldVecIdx_Binary_Private"
static PetscErrorCode SVMFoldVecIdx_Binary_Private(Vec idx,PetscInt nfolds,PetscInt i,IS *is_training,IS *is_test)
{
  PetscInt lo,hi;
  PetscInt first,n;

  IS       is_idx,is_idx_training;
  Vec      idx_training,idx_test;

  PetscFunctionBegin;
  PetscCall(VecGetOwnershipRange(idx,&lo,&hi));

  /* Determine first element of index set (local)*/
  first = lo + i - lo % nfolds;
  if (first < lo) first += nfolds;
  /* Determine length of index set (local) */
  n = (hi + nfolds - first - 1) / nfolds;

  /* Create index set for training */
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject) idx),n,first,nfolds,&is_idx));
  PetscCall(VecGetSubVector(idx,is_idx,&idx_test));
  PetscCall(ISCreateFromVec(idx_test,is_test));
  PetscCall(VecRestoreSubVector(idx,is_idx,&idx_test));

  /* Create index set for test */
  PetscCall(ISComplement(is_idx,lo,hi,&is_idx_training));
  PetscCall(VecGetSubVector(idx,is_idx_training,&idx_training));
  PetscCall(ISCreateFromVec(idx_training,is_training));
  PetscCall(VecRestoreSubVector(idx,is_idx_training,&idx_training));

  /* Free memory */
  PetscCall(ISDestroy(&is_idx));
  PetscCall(ISDestroy(&is_idx_training));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMStratifiedKFoldCrossValidation_Binary"
PetscErrorCode SVMStratifiedKFoldCrossValidation_Binary(SVM svm,PetscReal c_arr[],PetscInt m,PetscReal score[])
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  MPI_Comm         comm;
  SVM              cross_svm;

  QPS              qps;

  Mat              Xt,Xt_training,Xt_test;
  Vec              y,y_training,y_test;
  IS               is_training,is_test;

  IS               is_p,is_n;
  Vec              idx_p,idx_n;

  IS               is_training_p,is_test_p;
  IS               is_training_n,is_test_n;

  PetscInt         nfolds;
  PetscInt         i,j,k,l;

  PetscInt         p;
  PetscInt         svm_mod;
  SVMLossType      svm_loss;

  PetscBool        info_set;

  PetscReal        s;
  const ModelScore *model_scores;
  PetscInt         nscores;

  const char       *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) svm,&comm));
  PetscCall(SVMGetLossType(svm,&svm_loss));
  PetscCall(SVMGetMod(svm,&svm_mod));
  PetscCall(SVMGetPenaltyType(svm,&p));

  /* Create SVM used in cross validation */
  PetscCall(SVMCreate(PetscObjectComm((PetscObject)svm),&cross_svm));
  PetscCall(SVMSetType(cross_svm,SVM_BINARY));

  /* Set options of SVM used in cross validation */
  PetscCall(SVMSetMod(cross_svm,svm_mod));
  PetscCall(SVMSetLossType(cross_svm,svm_loss));
  PetscCall(SVMSetPenaltyType(cross_svm,p));
  PetscCall(SVMAppendOptionsPrefix(cross_svm,"cross_"));
  PetscCall(SVMSetFromOptions(cross_svm));

  PetscCall(SVMGetOptionsPrefix(cross_svm,&prefix));
  PetscCall(PetscOptionsHasName(NULL,prefix,"-svm_info",&info_set));

  PetscCall(SVMGetTrainingDataset(svm,&Xt,&y));
  /* Split positive and negative training samples */
  is_p = svm_binary->is_p;
  is_n = svm_binary->is_n;

  /* Create vectors that contain indices of positive and negative samples */
  PetscCall(VecCreateFromIS(is_p,&idx_p));
  PetscCall(VecCreateFromIS(is_n,&idx_n));

  /* Perform cross validation */
  PetscCall(SVMGetHyperOptNScoreTypes(svm,&nscores));
  PetscCall(SVMGetHyperOptScoreTypes(svm,&model_scores));
  PetscCall(SVMGetNfolds(svm,&nfolds));
  for (i = 0; i < nfolds; ++i) {
    if (info_set) {
      PetscCall(PetscPrintf(comm,"SVM: fold %" PetscInt_FMT " of %" PetscInt_FMT "\n",i+1,nfolds));
    }

    /* Fold positive samples */
    PetscCall(SVMFoldVecIdx_Binary_Private(idx_p,nfolds,i,&is_training_p,&is_test_p));
    /* Fold positive samples */
    PetscCall(SVMFoldVecIdx_Binary_Private(idx_n,nfolds,i,&is_training_n,&is_test_n));

    /* Union indices of positive and negative training samples */
    PetscCall(ISExpand(is_training_p,is_training_n,&is_training));
    /* Union indices of positive and negative test samples */
    PetscCall(ISExpand(is_test_p,is_test_n,&is_test));

    /* Create training dataset */
    PetscCall(MatCreateSubMatrix(Xt,is_training,NULL,MAT_INITIAL_MATRIX,&Xt_training));
    PetscCall(VecGetSubVector(y,is_training,&y_training));

    /* Create test dataset */
    PetscCall(MatCreateSubMatrix(Xt,is_test,NULL,MAT_INITIAL_MATRIX,&Xt_test));
    PetscCall(VecGetSubVector(y,is_test,&y_test));

    PetscCall(SVMSetTrainingDataset(cross_svm,Xt_training,y_training));
    PetscCall(SVMSetTestDataset(cross_svm,Xt_test,y_test));
    /* Test C values on fold */
    PetscCall(SVMGetQPS(cross_svm,&qps));
    qps->max_it = 1e7;
    k = 0;
    for (j = 0; j < m; j += p) {
      PetscCall(SVMSetPenalty(cross_svm,p,&c_arr[j]));
      PetscCall(SVMTrain(cross_svm));
      PetscCall(SVMTest(cross_svm));
      /* Get model scores */
      for (l = 0; l < nscores; ++l) {
        PetscCall(SVMGetModelScore(cross_svm,model_scores[l],&s));
        score[k] += s;
      }
      ++k;
    }
    PetscCall(SVMReset(cross_svm));
    PetscCall(SVMSetOptionsPrefix(cross_svm,prefix));

    /* Free memory */
    PetscCall(MatDestroy(&Xt_training));
    PetscCall(MatDestroy(&Xt_test));
    PetscCall(VecRestoreSubVector(y,is_training,&y_training));
    PetscCall(VecRestoreSubVector(y,is_test,&y_test));
    PetscCall(ISDestroy(&is_training));
    PetscCall(ISDestroy(&is_test));
    PetscCall(ISDestroy(&is_training_n));
    PetscCall(ISDestroy(&is_training_p));
    PetscCall(ISDestroy(&is_test_n));
    PetscCall(ISDestroy(&is_test_p));
  }

  k = m / p;
  for (i = 0; i < k; ++i) score[i] /= (PetscReal) nscores * nfolds;

  /* Free memory */
  PetscCall(VecDestroy(&idx_n));
  PetscCall(VecDestroy(&idx_p));
  PetscCall(SVMDestroy(&cross_svm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
