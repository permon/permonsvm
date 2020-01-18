
#include "binaryimpl.h"

#undef __FUNCT__
#define __FUNCT__ "SVMCrossValidation_Binary"
PetscErrorCode SVMCrossValidation_Binary(SVM svm,PetscReal c_arr[],PetscInt m,PetscReal score[])
{
  SVM_Binary          *svm_binary;

  Vec                 y;
  PetscInt            p;

  CrossValidationType cv_type;
  PetscInt            nfolds;

  PetscFunctionBegin;
  /* Allocate working vectors for later warm starting of training final model */
  if (svm->warm_start) {
    svm_binary = (SVM_Binary *) svm->data;

    TRY( SVMGetTrainingDataset(svm,NULL,&y) );
    TRY( SVMGetNfolds(svm,&nfolds) );
    TRY( SVMGetPenaltyType(svm,&p) );

    /* TODO destroy when new nfold value is set in function SVMGetNfolds() */
    TRY( VecDestroyVecs(nfolds,&svm_binary->cv_best_x) );
    TRY( VecDuplicateVecs(y,nfolds,&svm_binary->cv_best_x) );
    TRY( PetscFree(svm_binary->cv_best_C) );
    TRY( PetscMalloc1(nfolds * p,&svm_binary->cv_best_C) );
  }

  TRY( SVMGetCrossValidationType(svm,&cv_type) );
  if (cv_type == CROSS_VALIDATION_KFOLD) {
    TRY( SVMKFoldCrossValidation(svm,c_arr,m,score) );
  } else {
    TRY( SVMStratifiedKFoldCrossValidation(svm,c_arr,m,score) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMKFoldCrossValidation_Binary"
PetscErrorCode SVMKFoldCrossValidation_Binary(SVM svm,PetscReal c_arr[],PetscInt m,PetscReal score[])
{
  SVM_Binary *svm_binary = (SVM_Binary *) svm->data;

  MPI_Comm          comm;
  SVM               cross_svm;

  QPS               qps;

  Mat               Xt,Xt_training,Xt_test;
  Vec               y,y_training,y_test;
  Vec               x,best_x_fold;

  PetscInt          lo,hi,first,n;
  PetscInt          i,j,k,l,nfolds;

  IS                is_training,is_test;

  PetscInt          p;       /* penalty type */
  PetscInt          svm_mod;
  SVMLossType       svm_loss;

  PetscReal         s,score_best,tmp;
  const ModelScore  *model_scores;
  PetscInt          nscores;

  PetscBool         info_set;

  const char        *prefix;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject) svm,&comm) );
  TRY( SVMGetLossType(svm,&svm_loss) );
  TRY( SVMGetMod(svm,&svm_mod) );
  TRY( SVMGetPenaltyType(svm,&p) );

  /* Create SVM used in cross validation */
  TRY( SVMCreate(comm,&cross_svm) );
  TRY( SVMSetType(cross_svm,SVM_BINARY) );

  /* Set options of SVM used in cross validation */
  TRY( SVMSetMod(cross_svm,svm_mod) );
  TRY( SVMSetLossType(cross_svm,svm_loss) );
  TRY( SVMSetPenaltyType(cross_svm,p) );
  TRY( SVMAppendOptionsPrefix(cross_svm,"cross_") );
  TRY( SVMSetFromOptions(cross_svm) );

  TRY( SVMGetOptionsPrefix(cross_svm,&prefix) );
  TRY( PetscOptionsHasName(NULL,prefix,"-svm_info",&info_set) );

  TRY( SVMGetTrainingDataset(svm,&Xt,&y) );
  TRY( MatGetOwnershipRange(Xt,&lo,&hi) );

  TRY( SVMGetHyperOptNScoreTypes(svm,&nscores) );
  TRY( SVMGetHyperOptScoreTypes(svm,&model_scores) );
  TRY( SVMGetNfolds(svm,&nfolds) );

  for (i = 0; i < nfolds; ++i) {
    if (info_set) {
      TRY( PetscPrintf(comm,"SVM: fold %d of %d\n",i+1,nfolds) );
    }

    first = lo + i - lo % nfolds;
    if (first < lo) first += nfolds;
    n = (hi + nfolds - first - 1) / nfolds;

    /* Create test dataset */
    TRY( ISCreateStride(comm,n,first,nfolds,&is_test) );
    TRY( MatCreateSubMatrix(Xt,is_test,NULL,MAT_INITIAL_MATRIX,&Xt_test) );
    TRY( VecGetSubVector(y,is_test,&y_test) );

    /* Create training dataset */
    TRY( ISComplement(is_test,lo,hi,&is_training) );
    TRY( MatCreateSubMatrix(Xt,is_training,NULL,MAT_INITIAL_MATRIX,&Xt_training) );
    TRY( VecGetSubVector(y,is_training,&y_training) );

    TRY( SVMSetTrainingDataset(cross_svm,Xt_training,y_training) );
    TRY( SVMSetTestDataset(cross_svm,Xt_test,y_test) );

    /* Test C values on fold */
    TRY( SVMGetQPS(cross_svm,&qps) );
    qps->max_it = 1e7; /* TODO method for setting max iteration of hyperopt */

    k = 0;
    score_best = 0.;   /* the best score on an actual fold */
    for (j = 0; j < m; j += p) {
      TRY( SVMSetPenalty(cross_svm,p,&c_arr[j]) );
      TRY( SVMTrain(cross_svm) );
      TRY( SVMTest(cross_svm) );
      /* Get model scores */
      tmp = 0.;
      for (l = 0; l < nscores; ++l) {
        TRY( SVMGetModelScore(cross_svm,model_scores[l],&s) );
        tmp += s;
      }
      /* Determine the best score on an actual fold */
      if (svm->warm_start && score_best < tmp) {
        TRY( SVMGetSolutionVector(cross_svm,&x) );

        TRY( VecGetSubVector(svm_binary->cv_best_x[i],is_training,&best_x_fold) );
        TRY( VecCopy(x,best_x_fold) );
        TRY( VecRestoreSubVector(svm_binary->cv_best_x[i],is_training,&best_x_fold) );

        svm_binary->cv_best_C[i] = c_arr[j];
        if (p == 2) svm_binary->cv_best_C[i + 1] = c_arr[j + 1];

        score_best = tmp;
      }

      score[k] += tmp;
      ++k;
    }
    TRY( SVMReset(cross_svm) );
    TRY( SVMSetOptionsPrefix(cross_svm,prefix) );

    TRY( VecRestoreSubVector(y,is_training,&y_training) );
    TRY( MatDestroy(&Xt_training) );
    TRY( VecRestoreSubVector(y,is_test,&y_test) );
    TRY( MatDestroy(&Xt_test) );
    TRY( ISDestroy(&is_training) );
    TRY( ISDestroy(&is_test) );
  }

  n = m / p;
  for (i = 0; i < n; ++i) score[i] /= (PetscReal) nscores * nfolds;

  TRY( SVMDestroy(&cross_svm) );
  PetscFunctionReturn(0);
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
  TRY( VecGetOwnershipRange(idx,&lo,&hi) );

  /* Determine first element of index set (local)*/
  first = lo + i - lo % nfolds;
  if (first < lo) first += nfolds;
  /* Determine length of index set (local) */
  n = (hi + nfolds - first - 1) / nfolds;

  /* Create index set for training */
  TRY( ISCreateStride(PetscObjectComm((PetscObject) idx),n,first,nfolds,&is_idx) );
  TRY( VecGetSubVector(idx,is_idx,&idx_test) );
  TRY( ISCreateFromVec(idx_test,is_test) );
  TRY( VecRestoreSubVector(idx,is_idx,&idx_test) );

  /* Create index set for test */
  TRY( ISComplement(is_idx,lo,hi,&is_idx_training) );
  TRY( VecGetSubVector(idx,is_idx_training,&idx_training) );
  TRY( ISCreateFromVec(idx_training,is_training) );
  TRY( VecRestoreSubVector(idx,is_idx_training,&idx_training) );

  /* Free memory */
  TRY( ISDestroy(&is_idx) );
  TRY( ISDestroy(&is_idx_training) );
  PetscFunctionReturn(0);
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

  Vec              x,best_x_fold;

  PetscInt         nfolds;
  PetscInt         i,j,k,l;

  PetscInt         p;
  PetscInt         svm_mod;
  SVMLossType      svm_loss;

  PetscBool        info_set;

  PetscReal        s,score_best,tmp;
  const ModelScore *model_scores;
  PetscInt         nscores;

  const char       *prefix;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject) svm,&comm) );
  TRY( SVMGetLossType(svm,&svm_loss) );
  TRY( SVMGetMod(svm,&svm_mod) );
  TRY( SVMGetPenaltyType(svm,&p) );

  /* Create SVM used in cross validation */
  TRY( SVMCreate(PetscObjectComm((PetscObject)svm),&cross_svm) );
  TRY( SVMSetType(cross_svm,SVM_BINARY) );

  /* Set options of SVM used in cross validation */
  TRY( SVMSetMod(cross_svm,svm_mod) );
  TRY( SVMSetLossType(cross_svm,svm_loss) );
  TRY( SVMSetPenaltyType(cross_svm,p) );
  TRY( SVMAppendOptionsPrefix(cross_svm,"cross_") );
  TRY( SVMSetFromOptions(cross_svm) );

  TRY( SVMGetOptionsPrefix(cross_svm,&prefix) );
  TRY( PetscOptionsHasName(NULL,prefix,"-svm_info",&info_set) );

  TRY( SVMGetTrainingDataset(svm,&Xt,&y) );
  /* Split positive and negative training samples */
  is_p = svm_binary->is_p;
  is_n = svm_binary->is_n;

  /* Create vectors that contain indices of positive and negative samples */
  TRY( VecCreateFromIS(is_p,&idx_p) );
  TRY( VecCreateFromIS(is_n,&idx_n) );

  /* Perform cross validation */
  TRY( SVMGetHyperOptNScoreTypes(svm,&nscores) );
  TRY( SVMGetHyperOptScoreTypes(svm,&model_scores) );
  TRY( SVMGetNfolds(svm,&nfolds) );
  for (i = 0; i < nfolds; ++i) {
    if (info_set) {
      TRY( PetscPrintf(comm,"SVM: fold %d of %d\n",i+1,nfolds) );
    }

    /* Fold positive samples */
    TRY( SVMFoldVecIdx_Binary_Private(idx_p,nfolds,i,&is_training_p,&is_test_p) );
    /* Fold positive samples */
    TRY( SVMFoldVecIdx_Binary_Private(idx_n,nfolds,i,&is_training_n,&is_test_n) );

    /* Union indices of positive and negative training samples */
    TRY( ISExpand(is_training_p,is_training_n,&is_training) );
    /* Union indices of positive and negative test samples */
    TRY( ISExpand(is_test_p,is_test_n,&is_test) );

    /* Create training dataset */
    TRY( MatCreateSubMatrix(Xt,is_training,NULL,MAT_INITIAL_MATRIX,&Xt_training) );
    TRY( VecGetSubVector(y,is_training,&y_training) );

    /* Create test dataset */
    TRY( MatCreateSubMatrix(Xt,is_test,NULL,MAT_INITIAL_MATRIX,&Xt_test) );
    TRY( VecGetSubVector(y,is_test,&y_test) );

    TRY( SVMSetTrainingDataset(cross_svm,Xt_training,y_training) );
    TRY( SVMSetTestDataset(cross_svm,Xt_test,y_test) );
    /* Test C values on fold */
    TRY( SVMGetQPS(cross_svm,&qps) );
    qps->max_it = 1e7;

    k = 0;
    score_best = 0.;   /* the best score on an actual fold */
    for (j = 0; j < m; j += p) {
      TRY( SVMSetPenalty(cross_svm,p,&c_arr[j]) );
      TRY( SVMTrain(cross_svm) );
      TRY( SVMTest(cross_svm) );

      /* Get model scores */
      tmp = 0.;
      for (l = 0; l < nscores; ++l) {
        TRY( SVMGetModelScore(cross_svm,model_scores[l],&s) );
        tmp += s;
      }
      /* Determine the best score on an actual fold */
      if (svm->warm_start && score_best < tmp) {
        TRY( SVMGetSolutionVector(cross_svm,&x) );

        TRY( VecGetSubVector(svm_binary->cv_best_x[i],is_training,&best_x_fold) );
        TRY( VecCopy(x,best_x_fold) );
        TRY( VecRestoreSubVector(svm_binary->cv_best_x[i],is_training,&best_x_fold) );

        svm_binary->cv_best_C[i] = c_arr[j];
        if (p == 2) svm_binary->cv_best_C[i + 1] = c_arr[j + 1];

        score_best = tmp;
      }

      score[k] += tmp;
      ++k;
    }
    TRY( SVMReset(cross_svm) );
    TRY( SVMSetOptionsPrefix(cross_svm,prefix) );

    /* Free memory */
    TRY( MatDestroy(&Xt_training) );
    TRY( MatDestroy(&Xt_test) );
    TRY( VecRestoreSubVector(y,is_training,&y_training) );
    TRY( VecRestoreSubVector(y,is_test,&y_test) );
    TRY( ISDestroy(&is_training) );
    TRY( ISDestroy(&is_test) );
    TRY( ISDestroy(&is_training_n) );
    TRY( ISDestroy(&is_training_p) );
    TRY( ISDestroy(&is_test_n) );
    TRY( ISDestroy(&is_test_p) );
  }

  k = m / p;
  for (i = 0; i < k; ++i) score[i] /= (PetscReal) nscores * nfolds;

  /* Free memory */
  TRY( VecDestroy(&idx_n) );
  TRY( VecDestroy(&idx_p) );
  TRY( SVMDestroy(&cross_svm) );
  PetscFunctionReturn(0);
}
