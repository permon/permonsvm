
#include <permon/private/svmimpl.h>

#undef __FUNCT__
#define __FUNCT__ "SVMKFoldCrossValidation_Binary"
PetscErrorCode SVMKFoldCrossValidation_Binary(SVM svm,PetscReal c_arr[],PetscInt m,PetscReal score[]) {
  MPI_Comm    comm;
  SVM         cross_svm;

  QP          qp;
  QPS         qps;

  Mat         Xt,Xt_training,Xt_test;
  Vec         x,y,y_training,y_test;

  PetscInt    lo,hi,first,n;
  PetscInt    i,j,nfolds;
  PetscReal   s;

  IS          is_training,is_test;

  PetscInt    svm_mod;
  SVMLossType svm_loss;

  ModelScore  model_score;

  PetscBool   info_set;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject) svm,&comm) );
  TRY( SVMGetLossType(svm,&svm_loss) );
  TRY( SVMGetMod(svm,&svm_mod) );

  /* Create SVM used in cross validation */
  TRY( SVMCreate(PetscObjectComm((PetscObject)svm),&cross_svm) );
  TRY( SVMSetType(cross_svm,SVM_BINARY) );

  /* Set options of SVM used in cross validation */
  TRY( SVMSetMod(cross_svm,svm_mod) );
  TRY( SVMSetLossType(cross_svm,svm_loss) );
  TRY( SVMAppendOptionsPrefix(cross_svm,"cross_") );
  TRY( SVMSetFromOptions(cross_svm) );

  TRY( PetscOptionsHasName(NULL,((PetscObject)cross_svm)->prefix,"-svm_info",&info_set) );

  TRY( SVMGetTrainingDataset(svm,&Xt,&y) );
  TRY( MatGetOwnershipRange(Xt,&lo,&hi) );

  TRY( ISCreate(PetscObjectComm((PetscObject) svm),&is_test) );
  TRY( ISSetType(is_test,ISSTRIDE) );

  TRY( SVMGetCrossValidationScoreType(svm,&model_score) );
  TRY( SVMGetNfolds(svm,&nfolds) );
  for (i = 0; i < nfolds; ++i) {
    if (info_set) {
      TRY( PetscPrintf(comm,"SVM: fold %d of %d\n",i+1,nfolds) );
    }

    first = lo + i - 1;
    if (first < lo) first += nfolds;
    n = (hi + nfolds - first - 1) / nfolds;

    /* Create test dataset */
    TRY( ISStrideSetStride(is_test,n,first,nfolds) );
    TRY( MatCreateSubMatrix(Xt,is_test,NULL,MAT_INITIAL_MATRIX,&Xt_test) );
    TRY( VecGetSubVector(y,is_test,&y_test) );

    /* Create training dataset */
    TRY( ISComplement(is_test,lo,hi,&is_training) );
    TRY( MatCreateSubMatrix(Xt,is_training,NULL,MAT_INITIAL_MATRIX,&Xt_training) );
    TRY( VecGetSubVector(y,is_training,&y_training) );

    TRY( SVMSetTrainingDataset(cross_svm,Xt_training,y_training) );
    TRY( SVMSetTestDataset(cross_svm,Xt_test,y_test) );

    for (j = 0; j < m; ++j) {
      TRY( SVMSetC(cross_svm,c_arr[j]) );

      if (!cross_svm->warm_start && j > 0) {
        TRY( SVMGetQPS(cross_svm,&qps) );
        TRY( QPSGetSolvedQP(qps,&qp) );
        if (qp) {
          Vec ub;
          TRY( QPGetSolutionVector(qp,&x) );
          TRY( VecSet(x,c_arr[j] - 100 * PETSC_MACHINE_EPSILON) );
          TRY( QPGetBox(qp,NULL,NULL,&ub) );
          if (ub) { TRY( VecSet(ub,c_arr[j]) ); }
        }
        cross_svm->posttraincalled = PETSC_FALSE;
        cross_svm->setupcalled = PETSC_TRUE;
      }

      TRY( SVMTrain(cross_svm) );
      TRY( SVMTest(cross_svm) );

      TRY( SVMGetModelScore(cross_svm,model_score,&s) );
      score[j] += s;
    }
    TRY( SVMReset(cross_svm) );
    TRY( SVMGetQPS(cross_svm,&qps) );
    TRY( QPSResetStatistics(qps) );

    TRY( VecRestoreSubVector(y,is_training,&y_training) );
    TRY( MatDestroy(&Xt_training) );
    TRY( VecRestoreSubVector(y,is_test,&y_test) );
    TRY( MatDestroy(&Xt_test) );
    TRY( ISDestroy(&is_training) );
  }

  for (i = 0; i < m; ++i) score[i] /= (PetscReal) nfolds;

  TRY( ISDestroy(&is_test) );
  TRY( SVMDestroy(&cross_svm) );
  PetscFunctionReturn(0);
}