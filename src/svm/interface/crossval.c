
#include <permon/private/svmimpl.h>

#undef __FUNCT__
#define __FUNCT__ "PermonSVMCrossValidate"
/*@
   PermonSVMCrossValidate -

   Input Parameter:
.  svm - the SVM

@*/
PetscErrorCode PermonSVMCrossValidate(PermonSVM svm)
{
  MPI_Comm comm;
  PermonSVM cross_svm;
  IS is_test, is_train;
  PetscInt n_examples, n_attributes;  /* PETSC_DEFAULT or PETSC_DECIDE means all */
  PetscInt i, j, i_max;
  PetscInt nfolds, first, n;
  PetscInt lo, hi;
  PetscInt N_all, N_eq, c_count;
  PetscReal C, C_min, C_step, C_max, C_i;
  PetscReal *array_rate = NULL, rate_max, rate;
  PetscReal *array_C = NULL;
  Mat Xt, Xt_test, Xt_train;
  Vec y, y_test, y_train;
  const char *prefix;

  PetscFunctionBeginI;
  TRY( PetscObjectGetComm((PetscObject)svm,&comm) );
  TRY( PermonSVMGetTrainingSamples(svm,&Xt,&y) );
  TRY( MatGetSize(Xt,&n_examples,&n_attributes) );
  TRY( MatGetOwnershipRange(Xt,&lo,&hi) );
  TRY( PermonSVMGetPenaltyMin(svm, &C_min) );
  TRY( PermonSVMGetPenaltyStep(svm, &C_step) );
  TRY( PermonSVMGetPenaltyMax(svm, &C_max) );
  TRY( PermonSVMGetNfolds(svm, &nfolds) );
  TRY( PermonSVMGetOptionsPrefix(svm, &prefix) );

  if (nfolds > n_examples) FLLOP_SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"number of folds must not be greater than number of examples but %d > %d",nfolds,n_examples);

  c_count = 0;
  for (C_i = C_min; C_i <= C_max; C_i*=C_step) c_count++;
  if (!c_count) FLLOP_SETERRQ(comm,PETSC_ERR_ARG_INCOMP,"PenaltyMin must be less than PenaltyMax");

  TRY( PetscMalloc1(c_count,&array_rate) );
  TRY( PetscMalloc1(c_count,&array_C) );
  TRY( PetscMemzero(array_rate,c_count*sizeof(PetscReal)) );
  array_C[0] = C_min;
  for (i = 1; i < c_count; i++) {
    array_C[i] = array_C[i-1] * C_step;
  }
  TRY( PetscPrintf(comm, "### PermonSVM: following values of penalty C will be tested:\n") );
  TRY( PetscRealView(c_count,array_C,PETSC_VIEWER_STDOUT_(comm)) );

  TRY( ISCreate(PetscObjectComm((PetscObject)svm),&is_test) );
  TRY( ISSetType(is_test,ISSTRIDE) );

  for (i = 0; i < nfolds; ++i) {
    TRY( PetscPrintf(comm, "fold %d of %d\n",i+1,nfolds) );

    first = (lo-1)/nfolds*nfolds + i;
    if (first < lo) first += nfolds;
    n = (hi + nfolds - first - 1)/nfolds;

    TRY( ISStrideSetStride(is_test,n,first,nfolds) );
    TRY( MatGetSubMatrix(Xt,is_test,NULL,MAT_INITIAL_MATRIX,&Xt_test) );
    TRY( VecGetSubVector(y,is_test,&y_test) );
    TRY( ISComplement(is_test,lo,hi,&is_train) );
    TRY( MatGetSubMatrix(Xt,is_train,NULL,MAT_INITIAL_MATRIX,&Xt_train) );
    TRY( VecGetSubVector(y,is_train,&y_train) );

    TRY( PermonSVMCreate(PetscObjectComm((PetscObject)svm),&cross_svm) );
    TRY( PermonSVMSetOptionsPrefix(cross_svm,prefix) );
    TRY( PermonSVMAppendOptionsPrefix(cross_svm,"cross_") );
    TRY( PermonSVMSetTrainingSamples(cross_svm,Xt_train,y_train) );
    TRY( PermonSVMSetFromOptions(cross_svm) );
    for (j = 0; j < c_count; ++j) {
      TRY( PetscPrintf(comm, "  C[%d] = %.2e\n", j, array_C[j]) );
      TRY( PermonSVMSetPenalty(cross_svm,array_C[j]) );
      TRY( PermonSVMTrain(cross_svm) );
      TRY( PermonSVMTest(cross_svm,Xt_test,y_test,&N_all,&N_eq) );
      rate = ((PetscReal)N_eq) / ((PetscReal)N_all);
      array_rate[j] += rate;
      TRY( PetscPrintf(comm, "    N_all = %d, N_eq = %d, rate = %f, rate_acc = %f\n", N_all, N_eq, rate, array_rate[j]) );
    }
    TRY( PermonSVMDestroy(&cross_svm) );
    TRY( MatDestroy(&Xt_test) );
    TRY( VecRestoreSubVector(y,is_test,&y_test) );
    TRY( MatDestroy(&Xt_train) );
    TRY( VecRestoreSubVector(y,is_train,&y_train) );
    TRY( ISDestroy(&is_train) );
  }

  i_max = 0;
  for(i = 1; i < c_count; ++i) {
    if (array_rate[i] > array_rate[i_max]) {
      i_max = i;
    }
  }
  rate_max = array_rate[i_max] / nfolds;
  C = array_C[i_max];
  TRY( PetscPrintf(comm,"### PermonSVM: selecting C = %.2e with accumulated rate %f and average rate %f based on cross-validation with %d folds\n",C,array_rate[i_max],rate_max,nfolds) );
  TRY( PermonSVMSetPenalty(svm, C) );

  TRY( ISDestroy(&is_test) );
  TRY( PetscFree(array_rate) );
  TRY( PetscFree(array_C) );
  PetscFunctionReturnI(0);
}
