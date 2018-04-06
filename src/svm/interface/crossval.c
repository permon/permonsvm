
#include <permon/private/svmimpl.h>

#undef __FUNCT__
#define __FUNCT__ "PermonSVMSetWarmStart"
/*@
   PermonSVMSetWarmStart - Set flag specifying whether warm start is used in cross-validation.
   It is set to PETSC_TRUE by default.

   Logically Collective on PermonSVM

   Input Parameters:
+  svm - the SVM
-  flg - use warm start in cross-validation

   Options Database Keys:
.  -svm_warm_start - use warm start in cross-validation

   Level: advanced

.seealso PermonSVMType, PermonSVMGetLossType()
@*/
PetscErrorCode PermonSVMSetWarmStart(PermonSVM svm, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidLogicalCollectiveBool(svm,flg,2);
  svm->warm_start = flg;
  PetscFunctionReturn(0);
}

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
  PetscMPIInt rank;
  PermonSVM cross_svm;
  IS is_test, is_train;
  PetscInt n_examples, n_attributes;  /* PETSC_DEFAULT or PETSC_DECIDE means all */
  PetscInt i, j, i_max;
  PetscInt nfolds, first, n;
  PetscInt lo, hi;
  PetscInt N_all, N_eq, c_count;
  PetscInt its;
  PetscReal C, LogCMin, LogCBase, LogCMax, C_min;
  PetscReal *array_rate = NULL, rate_max, rate;
  PetscReal *array_C = NULL;
  Mat Xt, Xt_test, Xt_train;
  Vec y, y_test, y_train;
  const char *prefix;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( PetscObjectGetComm((PetscObject)svm,&comm) );
  TRY( MPI_Comm_rank(comm, &rank) );
  TRY( PermonSVMGetTrainingSamples(svm,&Xt,&y) );
  TRY( MatGetSize(Xt,&n_examples,&n_attributes) );
  TRY( MatGetOwnershipRange(Xt,&lo,&hi) );
  TRY( PermonSVMGetLogCMin(svm, &LogCMin) );
  TRY( PermonSVMGetLogCBase(svm, &LogCBase) );
  TRY( PermonSVMGetLogCMax(svm, &LogCMax) );
  TRY( PermonSVMGetNfolds(svm, &nfolds) );
  TRY( PermonSVMGetOptionsPrefix(svm, &prefix) );

  if (nfolds > n_examples) FLLOP_SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"number of folds must not be greater than number of examples but %d > %d",nfolds,n_examples);

  C_min = PetscPowReal(LogCBase,LogCMin);
  c_count = LogCMax - LogCMin + 1;
  i=0;
  if (!c_count) FLLOP_SETERRQ(comm,PETSC_ERR_ARG_INCOMP,"LogCMin must be less than or equal to LogCMax");

  TRY( PetscMalloc1(c_count,&array_rate) );
  TRY( PetscMalloc1(c_count,&array_C) );
  TRY( PetscMemzero(array_rate,c_count*sizeof(PetscReal)) );
  array_C[0] = C_min;
  for (i = 1; i < c_count; i++) {
    array_C[i] = array_C[i-1] * LogCBase;
  }
  TRY( PetscPrintf(comm, "### PermonSVM: following values of C will be tested:\n") );
  if (!rank) TRY( PetscRealView(c_count,array_C,PETSC_VIEWER_STDOUT_SELF) );

  TRY( ISCreate(PetscObjectComm((PetscObject)svm),&is_test) );
  TRY( ISSetType(is_test,ISSTRIDE) );

  for (i = 0; i < nfolds; ++i) {
    TRY( PetscPrintf(comm, "### PermonSVM: fold %d of %d\n",i+1,nfolds) );

    first = (lo-1)/nfolds*nfolds + i;
    if (first < lo) first += nfolds;
    n = (hi + nfolds - first - 1)/nfolds;

    TRY( ISStrideSetStride(is_test,n,first,nfolds) );
    TRY( MatCreateSubMatrix(Xt,is_test,NULL,MAT_INITIAL_MATRIX,&Xt_test) );
    TRY( VecGetSubVector(y,is_test,&y_test) );
    TRY( ISComplement(is_test,lo,hi,&is_train) );
    TRY( MatCreateSubMatrix(Xt,is_train,NULL,MAT_INITIAL_MATRIX,&Xt_train) );
    TRY( VecGetSubVector(y,is_train,&y_train) );

    TRY( PermonSVMCreate(PetscObjectComm((PetscObject)svm),&cross_svm) );
    TRY( PermonSVMSetOptionsPrefix(cross_svm,prefix) );
    TRY( PermonSVMAppendOptionsPrefix(cross_svm,"cross_") );
    TRY( PermonSVMSetTrainingSamples(cross_svm,Xt_train,y_train) );
    TRY( PermonSVMSetLossType(cross_svm,svm->loss_type) );
    TRY( PermonSVMSetFromOptions(cross_svm) );
    for (j = 0; j < c_count; ++j) {
      TRY( PermonSVMSetC(cross_svm,array_C[j]) );

      if (!svm->warm_start) {
        QP qp;
        Vec x;
        TRY( QPSGetSolvedQP(cross_svm->qps,&qp) );
        if (qp) {
          TRY( QPGetSolutionVector(qp,&x) );
          TRY( VecZeroEntries(x) );
        }
      }

      TRY( PermonSVMTrain(cross_svm) );
      TRY( PermonSVMTest(cross_svm,Xt_test,y_test,&N_all,&N_eq) );
      rate = ((PetscReal)N_eq) / ((PetscReal)N_all);
      array_rate[j] += rate;

      //TRY( PetscObjectTypeCompare((PetscObject)cross_svm->qps,QPSSMALXE,&flg) );
      /*if (flg) {
        QPS qps_inner;
        TRY( QPSSMALXEGetInnerQPS(cross_svm->qps,&qps_inner) );
        TRY( QPSGetAccumulatedIterationNumber(qps_inner,&its) );
      } else {*/
      TRY( QPSGetIterationNumber(cross_svm->qps,&its) );
      //}
      TRY( PetscPrintf(comm, "### PermonSVM:     %d of %d examples classified correctly (rate %.2f), accumulated rate for C=%.2e is %.2f, %d QPS iterations\n", N_eq, N_all, rate, array_C[j], array_rate[j], its) );
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
  TRY( PermonSVMSetC(svm, C) );

  TRY( ISDestroy(&is_test) );
  TRY( PetscFree(array_rate) );
  TRY( PetscFree(array_C) );
  PetscFunctionReturnI(0);
}
