#include <mpi.h>
#include <permonsvm.h>
#include <permonsvmio.h>

static MPI_Comm  comm;

#undef __FUNCT__
#define __FUNCT__ "SVMRunBinaryClassification"
PetscErrorCode SVMRunBinaryClassification() {
  SVM svm;

  PetscInt  N_all,N_eq;
  Mat       Xt_training,Xt_test;
  Vec       y_training,y_test;

  char      training_file[PETSC_MAX_PATH_LEN] = "examples/heart_scale";
  char      test_file[PETSC_MAX_PATH_LEN]     = "";
  PetscBool test_file_set = PETSC_FALSE;

  PetscFunctionBeginI;
  TRY( PetscOptionsGetString(NULL,NULL,"-f_training",training_file,sizeof(training_file),NULL) );
  TRY( PetscOptionsGetString(NULL,NULL,"-f_test",test_file,sizeof(test_file),&test_file_set) );
  
  TRY( SVMCreate(comm,&svm) );
  TRY( SVMSetType(svm,SVM_BINARY) );
  TRY( SVMSetFromOptions(svm) );

  TRY( SVMLoadData(comm,training_file,&Xt_training,&y_training) );
  TRY( PetscObjectSetName((PetscObject) Xt_training,"Xt_training") );
  TRY( PetscObjectSetName((PetscObject) y_training,"y_training") );

  TRY( SVMSetTrainingDataset(svm,Xt_training,y_training) );
  TRY( SVMTrain(svm) );
  TRY( SVMTest(svm,Xt_training,y_training,&N_all,&N_eq) );

  if (test_file_set) {
    TRY( SVMLoadData(comm,test_file,&Xt_test,&y_test) );
    TRY( PetscObjectSetName((PetscObject) Xt_test,"Xt_test") );
    TRY( PetscObjectSetName((PetscObject) y_test,"y_test") );

    TRY( SVMTest(svm,Xt_test,y_test,&N_all,&N_eq) );

    TRY( MatDestroy(&Xt_test) );
    TRY( VecDestroy(&y_test) );
  }

  TRY( MatDestroy(&Xt_training) );
  TRY( VecDestroy(&y_training) );
  TRY( SVMDestroy(&svm) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  PermonInitialize(&argc,&argv,(char *) 0,(char *) 0);

  PetscFunctionBegin;
  comm = PETSC_COMM_WORLD;
  TRY( SVMRunBinaryClassification() );
  TRY( PermonFinalize() );
  PetscFunctionReturn(0);
}
