#include <mpi.h>
#include <permonsvm.h>
#include <permonsvmio.h>

static MPI_Comm  comm;

#undef __FUNCT__
#define __FUNCT__ "SVMRunBinaryClassification"
PetscErrorCode SVMRunBinaryClassification() {
  SVM       svm;

  char      training_file[PETSC_MAX_PATH_LEN] = "examples/heart_scale";
  char      test_file[PETSC_MAX_PATH_LEN]     = "";
  PetscBool test_file_set = PETSC_FALSE;

  PetscFunctionBeginI;
  TRY( PetscOptionsGetString(NULL,NULL,"-f_training",training_file,sizeof(training_file),NULL) );
  TRY( PetscOptionsGetString(NULL,NULL,"-f_test",test_file,sizeof(test_file),&test_file_set) );

  TRY( SVMCreate(comm,&svm) );
  TRY( SVMSetType(svm,SVM_BINARY) );
  TRY( SVMSetFromOptions(svm) );

  TRY( SVMLoadTrainingDataset(svm,training_file) );
  TRY( SVMTrain(svm) );

  if (test_file_set) {
    TRY( SVMLoadTestDataset(svm,test_file) );
    TRY( SVMTest(svm) );
  }

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
