#include <mpi.h>
#include <permonsvm.h>
#include <permonsvmio.h>

static MPI_Comm    comm;
static PetscMPIInt comm_rank,comm_size;

#undef __FUNCT__
#define __FUNCT__ "PermonSVMRunBinaryClassification"
PetscErrorCode PermonSVMRunBinaryClassification() {
  PermonSVM svm;
  PetscReal C;
  PetscInt  N_all, N_eq,M,N;
  Mat       Xt_training,Xt_test;
  Vec       y_training,y_test;
  char      training_file[PETSC_MAX_PATH_LEN] = "dummy.txt";
  char      test_file[PETSC_MAX_PATH_LEN] = "";
  PetscBool test_file_set = PETSC_FALSE;

  PetscFunctionBeginI;
  TRY( PetscOptionsGetString(NULL,NULL,"-f_training",training_file,sizeof(training_file),NULL) );
  TRY( PetscOptionsGetString(NULL,NULL,"-f_test",test_file,sizeof(test_file),&test_file_set) );
  
  TRY( PermonSVMCreate(comm,&svm) );
  TRY( PermonSVMSetFromOptions(svm) );

  TRY( PermonSVMLoadData(comm,training_file,&Xt_training,&y_training) );
  TRY( MatGetSize(Xt_training,&M,&N));
  TRY( PetscPrintf(comm,"### PermonSVM:\tloaded %8d training samples with %8d attributes from file %s\n",M,N,training_file) );

  TRY( PermonSVMSetTrainingSamples(svm,Xt_training,y_training) );
  TRY( PermonSVMTrain(svm) );
  TRY( PermonSVMTest(svm,Xt_training,y_training,&N_all,&N_eq) );
  TRY( PermonSVMGetC(svm,&C) );
  TRY( PetscPrintf(comm,"\n### PermonSVM: %8d of %8d training samples classified correctly (%.2f%%) with C = %1.1e\n",
                   N_eq,N_all,((PetscReal)N_eq)/((PetscReal)N_all)*100.0,C) );

  if (test_file_set) {
    TRY( PermonSVMLoadData(comm,test_file,&Xt_test,&y_test) );
    TRY( MatGetSize(Xt_test,&M,&N) );
    TRY( PetscPrintf(comm,"### PermonSVM:\tloaded %8d test samples with %8d attributes from file %s\n",M,N,test_file) );

    TRY( PermonSVMTest(svm,Xt_test,y_test,&N_all,&N_eq) );
    TRY( PetscPrintf(comm,"\n### PermonSVM: %8d of %8d test samples classified correctly (%.2f%%)\n",N_eq,N_all,((PetscReal)N_eq)/((PetscReal)N_all)*100.0) );

    TRY( MatDestroy(&Xt_test) );
    TRY( VecDestroy(&y_test) );
  }

  TRY( MatDestroy(&Xt_training) );
  TRY( VecDestroy(&y_training) );
  TRY( PermonSVMDestroy(&svm) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  PermonInitialize(&argc,&argv,(char *) 0,(char *) 0);

  PetscFunctionBegin;
  comm = PETSC_COMM_WORLD;
  TRY(MPI_Comm_rank(comm,&comm_rank));
  TRY(MPI_Comm_size(comm,&comm_size));

  TRY(PetscPrintf(comm,"PETSC_DIR:\t%s\n",PETSC_DIR));
  TRY(PetscPrintf(comm,"PETSC_ARCH:\t%s\n",PETSC_ARCH));
#ifdef PETSC_RELEASE_DATE
#define DATE PETSC_RELEASE_DATE
#else
#define DATE PETSC_VERSION_DATE
#endif
  TRY(PetscPrintf(comm,"PETSc version:\t%d.%d.%d patch %d (%s)\n",PETSC_VERSION_MAJOR,PETSC_VERSION_MINOR,
                  PETSC_VERSION_SUBMINOR,PETSC_VERSION_PATCH,DATE));
#undef DATE
  TRY(PermonSVMRunBinaryClassification());

  TRY(PermonFinalize());
  PetscFunctionReturn(0);
}
