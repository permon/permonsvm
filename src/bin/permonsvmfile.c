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
  PetscInt  N_all, N_eq;
  Mat       Xt_training;
  Vec       y_training;
  char      training_file[PETSC_MAX_PATH_LEN] = "dummy.txt";

  PetscFunctionBeginI;
  TRY( PermonSVMCreate(comm,&svm) );
  TRY( PermonSVMSetFromOptions(svm) );

  TRY( PermonSVMLoadData(training_file,&Xt_training,&y_training) );

  if (Xt_training && y_training) {
    TRY( PermonSVMSetTrainingSamples(svm,Xt_training,y_training) );
    TRY( PermonSVMTrain(svm) );
    TRY( PermonSVMTest(svm,Xt_training,y_training,&N_all,&N_eq) );
    TRY( PermonSVMGetC(svm,&C) );
    TRY( PetscPrintf(comm,"\n### PermonSVM: %8d of %8d training samples classified correctly (%.2f%%) with C = %1.1e\n",
                     N_eq,N_all,((PetscReal)N_eq)/((PetscReal)N_all)*100.0,C) );
  }

  if (Xt_training) TRY( MatDestroy(&Xt_training) );
  if (y_training) TRY( VecDestroy(&y_training) );
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
