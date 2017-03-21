#include <mpi.h>
#include <petscksp.h>
#include <fllopqps.h>
#include <permonsvm.h>

#include "excape_parser.hxx"

extern char Fllop_input_dir[FLLOP_MAX_PATH_LEN], Fllop_output_dir[FLLOP_MAX_PATH_LEN];

static MPI_Comm     comm;
static PetscMPIInt  rank, commsize;

#undef __FUNCT__
#define __FUNCT__ "testSVM_files"
PetscErrorCode testSVM_files()
{
    excape::DataParser parser;
    PermonSVM svm;
    PetscReal C, C_min, C_max, C_step;
    PetscInt N_all, N_eq, M, N, nfolds;
    Mat Xt;
    Vec y;
    char           filename[PETSC_MAX_PATH_LEN] = "dummy.txt";
    PetscInt       n_examples = -1;  /* -1 means all */
    
    PetscFunctionBeginI;
    TRY( PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),NULL) );
    TRY( PetscOptionsGetInt(NULL,NULL,"-n_examples",&n_examples,NULL));
    
    nfolds = 4;
    C_min = 1e-4;
    C_step = 10;
    C_max = 1e4;
    
    parser.SetInputFileName(filename);
    TRY( parser.GetData(comm, n_examples, &Xt, &y) );
    
    TRY( MatGetSize(Xt, &M, &N));
    TRY( PetscPrintf(comm, "\n\n#### LOADED %d EXAMPLES WITH %d ATTRIBUTES FROM FILE %s\n\n",M,N,filename));
    
    /* ------------------------------------------------------------------------ */
    {
        TRY( PermonSVMCreate(comm, &svm) );
    }
    
    TRY( PermonSVMSetTrainingSamples(svm, Xt, y) );
    TRY( PermonSVMSetPenaltyMin(svm,C_min) );
    TRY( PermonSVMSetPenaltyMax(svm,C_max) );
    TRY( PermonSVMSetPenaltyStep(svm,C_step) );
    TRY( PermonSVMSetNfolds(svm, nfolds) );
    TRY( PermonSVMSetFromOptions(svm) );
    TRY( PermonSVMTrain(svm) );
    TRY( PermonSVMTest(svm, Xt, y, &N_all, &N_eq) );
    TRY( PermonSVMGetPenalty(svm, &C) );
    
    TRY( PetscPrintf(comm, "\n#### %d OF %d EXAMPLES CLASSIFIED CORRECTLY (%.2f%) WITH C = %1.1e\n", N_eq, N_all, ((PetscReal)N_eq)/((PetscReal)N_all)*100.0, C) );
    
    /* ------------------------------------------------------------------------ */ 
    TRY( PermonSVMDestroy(&svm) );
    TRY( MatDestroy(&Xt) );
    TRY( VecDestroy(&y) );
    PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char** argv) 
{ 
  FllopInitialize(&argc, &argv, (char*) 0);

  PetscFunctionBegin;
  comm = PETSC_COMM_WORLD;
  TRY( MPI_Comm_rank(comm, &rank) );
  TRY( MPI_Comm_size(comm, &commsize) );

  FLLOP_ASSERT(commsize==1,"currently only sequential");

  TRY( PetscPrintf(comm, "PETSC_DIR:\t" PETSC_DIR "\n") );
  TRY( PetscPrintf(comm, "PETSC_ARCH:\t" PETSC_ARCH "\n") );
#ifdef PETSC_RELEASE_DATE
#define DATE PETSC_RELEASE_DATE
#else
#define DATE PETSC_VERSION_DATE
#endif
  TRY( PetscPrintf(comm, "PETSc version:\t%d.%d.%d patch %d (%s)\n", PETSC_VERSION_MAJOR, PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR, PETSC_VERSION_PATCH, DATE) );
#undef DATE
  TRY( PetscPrintf(comm,"Input dir:\t%s\n", Fllop_input_dir) );
  TRY( PetscPrintf(comm,"Output dir:\t%s\n", Fllop_output_dir) );

  TRY( testSVM_files() );
  TRY( FllopFinalize() );
  PetscFunctionReturn(0);
}