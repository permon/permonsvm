#include <mpi.h>
#include <petscksp.h>
#include <fllopqps.h>
#include <permonsvm.h>

#include "parser.hxx"

extern char Fllop_input_dir[FLLOP_MAX_PATH_LEN], Fllop_output_dir[FLLOP_MAX_PATH_LEN];

static MPI_Comm     comm;
static PetscMPIInt  rank, commsize;

#undef __FUNCT__
#define __FUNCT__ "testSVM_load_data_from_file"
PetscErrorCode testSVM_load_data_from_file(const char *filename, PetscInt n_examples, PetscInt numbering_base, Mat *Xt_new, Vec *y_new)
{
  excape::DataParser parser;
  Mat Xt_seq=NULL, Xt=NULL;
  Vec y_seq=NULL, y=NULL;
  PetscViewer viewer=NULL;
  char filename_Xt_bin[PETSC_MAX_PATH_LEN];
  char filename_y_bin[PETSC_MAX_PATH_LEN];
  PetscBool Xt_bin, y_bin;
    
  PetscFunctionBeginI;
  parser.SetInputFileName(filename);
  parser.SetNumberingBase(numbering_base);
  
  //TODO workaround until parser works in parallel
  //TODO try to use MatDistribute_MPIAIJ
  TRY( PetscStrcpy(filename_Xt_bin,filename) );
  TRY( PetscStrcat(filename_Xt_bin, "_Xt.bin") );
  TRY( PetscTestFile(filename_Xt_bin, 'r', &Xt_bin) );

  TRY( PetscStrcpy(filename_y_bin,filename) );
  TRY( PetscStrcat(filename_y_bin, "_y.bin") );
  TRY( PetscTestFile(filename_y_bin, 'r', &y_bin) );

  if (!Xt_bin || !y_bin) {
    if (!rank) {
      TRY( PetscPrintf(PETSC_COMM_SELF, "### PermonSVM: converting input data into PETSc binary format\n") );
      TRY( parser.GetData(PETSC_COMM_SELF, n_examples, &Xt_seq, &y_seq) );
      
      TRY( PetscViewerBinaryOpen(PETSC_COMM_SELF, filename_Xt_bin, FILE_MODE_WRITE, &viewer) );
      TRY( MatView(Xt_seq, viewer) );
      TRY( PetscViewerDestroy(&viewer) );
      TRY( MatDestroy(&Xt_seq) );
      
      TRY( PetscViewerBinaryOpen(PETSC_COMM_SELF, filename_y_bin, FILE_MODE_WRITE, &viewer) );
      TRY( VecView(y_seq, viewer) );
      TRY( PetscViewerDestroy(&viewer) );
      TRY( VecDestroy(&y_seq) );
    }
  } else {
    TRY( PetscPrintf(PETSC_COMM_WORLD, "### PermonSVM: reusing input data in PETSc binary format\n") );
  }
  
  TRY( PetscViewerBinaryOpen(comm, filename_Xt_bin, FILE_MODE_READ, &viewer) );
  TRY( MatCreate(comm, &Xt) );
  TRY( MatSetType(Xt, MATAIJ) );
  TRY( MatLoad(Xt, viewer) );
  TRY( PetscViewerDestroy(&viewer) );
  
  TRY( PetscViewerBinaryOpen(comm, filename_y_bin, FILE_MODE_READ, &viewer) );
  TRY( MatCreateVecs(Xt, NULL, &y) );
  TRY( VecLoad(y, viewer) );
  TRY( PetscViewerDestroy(&viewer) );
  
  *Xt_new = Xt;
  *y_new = y;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "testSVM_load_data"
PetscErrorCode testSVM_load_data(Mat *Xt, Vec *y, Mat *Xt_test, Vec *y_test)
{
  PetscInt       M,N;
  char           filename[PETSC_MAX_PATH_LEN] = "dummy.txt";
  char           filename_test[PETSC_MAX_PATH_LEN] = "";
  PetscInt       n_examples = PETSC_DEFAULT;  /* PETSC_DEFAULT or PETSC_DECIDE means all */
  PetscInt       numbering_base;
  PetscBool      filename_test_set = PETSC_FALSE;

  PetscFunctionBeginI;
  TRY( PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),NULL) );
  TRY( PetscOptionsGetString(NULL,NULL,"-f_test",filename_test,sizeof(filename_test),&filename_test_set) );
  TRY( PetscOptionsGetInt(NULL,NULL,"-n_examples",&n_examples,NULL));
  TRY( PetscOptionsGetInt(NULL,NULL,"-numbering_base",&numbering_base, NULL));

  TRY( testSVM_load_data_from_file(filename, n_examples, numbering_base, Xt, y) );
  TRY( MatGetSize(*Xt, &M, &N));
  TRY( PetscPrintf(comm, "\n\n### PermonSVM: loaded %d training examples with %d attributes from file %s\n",M,N,filename));

  if (filename_test_set) {
    TRY( testSVM_load_data_from_file(filename_test, PETSC_DECIDE, numbering_base, Xt_test, y_test) );
    TRY( MatGetSize(*Xt_test, &M, &N));
    TRY( PetscPrintf(comm, "### PermonSVM: loaded %d testing examples with %d attributes from file %s\n",M,N,filename_test));
  } else {
    *Xt_test = NULL;
    *y_test = NULL;
  }

  TRY( PetscPrintf(comm, "\n") );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "testSVM_files"
PetscErrorCode testSVM_files()
{
    PermonSVM svm;
    PetscReal C;
    PetscInt N_all, N_eq;
    Mat Xt, Xt_test;
    Vec y, y_test;
    
    PetscFunctionBeginI;
    TRY( testSVM_load_data(&Xt, &y, &Xt_test, &y_test) );
    
    /* ------------------------------------------------------------------------ */
    TRY( PermonSVMCreate(comm, &svm) );
    TRY( PermonSVMSetTrainingSamples(svm, Xt, y) );
    TRY( PermonSVMSetFromOptions(svm) );
    TRY( PermonSVMTrain(svm) );
    TRY( PermonSVMTest(svm, Xt, y, &N_all, &N_eq) );
    TRY( PermonSVMGetPenalty(svm, &C) );
    TRY( PetscPrintf(comm, "\n### PermonSVM: %8d of %8d training examples classified correctly (%.2f%) with C = %1.1e\n", N_eq, N_all, ((PetscReal)N_eq)/((PetscReal)N_all)*100.0, C) );

    /* ------------------------------------------------------------------------ */ 
    if (Xt_test) {
      TRY( PermonSVMTest(svm, Xt_test, y_test, &N_all, &N_eq) );
      TRY( PermonSVMGetPenalty(svm, &C) );
      TRY( PetscPrintf(comm, "### PermonSVM: %8d of %8d  testing examples classified correctly (%.2f%) with C = %1.1e\n", N_eq, N_all, ((PetscReal)N_eq)/((PetscReal)N_all)*100.0, C) );
      TRY( MatDestroy(&Xt_test) );
      TRY( VecDestroy(&y_test) );
    }
    
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