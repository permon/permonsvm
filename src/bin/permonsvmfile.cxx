#include <mpi.h>
#include <petscksp.h>
#include <permonqps.h>
#include <permonsvm.h>

#include "parser.hxx"

static MPI_Comm     comm;
static PetscMPIInt  rank, commsize;

#undef __FUNCT__
#define __FUNCT__ "svm_file_convert"
//TODO workaround until parser works in parallel
//TODO try to use MatDistribute_MPIAIJ
PetscErrorCode svm_file_convert(const char filename[], const char filename_Xt_bin[], const char filename_y_bin[], PetscInt n_examples, PetscInt n_attributes, PetscInt numbering_base)
{
  excape::DataParser parser;
  PetscViewer viewer=NULL;
  Mat Xt_seq=NULL;
  Vec y_seq=NULL;

  PetscFunctionBegin;
  if (rank) PetscFunctionReturn(0);

  parser.SetInputFileName(filename);
  parser.SetNumberingBase(numbering_base);

  TRY( PetscPrintf(PETSC_COMM_SELF, "### PermonSVM: converting input data into PETSc binary format\n") );
  TRY( parser.GetData(PETSC_COMM_SELF, n_examples, n_attributes, &Xt_seq, &y_seq) );
  
  TRY( PetscViewerBinaryOpen(PETSC_COMM_SELF, filename_Xt_bin, FILE_MODE_WRITE, &viewer) );
  TRY( MatView(Xt_seq, viewer) );
  TRY( PetscViewerDestroy(&viewer) );
  TRY( MatDestroy(&Xt_seq) );
  
  TRY( PetscViewerBinaryOpen(PETSC_COMM_SELF, filename_y_bin, FILE_MODE_WRITE, &viewer) );
  TRY( VecView(y_seq, viewer) );
  TRY( PetscViewerDestroy(&viewer) );
  TRY( VecDestroy(&y_seq) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "svm_file_load_binary"
PetscErrorCode svm_file_load_binary(const char filename_Xt_bin[], const char filename_y_bin[], Mat *Xt_new, Vec *y_new)
{
  Mat Xt=NULL;
  Vec y=NULL;
  PetscViewer viewer=NULL;

  PetscFunctionBegin;
  TRY( PetscViewerBinaryOpen(comm, filename_Xt_bin, FILE_MODE_READ, &viewer) );
  TRY( MatCreate(comm, &Xt) );
  TRY( MatSetOptionsPrefix(Xt, "Xt_") );
  TRY( MatSetFromOptions(Xt) );
  TRY( MatLoad(Xt, viewer) );
  TRY( PetscViewerDestroy(&viewer) );
  
  TRY( PetscViewerBinaryOpen(comm, filename_y_bin, FILE_MODE_READ, &viewer) );
  TRY( MatCreateVecs(Xt, NULL, &y) );
  TRY( VecSetOptionsPrefix(y, "y_") );
  TRY( VecSetFromOptions(y) );
  TRY( VecLoad(y, viewer) );
  TRY( PetscViewerDestroy(&viewer) );

  *Xt_new = Xt;
  *y_new = y;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "svm_file_load"
PetscErrorCode svm_file_load(const char filename[], PetscInt n_examples, PetscInt n_attributes, PetscInt numbering_base, Mat *Xt_new, Vec *y_new)
{
  Mat Xt=NULL;
  Vec y=NULL;
  char filename_Xt_bin[PETSC_MAX_PATH_LEN];
  char filename_y_bin[PETSC_MAX_PATH_LEN];
  PetscBool Xt_bin, y_bin;
    
  PetscFunctionBeginI;
  TRY( PetscStrcpy(filename_Xt_bin,filename) );
  TRY( PetscStrcat(filename_Xt_bin, "_Xt.bin") );
  TRY( PetscTestFile(filename_Xt_bin, 'r', &Xt_bin) );

  TRY( PetscStrcpy(filename_y_bin,filename) );
  TRY( PetscStrcat(filename_y_bin, "_y.bin") );
  TRY( PetscTestFile(filename_y_bin, 'r', &y_bin) );

  if (!Xt_bin || !y_bin) {
    TRY( svm_file_convert(filename, filename_Xt_bin, filename_y_bin, n_examples, n_attributes, numbering_base) );
    TRY( svm_file_load_binary(filename_Xt_bin, filename_y_bin, &Xt, &y) );
  } else {
    PetscInt M;
    TRY( svm_file_load_binary(filename_Xt_bin, filename_y_bin, &Xt, &y) );
    TRY( MatGetSize(Xt,&M,NULL) );
    if (M != n_examples) {
      TRY( PetscPrintf(PETSC_COMM_WORLD, "### PermonSVM: input data in PETSc binary format have different size than n_examples, reconverting\n") );
      TRY( svm_file_convert(filename, filename_Xt_bin, filename_y_bin, n_examples, n_attributes, numbering_base) );
      TRY( svm_file_load_binary(filename_Xt_bin, filename_y_bin, &Xt, &y) );
    } else {
      TRY( PetscPrintf(PETSC_COMM_WORLD, "### PermonSVM: reusing input data in PETSc binary format\n") );
    }
  }
  
  *Xt_new = Xt;
  *y_new = y;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMRemoveZeroColumns"
PetscErrorCode PermonSVMRemoveZeroColumns(Mat *Xt, Mat *Xt_test) 
{
  Mat X, X_sub, Xt_sub;
  IS cis;

  PetscFunctionBeginI;
  TRY( FllopMatTranspose(*Xt,MAT_TRANSPOSE_EXPLICIT,&X) );
  TRY( MatFindNonzeroRows(X,&cis) );
  if (!cis) {
    TRY( PetscPrintf(comm, "### PermonSVM: no zero columns found in Xt\n") );
    PetscFunctionReturnI(0);
  }

  {
    PetscInt n,nnz;
    TRY( MatGetSize(*Xt,NULL,&n) );
    TRY( ISGetSize(cis,&nnz) );
    TRY( PetscPrintf(comm, "### PermonSVM: removing %d zero columns found in Xt\n", n-nnz) );
  }

  TRY( MatGetSubMatrix(X,cis,NULL,MAT_INITIAL_MATRIX,&X_sub) );
  TRY( FllopMatTranspose(X_sub,MAT_TRANSPOSE_EXPLICIT,&Xt_sub) );
  TRY( MatDestroy(&X) );
  TRY( MatDestroy(&X_sub) );
  TRY( MatDestroy(Xt) );
  *Xt = Xt_sub;

  if (Xt_test && *Xt_test) {
    TRY( FllopMatTranspose(*Xt_test,MAT_TRANSPOSE_EXPLICIT,&X) );
    TRY( MatGetSubMatrix(X,cis,NULL,MAT_INITIAL_MATRIX,&X_sub) );
    TRY( FllopMatTranspose(X_sub,MAT_TRANSPOSE_EXPLICIT,&Xt_sub) );
    TRY( MatDestroy(&X) );
    TRY( MatDestroy(&X_sub) );
    TRY( MatDestroy(Xt_test) );
    *Xt_test = Xt_sub;
  }
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMLoadData"
PetscErrorCode PermonSVMLoadData(Mat *Xt, Vec *y, Mat *Xt_test, Vec *y_test)
{
  PetscInt       M,N;
  char           filename[PETSC_MAX_PATH_LEN] = "dummy.txt";
  char           filename_test[PETSC_MAX_PATH_LEN] = "";
  PetscInt       n_examples = PETSC_DEFAULT;  /* PETSC_DEFAULT or PETSC_DECIDE means all */
  PetscInt       n_test_examples = PETSC_DEFAULT;
  PetscInt       numbering_base = 1;
  PetscBool      filename_test_set = PETSC_FALSE;
  PetscBool      remove_zero_columns = PETSC_FALSE;

  PetscFunctionBeginI;
  TRY( PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),NULL) );
  TRY( PetscOptionsGetString(NULL,NULL,"-f_test",filename_test,sizeof(filename_test),&filename_test_set) );
  TRY( PetscOptionsGetInt(NULL,NULL,"-n_examples",&n_examples,NULL));
  TRY( PetscOptionsGetInt(NULL,NULL,"-n_test_examples",&n_test_examples,NULL));
  TRY( PetscOptionsGetInt(NULL,NULL,"-numbering_base",&numbering_base, NULL));
  TRY( PetscOptionsGetBool(NULL,NULL,"-remove_zero_columns",&remove_zero_columns, NULL));

  TRY( svm_file_load(filename, n_examples, PETSC_DECIDE, numbering_base, Xt, y) );
  TRY( MatGetSize(*Xt, &M, &N));
  FLLOP_ASSERT(n_examples == PETSC_DECIDE || n_examples == PETSC_DEFAULT || (n_examples >= 0 && n_examples == M),
              "n_examples == PETSC_DECIDE || n_examples == PETSC_DEFAULT || (n_examples >= 0 && n_examples == M)");
  TRY( PetscPrintf(comm, "### PermonSVM: loaded %d training examples with %d attributes from file %s\n",M,N,filename));

  if (filename_test_set) {
    TRY( svm_file_load(filename_test, n_test_examples, N, numbering_base, Xt_test, y_test) );
    TRY( MatGetSize(*Xt_test, &M, &N));
    TRY( PetscPrintf(comm, "### PermonSVM: loaded %d testing examples with %d attributes from file %s\n",M,N,filename_test));
  } else {
    *Xt_test = NULL;
    *y_test = NULL;
  }

  if (remove_zero_columns) {
    TRY( PermonSVMRemoveZeroColumns(Xt, Xt_test) );
  }

  TRY( PetscObjectSetName((PetscObject)*Xt, "Xt") );
  TRY( PetscObjectSetName((PetscObject)*y,  "y") );
  if (*Xt_test) TRY( PetscObjectSetName((PetscObject)*Xt_test, "Xt_test") );
  if (*y_test)  TRY( PetscObjectSetName((PetscObject)*y_test,  "y_test") );

  TRY( MatPrintInfo(*Xt) );
  TRY( VecPrintInfo(*y) );
  if (*Xt_test) TRY( MatPrintInfo(*Xt_test) );
  if (*y_test)  TRY( VecPrintInfo(*y_test) );
  TRY( PetscPrintf(comm, "\n") );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMRun"
PetscErrorCode PermonSVMRun()
{
    PermonSVM svm;
    PetscReal C;
    PetscInt N_all, N_eq;
    Mat Xt, Xt_test;
    Vec y, y_test;
    
    PetscFunctionBeginI;
    TRY( PermonSVMLoadData(&Xt, &y, &Xt_test, &y_test) );
    
    /* ------------------------------------------------------------------------ */
    TRY( PermonSVMCreate(comm, &svm) );
    TRY( PermonSVMSetTrainingSamples(svm, Xt, y) );
    TRY( PermonSVMSetFromOptions(svm) );
    TRY( PermonSVMTrain(svm) );
    TRY( PermonSVMTest(svm, Xt, y, &N_all, &N_eq) );
    TRY( PermonSVMGetC(svm, &C) );
    TRY( PetscPrintf(comm, "\n### PermonSVM: %8d of %8d training examples classified correctly (%.2f%%) with C = %1.1e\n", N_eq, N_all, ((PetscReal)N_eq)/((PetscReal)N_all)*100.0, C) );

    /* ------------------------------------------------------------------------ */ 
    if (Xt_test) {
      TRY( PermonSVMTest(svm, Xt_test, y_test, &N_all, &N_eq) );
      TRY( PermonSVMGetC(svm, &C) );
      TRY( PetscPrintf(comm, "### PermonSVM: %8d of %8d  testing examples classified correctly (%.2f%%) with C = %1.1e\n", N_eq, N_all, ((PetscReal)N_eq)/((PetscReal)N_all)*100.0, C) );
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

  TRY( PermonSVMRun() );
  TRY( FllopFinalize() );
  PetscFunctionReturn(0);
}
