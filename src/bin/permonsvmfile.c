#include <permonsvm.h>

static MPI_Comm comm;

#define h5       "h5"
#define bin      "bin"
#define SVMLight "svmlight"

#undef __FUNCT__
#define __FUNCT__ "GetFilenameExtension"
PetscErrorCode GetFilenameExtension(const char *filename,char **extension)
{
  char *extension_inner;

  PetscFunctionBegin;
  TRY( PetscStrrchr(filename,'.',&extension_inner) );
  *extension = extension_inner;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMRunBinaryClassification"
PetscErrorCode SVMRunBinaryClassification()
{
  SVM         svm;

  char        training_file[PETSC_MAX_PATH_LEN] = "examples/data/heart_scale.bin";
  char        test_file[PETSC_MAX_PATH_LEN]     = "";
  char        kernel_file[PETSC_MAX_PATH_LEN]   = "";
  char        training_result_file[PETSC_MAX_PATH_LEN] = "";
  char        test_result_file[PETSC_MAX_PATH_LEN]     = "";
  PetscBool   test_file_set = PETSC_FALSE,kernel_file_set = PETSC_FALSE;
  PetscBool   training_result_file_set=PETSC_FALSE,test_result_file_set=PETSC_FALSE;

  char        *extension = NULL;

  PetscViewer viewer;
  PetscBool   ishdf5,isbinary,issvmlight;

  PetscFunctionBeginI;
  TRY( PetscOptionsGetString(NULL,NULL,"-f_training",training_file,sizeof(training_file),NULL) );
  TRY( PetscOptionsGetString(NULL,NULL,"-f_test",test_file,sizeof(test_file),&test_file_set) );
  TRY( PetscOptionsGetString(NULL,NULL,"-f_kernel",kernel_file,sizeof(kernel_file),&kernel_file_set) );

  TRY( PetscOptionsGetString(NULL,NULL,"-f_training_predictions",training_result_file,sizeof(training_result_file),&training_result_file_set) );
  TRY( PetscOptionsGetString(NULL,NULL,"-f_test_predictions",test_result_file,sizeof(test_result_file),&test_result_file_set) );

  TRY( SVMCreate(comm,&svm) );
  TRY( SVMSetType(svm,SVM_BINARY) );
  TRY( SVMSetFromOptions(svm) );

  /* Load training dataset */
  TRY( GetFilenameExtension(training_file,&extension) );
  TRY( PetscStrcmp(extension,h5,&ishdf5) );
  TRY( PetscStrcmp(extension,bin,&isbinary) );
  TRY( PetscStrcmp(extension,SVMLight,&issvmlight) );
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    TRY( PetscViewerHDF5Open(comm,training_file,FILE_MODE_READ,&viewer) );
#else
    FLLOP_SETERRQ(comm,PETSC_ERR_SUP,"PETSc is not configured with HDF5");
#endif
  } else if (isbinary) {
    TRY( PetscViewerBinaryOpen(comm,training_file,FILE_MODE_READ,&viewer) );
  } else if (issvmlight) {
    TRY( PetscViewerSVMLightOpen(comm,training_file,&viewer) );
  } else {
    FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"File type %s not supported",extension);
  }
  TRY( SVMLoadTrainingDataset(svm,viewer) );
  TRY( PetscViewerDestroy(&viewer) );

  /* Load precomputed Gramian (kernel matrix), i.e. phi(X^T) * phi(X) generally */
  if (kernel_file_set) {
    TRY( GetFilenameExtension(kernel_file,&extension) );
    TRY( PetscStrcmp(extension,h5,&ishdf5) );
    TRY( PetscStrcmp(extension,bin,&isbinary) );

    if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
      TRY( PetscViewerHDF5Open(comm,kernel_file,FILE_MODE_READ,&viewer) );
#else
      FLLOP_SETERRQ(comm,PETSC_ERR_SUP,"PETSc is not configured with HDF5");
#endif
    } else if (isbinary) {
      TRY( PetscViewerBinaryOpen(comm,kernel_file,FILE_MODE_READ,&viewer) );
    } else {
      FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"File type %s not supported",extension);
    }
    TRY( SVMLoadGramian(svm,viewer) );
    TRY( PetscViewerDestroy(&viewer) );
  }
  /* Load test dataset */
  if (test_file_set) {
    TRY( GetFilenameExtension(test_file,&extension) );
    TRY( PetscStrcmp(extension,h5,&ishdf5) );
    TRY( PetscStrcmp(extension,bin,&isbinary) );
    TRY( PetscStrcmp(extension,SVMLight,&issvmlight) );
    if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
      TRY( PetscViewerHDF5Open(comm,test_file,FILE_MODE_READ,&viewer) );
#else
      FLLOP_SETERRQ(comm,PETSC_ERR_SUP,"PETSc is not configured with HDF5");
#endif
    } else if (isbinary) {
      TRY( PetscViewerBinaryOpen(comm,test_file,FILE_MODE_READ,&viewer) );
    } else if (issvmlight) {
      TRY( PetscViewerSVMLightOpen(comm,test_file,&viewer) );
    } else {
      FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"File type %s not supported",extension);
    }
    TRY( SVMLoadTestDataset(svm,viewer) );
    TRY( PetscViewerDestroy(&viewer) );
  }

  /* Train SVM model */
  TRY( SVMTrain(svm) );
  /* Test performance of SVM model */
  if (test_file_set) {
    TRY( SVMTest(svm) );
  }

  /* Save results */
  if (training_result_file_set) {
    TRY( GetFilenameExtension(training_result_file,&extension) );
    TRY( PetscStrcmp(extension,h5,&ishdf5) );
    TRY( PetscStrcmp(extension,bin,&isbinary) );
    if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
      TRY( PetscViewerHDF5Open(PETSC_COMM_WORLD,training_result_file,FILE_MODE_WRITE,&viewer) );
#else
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PETSc is not configured with HDF5");
#endif
    } else if (isbinary) {
      TRY( PetscViewerBinaryOpen(PETSC_COMM_WORLD,training_result_file,FILE_MODE_WRITE,&viewer) );
    } else {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"File type %s not supported",extension);
    }
    TRY( SVMViewTrainingPredictions(svm,viewer) );
    TRY( PetscViewerDestroy(&viewer) );
  }

  if (test_result_file_set) {
    TRY( GetFilenameExtension(test_result_file,&extension) );
    TRY( PetscStrcmp(extension,h5,&ishdf5) );
    TRY( PetscStrcmp(extension,bin,&isbinary) );
    if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
      TRY( PetscViewerHDF5Open(PETSC_COMM_WORLD,test_result_file,FILE_MODE_WRITE,&viewer) );
#else
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PETSc is not configured with HDF5");
#endif
    } else if (isbinary) {
      TRY( PetscViewerBinaryOpen(PETSC_COMM_WORLD,test_result_file,FILE_MODE_WRITE,&viewer) );
    } else {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"File type %s not supported",extension);
    }
    TRY( SVMViewTestPredictions(svm,viewer) );
    TRY( PetscViewerDestroy(&viewer) );
  }

  TRY( SVMDestroy(&svm) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PermonInitialize(&argc,&argv,(char *) 0,(char *) 0);

  PetscFunctionBegin;
  comm = PETSC_COMM_WORLD;
  TRY( SVMRunBinaryClassification() );
  TRY( PermonFinalize() );
  PetscFunctionReturn(0);
}
