#include <permonsvm.h>

static MPI_Comm comm;

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
  char        training_file[PETSC_MAX_PATH_LEN] = "examples/heart_scale.h5";
  char        test_file[PETSC_MAX_PATH_LEN]     = "";
  char        *extension = NULL;

  PetscViewer viewer;

  PetscBool   ishdf5 = PETSC_FALSE;
  PetscBool   test_file_set = PETSC_FALSE;

  PetscFunctionBeginI;
  TRY( PetscOptionsGetString(NULL,NULL,"-f_training",training_file,sizeof(training_file),NULL) );
  TRY( PetscOptionsGetString(NULL,NULL,"-f_test",test_file,sizeof(test_file),&test_file_set) );

  TRY( SVMCreate(comm,&svm) );
  TRY( SVMSetType(svm,SVM_BINARY) );
  TRY( SVMSetFromOptions(svm) );

  /* Load training dataset */
  TRY( GetFilenameExtension(training_file,&extension) );
  TRY( PetscStrcmp(extension,"h5",&ishdf5) );
  if (ishdf5) {
    TRY( PetscViewerHDF5Open(comm,training_file,FILE_MODE_READ,&viewer) );
  } else {
    TRY( PetscViewerSVMLightOpen(comm,training_file,&viewer) );
  }
  TRY( SVMLoadTrainingDataset(svm,viewer) );
  TRY( PetscViewerDestroy(&viewer) );
  /* Train SVM model */
  TRY( SVMTrain(svm) );

  /* Load test dataset and test classification model */
  if (test_file_set) {
    TRY( GetFilenameExtension(test_file,&extension) );
    TRY( PetscStrcmp(extension,"h5",&ishdf5) );
    if (ishdf5) {
      TRY( PetscViewerHDF5Open(comm,test_file,FILE_MODE_READ,&viewer) );
    } else {
      TRY( PetscViewerSVMLightOpen(comm,test_file,&viewer) );
    }
    TRY( SVMLoadTestDataset(svm,viewer) );
    TRY( PetscViewerDestroy(&viewer) );

    TRY( SVMTest(svm) );
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
