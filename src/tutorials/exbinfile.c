
static char help[] = "Computes binary SVM classification model.\n\
Input parameters include:\n\
  -f      : training dataset file\n\
  -f_test : test dataset file\n";

/*
* Include "permonsvm.h" so that we can use PERMON SVM objects and PermonQP solvers.
*/
#include <permonsvm.h>

#define h5       "h5"
#define bin      "bin"
#define SVMLight "svmlight"

PetscErrorCode GetFilenameExtension(const char *filename,char **extension)
{
  char           *extension_inner;

  PetscFunctionBegin;
  PetscCall(PetscStrrchr(filename,'.',&extension_inner));
  *extension = extension_inner;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc,char **argv)
{
  SVM            svm;
  char           training_file[PETSC_MAX_PATH_LEN] = "data/heart_scale.bin";
  char           test_file[PETSC_MAX_PATH_LEN]     = "";
  char           *extension = NULL;
  PetscViewer    viewer;
  PetscBool      test_file_set = PETSC_FALSE;
  PetscBool      ishdf5,isbinary,issvmlight;

  PetscCall(PermonInitialize(&argc,&argv,(char *)0,help));

  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",training_file,sizeof(training_file),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_test",test_file,sizeof(test_file),&test_file_set));

  /* Create SVM object */
  PetscCall(SVMCreate(PETSC_COMM_WORLD,&svm));
  PetscCall(SVMSetType(svm,SVM_BINARY));
  PetscCall(SVMSetFromOptions(svm));

  /* Load training dataset */
  PetscCall(GetFilenameExtension(training_file,&extension));
  PetscCall(PetscStrcmp(extension,h5,&ishdf5));
  PetscCall(PetscStrcmp(extension,bin,&isbinary));
  PetscCall(PetscStrcmp(extension,SVMLight,&issvmlight));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,training_file,FILE_MODE_READ,&viewer));
#else
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PETSc is not configured with HDF5");
#endif
  } else if (isbinary) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,training_file,FILE_MODE_READ,&viewer));
  } else if (issvmlight) {
    PetscCall(PetscViewerSVMLightOpen(PETSC_COMM_WORLD,training_file,&viewer));
  } else {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"File type %s not supported",extension);
  }
  PetscCall(SVMLoadTrainingDataset(svm,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* Load test dataset */
  if (test_file_set) {
    PetscCall(GetFilenameExtension(test_file,&extension));
    PetscCall(PetscStrcmp(extension,h5,&ishdf5));
    PetscCall(PetscStrcmp(extension,bin,&isbinary));
    PetscCall(PetscStrcmp(extension,SVMLight,&issvmlight));
    if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
      PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,test_file,FILE_MODE_READ,&viewer));
#else
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PETSc is not configured with HDF5");
#endif
    } else if (isbinary) {
      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,test_file,FILE_MODE_READ,&viewer));
    } else if (issvmlight) {
      PetscCall(PetscViewerSVMLightOpen(PETSC_COMM_WORLD,test_file,&viewer));
    } else {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"File type %s not supported",extension);
    }
    PetscCall(SVMLoadTestDataset(svm,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* Train SVM model */
  PetscCall(SVMTrain(svm));
  /* Test performance of SVM model */
  if (test_file_set) {
    PetscCall(SVMTest(svm));
  }

  PetscCall(SVMDestroy(&svm));
  PetscCall(PermonFinalize());

  return 0;
}


/*TEST
  testset:
    suffix: 1
    args: -qps_view_convergence -svm_view -svm_view_score
    filter: grep -v MPI
    test:
      args: -f $PERMON_SVM_DIR/src/tutorials/data/heart_scale.bin -f_test $PERMON_SVM_DIR/src/tutorials/data/heart_scale.t.bin
    # TODO use when requires works for testset/test
    #test:
    #  requires: hdf5
    #  nsize: 2
    #  args: -f $PERMON_SVM_DIR/src/tutorials/data/heart_scale.h5 -f_test $PERMON_SVM_DIR/src/tutorials/data/heart_scale.t.h5
    test:
      nsize: 3
      args: -f $PERMON_SVM_DIR/src/tutorials/data/heart_scale.bin -f_test $PERMON_SVM_DIR/src/tutorials/data/heart_scale.t.bin
      args: -Xt_training_mat_type dense -Xt_test_mat_type aij
    test:
      nsize: 4
      args: -f $PERMON_SVM_DIR/src/tutorials/data/heart_scale.svmlight -f_test $PERMON_SVM_DIR/src/tutorials/data/heart_scale.t.svmlight
  test:
    suffix: hdf5
    requires: hdf5
    filter: grep -v MPI
    nsize: 2
    args: -qps_view_convergence -svm_view -svm_view_score
    args: -f $PERMON_SVM_DIR/src/tutorials/data/heart_scale.h5 -f_test $PERMON_SVM_DIR/src/tutorials/data/heart_scale.t.h5
    output_file: output/exbinfile_1.out
TEST*/

