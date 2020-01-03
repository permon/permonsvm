
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrrchr(filename,'.',&extension_inner);CHKERRQ(ierr);
  *extension = extension_inner;
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  SVM            svm;
  char           training_file[PETSC_MAX_PATH_LEN] = "examples/data/heart_scale.bin";
  char           test_file[PETSC_MAX_PATH_LEN]     = "";
  char           *extension = NULL;
  PetscViewer    viewer;
  PetscBool      test_file_set = PETSC_FALSE;
  PetscBool      ishdf5,isbinary,issvmlight;
  PetscErrorCode ierr;

  ierr = PermonInitialize(&argc,&argv,(char *)0,help);if (ierr) return ierr;

  ierr = PetscOptionsGetString(NULL,NULL,"-f",training_file,sizeof(training_file),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f_test",test_file,sizeof(test_file),&test_file_set);CHKERRQ(ierr);

  /* Create SVM object */
  ierr = SVMCreate(PETSC_COMM_WORLD,&svm);CHKERRQ(ierr);
  ierr = SVMSetType(svm,SVM_BINARY);CHKERRQ(ierr);
  ierr = SVMSetFromOptions(svm);CHKERRQ(ierr);

  /* Load training dataset */
  ierr = GetFilenameExtension(training_file,&extension);CHKERRQ(ierr);
  ierr = PetscStrcmp(extension,h5,&ishdf5);CHKERRQ(ierr);
  ierr = PetscStrcmp(extension,bin,&isbinary);CHKERRQ(ierr);
  ierr = PetscStrcmp(extension,SVMLight,&issvmlight);CHKERRQ(ierr);
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,training_file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
#else
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PETSc is not configured with HDF5");
#endif
  } else if (isbinary) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,training_file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  } else if (issvmlight) {
    ierr = PetscViewerSVMLightOpen(PETSC_COMM_WORLD,training_file,&viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"File type %s not supported",extension);
  }
  ierr = SVMLoadTrainingDataset(svm,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* Load test dataset */
  if (test_file_set) {
    ierr = GetFilenameExtension(test_file,&extension);CHKERRQ(ierr);
    ierr = PetscStrcmp(extension,h5,&ishdf5);CHKERRQ(ierr);
    ierr = PetscStrcmp(extension,bin,&isbinary);CHKERRQ(ierr);
    ierr = PetscStrcmp(extension,SVMLight,&issvmlight);CHKERRQ(ierr);
    if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
      ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,test_file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
#else
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PETSc is not configured with HDF5");
#endif
    } else if (isbinary) {
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,test_file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    } else if (issvmlight) {
      ierr = PetscViewerSVMLightOpen(PETSC_COMM_WORLD,test_file,&viewer);CHKERRQ(ierr);
    } else {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"File type %s not supported",extension);
    }
    ierr = SVMLoadTestDataset(svm,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* Train SVM model */
  ierr = SVMTrain(svm);CHKERRQ(ierr);
  /* Test performance of SVM model */
  if (test_file_set) {
    ierr = SVMTest(svm);CHKERRQ(ierr);
  }

  ierr = SVMDestroy(&svm);CHKERRQ(ierr);
  ierr = PermonFinalize();
  return ierr;
}


/*TEST
  testset:
    suffix: 1
    args: -qps_view_convergence -svm_view -svm_view_score
    filter: grep -v MPI
    test:
      args: -f $PERMON_SVM_DIR/examples/data/heart_scale.bin -f_test $PERMON_SVM_DIR/examples/data/heart_scale.t.bin
    # TODO use when requires works for testset/test
    #test:
    #  requires: hdf5
    #  nsize: 2
    #  args: -f $PERMON_SVM_DIR/examples/data/heart_scale.h5 -f_test $PERMON_SVM_DIR/examples/data/heart_scale.t.h5
    test:
      nsize: 3
      args: -f $PERMON_SVM_DIR/examples/data/heart_scale.bin -f_test $PERMON_SVM_DIR/examples/data/heart_scale.t.bin
      args: -Xt_training_mat_type dense -Xt_test_mat_type aij
    test:
      nsize: 4
      args: -f $PERMON_SVM_DIR/examples/data/heart_scale.svmlight -f_test $PERMON_SVM_DIR/examples/data/heart_scale.t.svmlight
  test:
    suffix: hdf5
    requires: hdf5
    filter: grep -v MPI
    nsize: 2
    args: -qps_view_convergence -svm_view -svm_view_score
    args: -f $PERMON_SVM_DIR/examples/data/heart_scale.h5 -f_test $PERMON_SVM_DIR/examples/data/heart_scale.t.h5
    output_file: output/exbinfile_1.out
TEST*/

