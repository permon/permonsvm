
static char help[] = "Trains binary SVM classification model using precomputed Gramian matrix.\n\
Input parameters:\n\
  -f       : training dataset file\n\
  -f_kernel: file contains precomputed Gramian\n\
  -f_test  : test dataset file\n";

#include <permonsvm.h>

int main(int argc,char **argv)
{
  SVM            svm;

  char           training_file[PETSC_MAX_PATH_LEN] = "data/heart_scale.bin";
  char           kernel_file[PETSC_MAX_PATH_LEN]   = "";
  char           test_file[PETSC_MAX_PATH_LEN]     = "";
  PetscBool      test_file_set = PETSC_FALSE,kernel_file_set = PETSC_FALSE;

  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = PermonInitialize(&argc,&argv,(char *)0,help); if (ierr) return ierr;

  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_training",training_file,sizeof(training_file),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_kernel",kernel_file,sizeof(kernel_file),&kernel_file_set));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_test",test_file,sizeof(test_file),&test_file_set));

  /* Create SVM object */
  PetscCall(SVMCreate(PETSC_COMM_WORLD,&svm));
  PetscCall(SVMSetType(svm,SVM_BINARY));
  PetscCall(SVMSetFromOptions(svm));

  /* Load training dataset */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,training_file,FILE_MODE_READ,&viewer));
  PetscCall(SVMLoadTrainingDataset(svm,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* Load Gramian matrix */
  if (kernel_file_set) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,kernel_file,FILE_MODE_READ,&viewer));
    PetscCall(SVMLoadGramian(svm,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* Load test dataset */
  if (test_file_set) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,test_file,FILE_MODE_READ,&viewer));
    PetscCall(SVMLoadTestDataset(svm,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* Train classification model */
  PetscCall(SVMTrain(svm));
  /* Test performance of SVM model */
  if (test_file_set) {
    PetscCall(SVMTest(svm));
  }

  /* Free memory */
  PetscCall(SVMDestroy(&svm));
  ierr = PermonFinalize();
  return ierr;
}

/*TEST

  test:
    filter: grep -v MPI
    args: -qps_view_convergence -svm_view -svm_view_score
    args: -f_training $PERMON_SVM_DIR/src/tutorials/data/heart_scale.bin -f_kernel $PERMON_SVM_DIR/src/tutorials/data/heart_scale.kernel.bin
    args: -f_test $PERMON_SVM_DIR/src/tutorials/data/heart_scale.t.bin
    output_file: output/exbinfile_1.out
TEST*/
