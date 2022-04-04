
static char help[] = "Trains binary SVM classification model. Predictions on training and test datasets are printed into\
stdout then.\n\
Input parameters:\n\
  -f       : training dataset file\n\
  -f_test  : test dataset file\n\
  -view_training_predictions: print predictions on training dataset into stdout\n\
  -view_test_predictions    : print predictions on test dataset into stdout\n";

#include <permonsvm.h>

int main(int argc,char **argv)
{
  SVM            svm;

  char           training_file[PETSC_MAX_PATH_LEN] = "data/heart_scale.bin";
  char           test_file[PETSC_MAX_PATH_LEN]     = "";
  PetscBool      test_file_set = PETSC_FALSE;
  PetscBool      training_result_view=PETSC_FALSE,test_result_view=PETSC_FALSE;

  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = PermonInitialize(&argc,&argv,(char *)0,help); if (ierr) return ierr;

  ierr = PetscOptionsGetString(NULL,NULL,"-f_training",training_file,sizeof(training_file),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f_test",test_file,sizeof(test_file),&test_file_set);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-view_training_predictions",&training_result_view);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-view_test_predictions",&test_result_view);CHKERRQ(ierr);

  /* Create SVM object */
  ierr = SVMCreate(PETSC_COMM_WORLD,&svm);CHKERRQ(ierr);
  ierr = SVMSetType(svm,SVM_BINARY);CHKERRQ(ierr);
  ierr = SVMSetFromOptions(svm);CHKERRQ(ierr);

  /* Load training dataset */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,training_file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = SVMLoadTrainingDataset(svm,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* Load test dataset */
  if (test_file_set) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,test_file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = SVMLoadTestDataset(svm,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* Train classification model */
  ierr = SVMTrain(svm);CHKERRQ(ierr);
  /* Test performance of SVM model */
  if (test_file_set) {
    ierr = SVMTest(svm);CHKERRQ(ierr);
  }

  /* Print predictions on training samples into stdout */
  if (training_result_view) {
    ierr = SVMViewTrainingPredictions(svm,NULL);CHKERRQ(ierr);
  }

  /* Print predictions on test samples  into stdout */
  if (test_file_set && test_result_view) {
    ierr = SVMViewTestPredictions(svm,NULL);CHKERRQ(ierr);
  }

  /* Free memory */
  ierr = SVMDestroy(&svm);CHKERRQ(ierr);
  ierr = PermonFinalize();
  return ierr;
}

/*TEST

  test:
    filter: grep -v MPI
    args: -qps_view_convergence -svm_view -svm_view_score
    args: -f_training $PERMON_SVM_DIR/src/tutorials/data/heart_scale.bin
    args: -f_test $PERMON_SVM_DIR/src/tutorials/data/heart_scale.t.bin
    args: -view_training_predictions -view_test_predictions
    output_file: output/ex3.out
TEST*/
