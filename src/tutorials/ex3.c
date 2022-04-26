
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

  PetscCall(PermonInitialize(&argc,&argv,(char *)0,help));

  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_training",training_file,sizeof(training_file),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_test",test_file,sizeof(test_file),&test_file_set));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-view_training_predictions",&training_result_view));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-view_test_predictions",&test_result_view));

  /* Create SVM object */
  PetscCall(SVMCreate(PETSC_COMM_WORLD,&svm));
  PetscCall(SVMSetType(svm,SVM_BINARY));
  PetscCall(SVMSetFromOptions(svm));

  /* Load training dataset */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,training_file,FILE_MODE_READ,&viewer));
  PetscCall(SVMLoadTrainingDataset(svm,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

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

  /* Print predictions on training samples into stdout */
  if (training_result_view) {
    PetscCall(SVMViewTrainingPredictions(svm,NULL));
  }

  /* Print predictions on test samples  into stdout */
  if (test_file_set && test_result_view) {
    PetscCall(SVMViewTestPredictions(svm,NULL));
  }

  /* Free memory */
  PetscCall(SVMDestroy(&svm));
  PetscCall(PermonFinalize());

  return 0;
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
