
static char help[] = "";

#include <permonsvm.h>

int main(int argc,char **argv)
{
  SVM         svm;

  // TODO change to data set containing calibration data set
  char        training_file[PETSC_MAX_PATH_LEN] = "data/heart_scale.bin";
  char        calibration_file[PETSC_MAX_PATH_LEN] = "data/heart_scale.bin";
  char        test_file[PETSC_MAX_PATH_LEN] = "data/heart_scale.t.bin";

  PetscViewer viewer;

  PetscCall(PermonInitialize(&argc,&argv,(char *)0,help));

  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_training",training_file,sizeof(training_file),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_calibration",calibration_file,sizeof(calibration_file),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_test",test_file,sizeof(test_file),NULL));

  /* Create SVM object */
  PetscCall(SVMCreate(PETSC_COMM_WORLD,&svm));
  PetscCall(SVMSetType(svm,SVM_PROBABILITY));
  PetscCall(SVMSetFromOptions(svm));

  /* Load training data set */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,training_file,FILE_MODE_READ,&viewer));
  PetscCall(SVMLoadTrainingDataset(svm,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* Load calibration data set */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,calibration_file,FILE_MODE_READ,&viewer));
  PetscCall(SVMLoadCalibrationDataset(svm,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* Load test data set */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,test_file,FILE_MODE_READ,&viewer));
  PetscCall(SVMLoadTestDataset(svm,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(SVMTrain(svm));
  PetscCall(SVMTest(svm));

  /* Free memory */
  PetscCall(SVMDestroy(&svm));
  PetscCall(PermonFinalize());

  return 0;
}

/*TEST

  test:
    filter: grep -v MPI
    args: -uncalibrated_svm_loss_type L1 -svm_view_io
    args: -f_training $PERMON_SVM_DIR/src/tutorials/data/heart_scale.bin
    args: -f_calibration $PERMON_SVM_DIR/src/tutorials/data/heart_scale.bin
    args: -f_test $PERMON_SVM_DIR/src/tutorials/data/heart_scale.t.bin
    args: -tao_type nls tao_view -svm_view_report
    args: -view_test_predictions
    args: -svm_threshold 0.47
    output_file: output/ex5.out
TEST*/
