#include <permonsvm.h>

static char help[] = "Trains binary SVM classification model that hyper-parameter optimization (grid-search combined with cross-validation) is performed.\n\
Input parameters:\n\
  -f       : training dataset file\n\
  -f_test  : test dataset file\n";

int main(int argc, char **argv)
{
  SVM svm;

  char training_file[PETSC_MAX_PATH_LEN] = "data/heart_scale.bin";
  char test_file[PETSC_MAX_PATH_LEN]     = "data/heart_scale.t.bin";

  PetscViewer viewer;
  PetscBool   test_file_set = PETSC_FALSE;

  PetscCall(PermonInitialize(&argc, &argv, (char *)0, help));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-f_training", training_file, sizeof(training_file), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f_test", test_file, sizeof(test_file), &test_file_set));

  /* Create SVM object */
  PetscCall(SVMCreate(PETSC_COMM_WORLD, &svm));
  PetscCall(SVMSetType(svm, SVM_BINARY));

  /* Set hyper-parameter optimization will be performed */
  PetscCall(SVMSetHyperOpt(svm, PETSC_TRUE));

  /* Set k-fold cross-validation on 3 folds */
  PetscCall(SVMSetCrossValidationType(svm, CROSS_VALIDATION_KFOLD));
  PetscCall(SVMSetNfolds(svm, 3));

  /* Perform grid-search on S = {3^-2, 3^-1.5, ... 3^-0.5, 1, 3^0.5 ..., 3^1.5, 3^2} */
  PetscCall(SVMSetPenaltyType(svm, 1));
  PetscCall(SVMGridSearchSetBaseLogC(svm, 3));
  PetscCall(SVMGridSearchSetStrideLogC(svm, -2, 2, 0.5));

  PetscCall(SVMSetFromOptions(svm));

  /* Load training dataset */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, training_file, FILE_MODE_READ, &viewer));
  PetscCall(SVMLoadTrainingDataset(svm, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* Load test dataset */
  if (test_file_set) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, test_file, FILE_MODE_READ, &viewer));
    PetscCall(SVMLoadTestDataset(svm, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* Train SVM model */
  PetscCall(SVMTrain(svm));
  /* Test performance of SVM model */
  if (test_file_set) { PetscCall(SVMTest(svm)); }

  /* Free memory */
  PetscCall(SVMDestroy(&svm));
  PetscCall(PermonFinalize());

  return 0;
}

/*TEST

  test:
    filter: grep -v MPI
    args: -f_training $PERMON_SVM_DIR/src/tutorials/data/heart_scale.bin -f_test $PERMON_SVM_DIR/src/tutorials/data/heart_scale.t.bin
    args: -cross_svm_view -cross_svm_view_report -cross_qps_view_convergence
    args: -svm_view -svm_view_report -qps_view_convergence
    output_file: output/ex2.out
TEST*/
