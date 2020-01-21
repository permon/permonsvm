#include <permonsvm.h>

static char help[] = "Trains binary SVM classification model that hyper-parameter optimization (grid-search combined with cross-validation) is performed.\n\
Input parameters:\n\
  -f       : training dataset file\n\
  -f_test  : test dataset file\n";

int main(int argc,char **argv)
{
  SVM            svm;

  char           training_file[PETSC_MAX_PATH_LEN] = "data/heart_scale.bin";
  char           test_file[PETSC_MAX_PATH_LEN]     = "data/heart_scale.t.bin";

  PetscViewer    viewer;
  PetscBool      test_file_set = PETSC_FALSE;

  PetscErrorCode ierr;

  ierr = PermonInitialize(&argc,&argv,(char *)0,help); if (ierr) return ierr;

  ierr = PetscOptionsGetString(NULL,NULL,"-f_training",training_file,sizeof(training_file),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f_test",test_file,sizeof(test_file),&test_file_set);CHKERRQ(ierr);

  /* Create SVM object */
  ierr = SVMCreate(PETSC_COMM_WORLD,&svm);CHKERRQ(ierr);
  ierr = SVMSetType(svm,SVM_BINARY);CHKERRQ(ierr);

  /* Set hyper-parameter optimization will be performed */
  ierr = SVMSetHyperOpt(svm,PETSC_TRUE);CHKERRQ(ierr);

  /* Set k-fold cross-validation on 3 folds */
  ierr = SVMSetCrossValidationType(svm,CROSS_VALIDATION_KFOLD);CHKERRQ(ierr);
  ierr = SVMSetNfolds(svm,3);CHKERRQ(ierr);

  /* Perform grid-search on S = {3^-2, 3^-1.5, ... 3^-0.5, 1, 3^0.5 ..., 3^1.5, 3^2} */
  ierr = SVMSetPenaltyType(svm,1);CHKERRQ(ierr);
  ierr = SVMGridSearchSetBaseLogC(svm,3);CHKERRQ(ierr);
  ierr = SVMGridSearchSetStrideLogC(svm,-2,2,0.5);CHKERRQ(ierr);

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

  /* Train SVM model */
  ierr = SVMTrain(svm);CHKERRQ(ierr);
  /* Test performance of SVM model */
  if (test_file_set) {
    ierr = SVMTest(svm);CHKERRQ(ierr);
  }

  /* Free memory */
  ierr = SVMDestroy(&svm);CHKERRQ(ierr);
  ierr = PermonFinalize();
  return ierr;
}

/*TEST

  test:
    filter: grep -v MPI
    args: -f_training $PERMON_SVM_DIR/examples/data/heart_scale.bin -f_test $PERMON_SVM_DIR/examples/data/heart_scale.t.bin
    args: -cross_svm_view -cross_svm_view_score -cross_qps_view_convergence
    args: -svm_view -svm_view_score -qps_view_convergence
    output_file: output/ex2.out
TEST*/
