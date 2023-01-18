
static char help[] = "Trains binary L2-loss SVM classification model that training process was stopped using terminating condition\
based on duality gap.\n\
Input parameters:\n\
  -f       : training dataset file\n\
  -f_test  : test dataset file\n\
  -svm_binary_convergence: duality_gap\n\
";

#include <permonsvm.h>

int main(int argc,char **argv)
{
  SVM         svm;

  char        training_file[PETSC_MAX_PATH_LEN] = "data/heart_scale.bin";
  char        test_file[PETSC_MAX_PATH_LEN]     = "data/heart_scale.t.bin";

  PetscViewer viewer;

  PetscCall(PermonInitialize(&argc,&argv,(char *)0,help));

  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_training",training_file,sizeof(training_file),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_test",test_file,sizeof(test_file),NULL));

  /* Create SVM object */
  PetscCall(SVMCreate(PETSC_COMM_WORLD,&svm));
  PetscCall(SVMSetType(svm,SVM_BINARY));
  PetscCall(SVMSetFromOptions(svm));

  /* Load training dataset */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,training_file,FILE_MODE_READ,&viewer));
  PetscCall(SVMLoadTrainingDataset(svm,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* Load test dataset */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,test_file,FILE_MODE_READ,&viewer));
  PetscCall(SVMLoadTestDataset(svm,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* Train classification model */
  PetscCall(SVMTrain(svm));
  /* Test performance of SVM model */
  PetscCall(SVMTest(svm));

  /* Free memory */
  PetscCall(SVMDestroy(&svm));
  PetscCall(PermonFinalize());

  return 0;
}

/*TEST

  test:
    suffix: duality_gap
    filter: grep -v MPI
    args: -qps_view_convergence -svm_view -svm_view_score -svm_binary_convergence_test duality_gap
    args: -svm_loss_type L2
    args: -f_training $PERMON_SVM_DIR/src/tutorials/data/heart_scale.bin
    args: -f_test $PERMON_SVM_DIR/src/tutorials/data/heart_scale.t.bin
    output_file: output/ex4_duality_gap.out

  test:
    suffix: dual_violation
    filter: grep -v MPI
    args: -qps_view_convergence -svm_view -svm_view_score -svm_binary_convergence_test dual_violation
    args: -svm_loss_type L1 -svm_binary_mod 2 -qps_atol 1e-2
    args: -f_training $PERMON_SVM_DIR/src/tutorials/data/heart_scale.bin
    args: -f_test $PERMON_SVM_DIR/src/tutorials/data/heart_scale.t.bin
    output_file: output/ex4_dual_violation.out
TEST*/
