
static char help[] = "";

#include <permonsvm.h>

int main(int argc,char **argv)
{
  SVM svm;

  PetscCall(PermonInitialize(&argc,&argv,(char *)0,help));

  /* Create SVM object */
  PetscCall(SVMCreate(PETSC_COMM_WORLD,&svm));
  PetscCall(SVMSetType(svm,SVM_PROBABILITY));

  /* Free memory */
  PetscCall(SVMDestroy(&svm));
  PetscCall(PermonFinalize());

  return 0;
}

/*TEST

  test:

    args: -svm_loss_type L2

TEST*/
