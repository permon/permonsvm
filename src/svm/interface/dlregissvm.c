
#include <permon/private/svmimpl.h>

static PetscBool SVMPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "SVMInitializePackage"
PetscErrorCode SVMInitializePackage()
{

  PetscFunctionBegin;
  if (SVMPackageInitialized) PetscFunctionReturn(0);
  SVMPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  PetscCall(PetscClassIdRegister("SVM",&SVM_CLASSID));
  /* Register constructors */
  PetscCall(SVMRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("SVMLoadDataset",SVM_CLASSID,&SVM_LoadDataset));
  PetscCall(PetscLogEventRegister("SVMLoadGramian",SVM_CLASSID,&SVM_LoadGramian));

  PetscCall(PetscRegisterFinalize(SVMFinalizePackage));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMFinalizePackage"
PetscErrorCode SVMFinalizePackage()
{

  PetscFunctionBegin;
  if (SVMPackageInitialized) {
    PetscCall(PetscFunctionListDestroy(&SVMList));
  }

  SVMPackageInitialized = PETSC_FALSE;
  SVMRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
