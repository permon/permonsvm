
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
  TRY( PetscClassIdRegister("SVM",&SVM_CLASSID) );
  /* Register constructors */
  TRY( SVMRegisterAll() );
  /* Register Events */
  TRY( PetscLogEventRegister("SVMLoadDataset",SVM_CLASSID,&SVM_LoadDataset) );
  TRY( PetscLogEventRegister("SVMLoadGramian",SVM_CLASSID,&SVM_LoadGramian) );

  TRY( PetscRegisterFinalize(SVMFinalizePackage) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMFinalizePackage"
PetscErrorCode SVMFinalizePackage()
{

  PetscFunctionBegin;
  if (SVMPackageInitialized) {
    TRY( PetscFunctionListDestroy(&SVMList) );
  }

  SVMPackageInitialized = PETSC_FALSE;
  SVMRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
