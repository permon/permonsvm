#include <permon/private/svmimpl.h>

PERMON_EXTERN PetscErrorCode SVMCreate_Binary(SVM);
PERMON_EXTERN PetscErrorCode SVMCreate_Probability(SVM);

/*
   Contains the list of registered Create routines of all SVM types
*/
PetscFunctionList SVMList              = 0;
PetscBool         SVMRegisterAllCalled = PETSC_FALSE;

PetscErrorCode SVMRegisterAll()
{
  PetscFunctionBegin;
  if (SVMRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  SVMRegisterAllCalled = PETSC_TRUE;

  PetscCall(SVMRegister(SVM_BINARY, SVMCreate_Binary));
  PetscCall(SVMRegister(SVM_PROBABILITY, SVMCreate_Probability));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVMRegister(const char sname[], PetscErrorCode (*function)(SVM))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&SVMList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}
