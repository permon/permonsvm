#include <permon/private/svmimpl.h>

PERMON_EXTERN PetscErrorCode SVMCreate_Binary(SVM);
PERMON_EXTERN PetscErrorCode SVMCreate_Probability(SVM);

/*
   Contains the list of registered Create routines of all SVM types
*/
PetscFunctionList SVMList = 0;
PetscBool SVMRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "SVMRegisterAll"
PetscErrorCode SVMRegisterAll()
{

  PetscFunctionBegin;
  if (SVMRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  SVMRegisterAllCalled = PETSC_TRUE;

  PetscCall(SVMRegister(SVM_BINARY,SVMCreate_Binary));
  PetscCall(SVMRegister(SVM_PROBABILITY,SVMCreate_Probability));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMRegister"
PetscErrorCode SVMRegister(const char sname[],PetscErrorCode (*function)(SVM))
{

  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&SVMList,sname,function));
  PetscFunctionReturn(PETSC_SUCCESS);
}
