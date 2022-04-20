
#include <permon/private/svmimpl.h>

FLLOP_EXTERN PetscErrorCode SVMCreate_Binary(SVM);

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
  if (SVMRegisterAllCalled) PetscFunctionReturn(0);
  SVMRegisterAllCalled = PETSC_TRUE;

  PetscCall(SVMRegister(SVM_BINARY,SVMCreate_Binary));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMRegister"
PetscErrorCode SVMRegister(const char sname[],PetscErrorCode (*function)(SVM))
{

  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&SVMList,sname,function));
  PetscFunctionReturn(0);
}
