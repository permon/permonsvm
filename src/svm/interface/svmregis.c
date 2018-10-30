
#include <permon/private/svmimpl.h>

FLLOP_EXTERN PetscErrorCode SVMCreate_Binary(SVM);

#undef __FUNCT__
#define __FUNCT__ "SVMRegisterAll"
PetscErrorCode SVMRegisterAll()
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMRegister"
PetscErrorCode SVMRegister(const char sname[],PetscErrorCode (*function)(SVM))
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
