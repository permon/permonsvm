
#include <permon/private/svmimpl.h>

static PetscBool SVMPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PermonSVMInitializePackage"
PetscErrorCode PermonSVMInitializePackage() {
    PetscFunctionBegin;
    if (SVMPackageInitialized) PetscFunctionReturn(0);
    SVMPackageInitialized = PETSC_TRUE;
    
    /* Register Classes */
    TRY( PetscClassIdRegister("SVM Problem", &SVM_CLASSID) );
    
    PetscFunctionReturn(0);
}
