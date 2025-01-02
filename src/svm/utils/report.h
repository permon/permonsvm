#pragma once

#include <permonsvm.h>

PERMON_EXTERN PetscErrorCode SVMGetBinaryClassificationReport(SVM,Vec,Vec,PetscInt *,PetscReal *);
PERMON_EXTERN PetscErrorCode SVMViewBinaryClassificationReport(SVM,PetscInt *,PetscReal *,PetscViewer);
