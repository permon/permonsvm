#pragma once

#include <permonsvm.h>

FLLOP_EXTERN PetscErrorCode SVMGetBinaryClassificationReport(SVM,Vec,Vec,PetscInt *,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMViewBinaryClassificationReport(SVM,PetscInt *,PetscReal *,PetscViewer);
