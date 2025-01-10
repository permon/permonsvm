#pragma once

#include <permonsvm.h>

PERMON_EXTERN PetscErrorCode DatasetLoad_SVMLight(Mat, Vec, PetscViewer);
PERMON_EXTERN PetscErrorCode DatasetLoad_Binary(Mat, Vec, PetscViewer);
