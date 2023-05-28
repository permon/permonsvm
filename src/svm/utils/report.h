
#if !defined(__REPORT_H)
#define	__REPORT_H

#include <permonsvm.h>

FLLOP_EXTERN PetscErrorCode SVMGetBinaryClassificationReport(SVM,Vec,Vec,PetscInt *,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMPrintBinaryClassificationReport(SVM,PetscInt *,PetscReal *,PetscViewer);

#endif //__REPORT_H
