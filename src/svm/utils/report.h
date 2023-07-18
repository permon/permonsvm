
#if !defined(__PERMONSVMREPORT_H)
#define	__PERMONSVMREPORT_H

#include <permonsvm.h>

FLLOP_EXTERN PetscErrorCode SVMGetBinaryClassificationReport(SVM,Vec,Vec,PetscInt *,PetscReal *);
FLLOP_EXTERN PetscErrorCode SVMViewBinaryClassificationReport(SVM,PetscInt *,PetscReal *,PetscViewer);

#endif //__REPORT_H
