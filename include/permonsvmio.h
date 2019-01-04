#if !defined(__PERMONSVMIO_H)
#define	__PERMONSVMIO_H

#include <permonsvm.h>

FLLOP_EXTERN PetscErrorCode SVMLoadData(SVM svm,const char *,Mat *,Vec *);

FLLOP_EXTERN PetscErrorCode SVMLoadTrainingDataset(SVM,const char *);
#endif //__PERMONSVMIO_H
