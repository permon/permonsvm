#if !defined(__PERMONSVMIO_H)
#define	__PERMONSVMIO_H

#include <permonsvm.h>

#define SVM_TRAINING_DATASET "Training dataset"

FLLOP_EXTERN PetscErrorCode SVMLoadData(SVM svm,const char *,Mat *,Vec *);

FLLOP_EXTERN PetscErrorCode SVMLoadTrainingDataset(SVM,const char *);
#endif //__PERMONSVMIO_H
