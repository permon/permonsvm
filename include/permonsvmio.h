#if !defined(__PERMONSVMIO_H)
#define	__PERMONSVMIO_H

#include <permonsvm.h>

#define SVM_TRAINING_DATASET "Training dataset"
#define SVM_TEST_DATASET     "Test dataset"

FLLOP_EXTERN PetscErrorCode SVMLoadDataset_SVMLight(SVM svm,const char *,Mat *,Vec *);
FLLOP_EXTERN PetscErrorCode SVMLoadTestDataset(SVM svm,const char *);
FLLOP_EXTERN PetscErrorCode SVMLoadTrainingDataset(SVM,const char *);
#endif //__PERMONSVMIO_H
