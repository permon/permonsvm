#if !defined(__PERMONSVMIO_H)
#define	__PERMONSVMIO_H

#include <permonsvm.h>

FLLOP_EXTERN PetscErrorCode PermonSVMLoadData(MPI_Comm,const char *,Mat *,Vec *);

#endif //__PERMONSVMIO_H
