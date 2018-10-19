#include <permonsvmio.h>

#undef __FUNCT__
#define __FUNCT__ "PermonSVMLoadData"
PetscErrorCode PermonSVMLoadData(MPI_Comm comm,const char *filename,Mat *Xt,Vec *y) {

  PetscFunctionBeginI;
  *Xt = NULL;
  *y = NULL;
  PetscFunctionReturnI(0);
}
