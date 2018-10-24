#include <mpi.h>
#include <permonsvm.h>

static MPI_Comm    comm;
static PetscMPIInt comm_rank,comm_size;

#undef __FUNCT__
#define __FUNCT__ "PermonSVMRunBinaryClassification"
PetscErrorCode PermonSVMRunBinaryClassification() {
  
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  PermonInitialize(&argc,&argv,(char *) 0,(char *) 0);

  PetscFunctionBegin;
  comm = PETSC_COMM_WORLD;
  TRY(MPI_Comm_rank(comm,&comm_rank));
  TRY(MPI_Comm_size(comm,&comm_size));

  TRY(PetscPrintf(comm,"PETSC_DIR:\t%s\n",PETSC_DIR));
  TRY(PetscPrintf(comm,"PETSC_ARCH:\t%s\n",PETSC_ARCH));
#ifdef PETSC_RELEASE_DATE
#define DATE PETSC_RELEASE_DATE
#else
#define DATE PETSC_VERSION_DATE
#endif
  TRY(PetscPrintf(comm,"PETSc version:\t%d.%d.%d patch %d (%s)\n",PETSC_VERSION_MAJOR,PETSC_VERSION_MINOR,
                  PETSC_VERSION_SUBMINOR,PETSC_VERSION_PATCH,DATE));
#undef DATE
  TRY(PermonSVMRunBinaryClassification());

  TRY(PermonFinalize());
  PetscFunctionReturn(0);
}
