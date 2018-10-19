#include <petsc/private/petscimpl.h>
#include <permonsvmio.h>
#include "ioutils.h"

#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif

struct ArrInt  PermonDynamicArray_(PetscInt);
struct ArrReal PermonDynamicArray_(PetscReal);

#undef __FUNCT__
#define __FUNCT__ "PermonSVMLoadBuffer"
PetscErrorCode PermonSVMReadBuffer(MPI_Comm comm,const char *filename,char **chunk_buff) {
  PetscMPIInt comm_size,comm_rank;

  MPI_File    fh;
  char        *chunk_buff_inner;

  PetscInt    chunk_size,chunk_size_overlaped,chunk_size_shrink,chunk_size_tmp,overlap;
  MPI_Offset  file_size,offset;
  PetscInt    eol_pos,eol_start;
  PetscBool   eol_found;

  PetscInt    p;

  PetscFunctionBegin;
  TRY( MPI_Comm_size(comm,&comm_size) );
  TRY( MPI_Comm_rank(comm,&comm_rank) );

  chunk_buff_inner = NULL;
  overlap = PETSC_DECIDE;
  eol_start = 0;
  eol_pos = 0;
  eol_found = PETSC_FALSE;

  TRY( PetscOptionsGetInt(NULL,NULL,"-svm_io_overlap",&overlap,NULL) );
  if (overlap < 0 && overlap != PETSC_DECIDE) FLLOP_SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Overlap must be greater or equal to zero");

  TRY( MPI_File_open(comm,filename,MPI_MODE_RDONLY,MPI_INFO_NULL,&fh) );
  TRY( MPI_File_get_size(fh,&file_size) );

  /* Determine overlap size of chunk buffers, and reading offsets */
  if (overlap == PETSC_DECIDE) {
    overlap = 200;
    while (file_size / comm_size < overlap) {
      overlap /= 10;
    }
    if (overlap == 0) overlap = 1;
  }

  chunk_size = file_size / comm_size + ((file_size % comm_size) > comm_rank);
  chunk_size_overlaped = chunk_size;
  if (comm_rank != comm_size-1) chunk_size_overlaped += overlap;
  {
    MPI_Offset chunk_size_o = chunk_size;
    TRY( MPI_Scan(&chunk_size_o,&offset,1,MPI_OFFSET,MPI_SUM,comm) );
  }
  offset -= chunk_size;

  /* Allocation of chunk buffers and reading appropriate part of file */
  TRY( PetscMalloc(chunk_size_overlaped * sizeof(char),&chunk_buff_inner) );

  TRY( MPI_File_read_at_all(fh,offset,chunk_buff_inner,(PetscMPIInt)chunk_size_overlaped,MPI_CHAR,MPI_STATUS_IGNORE) );

  /* Check EOL in end of buffers without overlaps */
  if (chunk_buff_inner[chunk_size-1] == '\n') eol_found = PETSC_TRUE;

  /* Determine EOLs in overlaps */
  chunk_size_tmp = chunk_size;
  if ((comm_rank != comm_size - 1) && !eol_found) {
    while (!eol_found) {

      for (p = chunk_size; p < chunk_size_overlaped; ++p) {
        ++eol_pos;
        if (chunk_buff_inner[p] == '\n') {
          eol_found = PETSC_TRUE;
          break;
        }
      }

      if (!eol_found) {
        chunk_size = chunk_size_overlaped;
        if ((chunk_size + overlap + offset) < file_size) {
          chunk_size_overlaped += overlap;
        } else {
          overlap = file_size - (chunk_size + offset);
          chunk_size_overlaped += overlap;

          eol_pos += overlap;
          eol_found = PETSC_TRUE;
        }

        TRY( PetscRealloc(chunk_size_overlaped * sizeof(char),&chunk_buff_inner) );
        TRY( MPI_File_read_at(fh,offset + chunk_size,&chunk_buff_inner[chunk_size],overlap,MPI_CHAR,MPI_STATUS_IGNORE) );
      }
    }
  }

  if (comm_rank != comm_size-1) { 
    TRY( MPI_Send(&eol_pos,1,MPIU_INT,comm_rank + 1,0,comm) ); 
  }
  if (comm_rank != 0) {
    TRY( MPI_Recv(&eol_start,1,MPIU_INT,comm_rank - 1,0,comm,MPI_STATUS_IGNORE) );
  }

  /* Buffer shrink */
  if (comm_rank == 0) {
    TRY( PetscRealloc((chunk_size_tmp+eol_pos+1) * sizeof(char),&chunk_buff_inner) );
    chunk_buff_inner[chunk_size_tmp+eol_pos] = 0;
  } else if ((chunk_size_shrink = chunk_size_tmp + eol_pos - eol_start) != 0) {
    TRY( PetscMemmove(chunk_buff_inner,chunk_buff_inner + eol_start,chunk_size_shrink * sizeof(char)) );
    TRY( PetscRealloc((chunk_size_tmp+eol_pos-eol_start+1) * sizeof(char),&chunk_buff_inner) );
    chunk_buff_inner[chunk_size_shrink] = 0;
  } else {
    TRY( PetscFree(chunk_buff_inner) );
    chunk_buff_inner = NULL;
  }

  *chunk_buff = chunk_buff_inner;

  TRY( MPI_File_close(&fh) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMPAsseblyMatVec"
PetscErrorCode PermonSVMAsseblyMatVec(MPI_Comm comm,char *buff,Mat *Xt,Vec *labels) {

  PetscFunctionBegin;
  *Xt = NULL;
  *labels = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMLoadData"
PetscErrorCode PermonSVMLoadData(MPI_Comm comm,const char *filename,Mat *Xt,Vec *y) {
  char *chunk_buff = NULL;

  PetscFunctionBeginI;
  PetscValidPointer(Xt,3);
  PetscValidPointer(y,4);

  TRY( PermonSVMReadBuffer(comm,filename,&chunk_buff) );
  TRY( PermonSVMAsseblyMatVec(comm,chunk_buff,Xt,y) );

  if (chunk_buff) TRY( PetscFree(chunk_buff) );

  PetscFunctionReturnI(0);
}
