#include <petsc/private/petscimpl.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/vecimpl.h>

#include "io.h"

#include <limits.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif

#define DARRAY_INIT_CAPACITY 4
#define DARRAY_GROW_FACTOR 2.0

#define DynamicArray_(type) \
  {                               \
    type      *data;              \
    PetscInt  capacity;           \
    PetscInt  size;               \
    PetscReal grow_factor;        \
  }

#define DynamicArrayInit(a,_capacity_,_grow_factor_)      \
  PetscCall(PetscMalloc(_capacity_ * sizeof(*(a.data)),&(a.data))); \
  a.size = 0;                                                   \
  a.capacity = _capacity_;                                      \
  a.grow_factor = _grow_factor_;

#define DynamicArrayClear(a) \
  PetscCall(PetscFree(a.data)); \
  a.capacity = 0;           \
  a.size = 0;

#define DynamicArrayPushBack(a,v) \
  if (a.capacity == a.size) DynamicArrayResize(a); \
  a.data[a.size] = v;                              \
  ++a.size;

#define DynamicArrayResize(a) \
  a.capacity = a.grow_factor * a.capacity; \
  PetscCall(PetscRealloc(a.capacity * sizeof(*(a.data)),&a.data)); \

#define DynamicArrayAddValue(a,v) \
  PetscInt i; \
  for (i = 0; i < a.size; ++i) a.data[i] += v;

struct ArrInt  DynamicArray_(PetscInt);
struct ArrReal DynamicArray_(PetscReal);

#undef __FUNCT__
#define __FUNCT__ "IOReadBuffer_SVMLight_Private"
static PetscErrorCode IOReadBuffer_SVMLight_Private(MPI_Comm comm,const char *filename,char **chunk_buff)
{
  PetscMPIInt comm_size,comm_rank;

  MPI_File    fh;
  char        *chunk_buff_inner;

  PetscInt    chunk_size,chunk_size_overlaped,chunk_size_shrink,chunk_size_tmp,overlap;
  MPI_Offset  file_size,offset;
  PetscInt    eol_pos,eol_start;
  PetscBool   eol_found;

  PetscInt    p;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm,&comm_size));
  PetscCallMPI(MPI_Comm_rank(comm,&comm_rank));

  chunk_buff_inner = NULL;
  overlap = PETSC_DECIDE;
  eol_start = 0;
  eol_pos = 0;
  eol_found = PETSC_FALSE;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-svm_io_overlap",&overlap,NULL));
  if (overlap < 0 && overlap != PETSC_DECIDE) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Overlap must be greater or equal to zero");

  PetscCallMPI(MPI_File_open(comm,filename,MPI_MODE_RDONLY,MPI_INFO_NULL,&fh));
  PetscCallMPI(MPI_File_get_size(fh,&file_size));

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
    PetscCallMPI(MPI_Scan(&chunk_size_o,&offset,1,MPI_OFFSET,MPI_SUM,comm));
  }
  offset -= chunk_size;

  /* Allocation of chunk buffers and reading appropriate part of file */
  PetscCall(PetscMalloc(chunk_size_overlaped * sizeof(char),&chunk_buff_inner));

  PetscCallMPI(MPI_File_read_at_all(fh,offset,chunk_buff_inner,(PetscMPIInt)chunk_size_overlaped,MPI_CHAR,MPI_STATUS_IGNORE));

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

        PetscCall(PetscRealloc(chunk_size_overlaped * sizeof(char),&chunk_buff_inner));
        PetscCallMPI(MPI_File_read_at(fh,offset + chunk_size,&chunk_buff_inner[chunk_size],overlap,MPI_CHAR,MPI_STATUS_IGNORE));
      }
    }
  }

  if (comm_rank != comm_size-1) {
    PetscCallMPI(MPI_Send(&eol_pos,1,MPIU_INT,comm_rank + 1,0,comm));
  }
  if (comm_rank != 0) {
    PetscCallMPI(MPI_Recv(&eol_start,1,MPIU_INT,comm_rank - 1,0,comm,MPI_STATUS_IGNORE));
  }

  /* Buffer shrink */
  if (comm_rank == 0) {
    PetscCall(PetscRealloc((chunk_size_tmp+eol_pos+1) * sizeof(char),&chunk_buff_inner));
    chunk_buff_inner[chunk_size_tmp+eol_pos] = 0;
  } else if ((chunk_size_shrink = chunk_size_tmp + eol_pos - eol_start) != 0) {
    PetscCall(PetscMemmove(chunk_buff_inner,chunk_buff_inner + eol_start,chunk_size_shrink * sizeof(char)));
    PetscCall(PetscRealloc((chunk_size_tmp+eol_pos-eol_start+1) * sizeof(char),&chunk_buff_inner));
    chunk_buff_inner[chunk_size_shrink] = 0;
  } else {
    PetscCall(PetscFree(chunk_buff_inner));
    chunk_buff_inner = NULL;
  }

  *chunk_buff = chunk_buff_inner;

  PetscCallMPI(MPI_File_close(&fh));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "IOParseBuffer_SVMLight_Private"
static PetscErrorCode IOParseBuffer_SVMLight_Private(MPI_Comm comm,char *buff,struct ArrInt *i,struct ArrInt *j,struct ArrReal *a,struct ArrInt *k,struct ArrReal *y,PetscInt *N)
{
  struct ArrInt  i_in,j_in,k_in;
  struct ArrReal a_in,y_in;

  PetscInt   array_init_capacity;
  PetscReal  array_grow_factor;

  char       *line = NULL,*word = NULL,*key = NULL,*v = NULL;
#if (_POSIX_VERSION >= 200112L)
  char       *ptr_line = NULL,*ptr_word = NULL;
#else
  PetscToken token_line,token_word;
#endif

  PetscInt   col,N_in;
  PetscReal  value,yi;

  PetscFunctionBegin;
  array_init_capacity = DARRAY_INIT_CAPACITY;
  array_grow_factor = DARRAY_GROW_FACTOR;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-svm_io_darray_init_size",&array_init_capacity,NULL));
  if (array_init_capacity <= 0) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"Initial size of dynamic array must be greater than zero");

  PetscCall(PetscOptionsGetReal(NULL,NULL,"-svm_io_darray_grow_factor",&array_grow_factor,NULL));
  if (array_grow_factor <= 1.) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"Grow factor of dynamic array must be greater than one");

  DynamicArrayInit(i_in,array_init_capacity,array_grow_factor);
  DynamicArrayPushBack(i_in,0);

  if (buff) {
    DynamicArrayInit(y_in,array_init_capacity,array_grow_factor);
    DynamicArrayInit(k_in,array_init_capacity,array_grow_factor);

    DynamicArrayInit(j_in,array_init_capacity,array_grow_factor);
    DynamicArrayInit(a_in,array_init_capacity,array_grow_factor);

    N_in = 0;
#if (_POSIX_VERSION >= 200112L)
    line = strtok_r(buff,"\n",&ptr_line);

    while (line) {
      word = strtok_r(line," ",&ptr_word);
      yi = (PetscReal) atoi(word);

      DynamicArrayPushBack(k_in,y_in.size);
      DynamicArrayPushBack(y_in,yi);

      while ((word = strtok_r(NULL," ",&ptr_word))) {
        key = strtok(word,":");
        col = (PetscInt) atoi(key);

        if (col > N_in) N_in = col;
        col -= 1; /*Column indices start from 1 in SVMLight format*/

        DynamicArrayPushBack(j_in,col);

        v = strtok(NULL,":");
        value = (PetscReal) atof(v);
        DynamicArrayPushBack(a_in,value);
      }
      DynamicArrayPushBack(i_in,a_in.size);

      line = strtok_r(NULL,"\n",&ptr_line);
    }
#else
    PetscCall(PetscTokenCreate(buff,'\n',&token_line));
    while(PETSC_TRUE) {
      PetscCall(PetscTokenFind(token_line,&line));
      if (line) {
        PetscCall(PetscTokenCreate(line,' ',&token_word));

        PetscCall(PetscTokenFind(token_word,&word));
        yi = (PetscReal) atof(word);

        PermonDynamicArrayPushBack(k_in,y_in.size);
        PermonDynamicArrayPushBack(y_in,yi);

        while (PETSC_TRUE) {
          PetscCall(PetscTokenFind(token_word,&word));
          if (word) {
            key = strtok(word,":");
            col = (PetscInt) atoi(key);

            if (col > N_in) N_in = col;
            col -= 1; /*Column indices start from 1 in SVMLight format*/
            PermonDynamicArrayPushBack(j_in,col);

            v = strtok(NULL,":");
            value = (PetscReal) atof(v);
            PermonDynamicArrayPushBack(a_in,value);
          } else {
            break;
          }
        }
      } else {
        break;
      }
      PermonDynamicArrayPushBack(i_in,a_in.size);
      PetscCall(PetscTokenDestroy(&token_word));
    }
    PetscCall(PetscTokenDestroy(&token_line));
#endif
  } else {
    y_in.data = NULL;
    k_in.data = NULL;
    j_in.data = NULL;
    a_in.data = NULL;
  }

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,&N_in,1,MPIU_INT,MPI_MAX,comm));

  *i = i_in;
  *j = j_in;
  *a = a_in;
  *k = k_in;
  *y = y_in;
  *N = N_in;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "DatasetAssembly_SVMLight_Private"
static PetscErrorCode DatasetAssembly_SVMLight_Private(Mat Xt,Vec labels,char *buff)
{
  MPI_Comm comm;

  struct   ArrInt  i,j,k;
  struct   ArrReal a,y;

  PetscInt offset,m,N;
  PetscInt rbs;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) Xt,&comm));

  PetscCall(IOParseBuffer_SVMLight_Private(comm,buff,&i,&j,&a,&k,&y,&N));
  m = (buff) ? i.size - 1 : 0;
  /* local to global: label vector indices */
  offset = (k.data) ? k.size : 0;
  PetscCallMPI(MPI_Scan(MPI_IN_PLACE,&offset,1,MPIU_INT,MPI_SUM,comm));
  if (k.data) {
    offset -= k.size;
    DynamicArrayAddValue(k,offset);
  }

  /* Assembly the Xt matrix */
  PetscCall(MatSetType(Xt,MATAIJ));
  PetscCall(MatSetSizes(Xt,m,PETSC_DECIDE,PETSC_DECIDE,N));
  PetscCall(MatSeqAIJSetPreallocationCSR(Xt,i.data,j.data,a.data));
  PetscCall(MatMPIAIJSetPreallocationCSR(Xt,i.data,j.data,a.data));

  /* Assembly vector of labels */
  PetscCall(MatGetBlockSizes(Xt,&rbs,NULL));
  PetscCall(VecSetSizes(labels,Xt->rmap->n,PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(labels,rbs));
  PetscCall(VecSetType(labels,Xt->defaultvectype));
  PetscCall(PetscLayoutReference(Xt->rmap,&labels->map));

  if (y.data) PetscCall(VecSetValues(labels,y.size,k.data,y.data,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(labels));
  PetscCall(VecAssemblyEnd(labels));

  if (y.data) DynamicArrayClear(y);
  if (k.data) DynamicArrayClear(k);
  DynamicArrayClear(i);
  if (j.data) DynamicArrayClear(j);
  if (a.data) DynamicArrayClear(a);

  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "DatasetLoad_SVMLight"
PetscErrorCode DatasetLoad_SVMLight(Mat Xt,Vec y,PetscViewer v)
{
  MPI_Comm   comm;

  const char *file_name = NULL;
  char       *chunk_buff = NULL;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) Xt,&comm));
  PetscCall(PetscViewerFileGetName(v,&file_name));

  PetscCall(IOReadBuffer_SVMLight_Private(comm,file_name,&chunk_buff));
  PetscCall(DatasetAssembly_SVMLight_Private(Xt,y,chunk_buff));

  if (chunk_buff) { PetscCall(PetscFree(chunk_buff)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerSVMLightOpen"
PetscErrorCode PetscViewerSVMLightOpen(MPI_Comm comm,const char name[],PetscViewer *v)
{
  PetscViewer v_inner;

  PetscFunctionBegin;
  PetscCall(PetscViewerCreate(comm,&v_inner));
  PetscCall(PetscViewerSetType(v_inner,PETSCVIEWERASCII));
  PetscCall(PetscViewerFileSetMode(v_inner,FILE_MODE_READ));
  PetscCall(PetscViewerFileSetName(v_inner,name));

  *v = v_inner;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "DatasetLoad_Binary"
PetscErrorCode DatasetLoad_Binary(Mat Xt,Vec y,PetscViewer v)
{
  char       Xt_name[256];
  const char *Xt_name_tmp;
  char       y_name[256];
  const char *y_name_tmp;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetName((PetscObject) Xt,&Xt_name_tmp));
  PetscCall(PetscStrncpy(Xt_name,Xt_name_tmp,sizeof(Xt_name)));
  PetscCall(PetscObjectGetName((PetscObject) y,&y_name_tmp));
  PetscCall(PetscStrncpy(y_name,y_name_tmp,sizeof(y_name)));

  PetscCall(PetscObjectSetName((PetscObject) Xt,"X"));
  PetscCall(MatLoad(Xt,v));
  PetscCall(PetscObjectSetName((PetscObject) y,"y"));
  PetscCall(VecLoad(y,v));

  PetscCall(PetscObjectSetName((PetscObject) Xt,Xt_name));
  PetscCall(PetscObjectSetName((PetscObject) y,y_name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerLoadSVMDataset"
/*@
  PetscViewerLoadDataset - Loads dataset.

  Input Parameters:
- v - viewer

  Output Parameters:
+ Xt - matrix of samples
- y - known labels of samples

.seealso SVM
@*/
PetscErrorCode PetscViewerLoadSVMDataset(Mat Xt,Vec y,PetscViewer v)
{
  MPI_Comm   comm;
  const char *type_name = NULL;

  PetscBool  isascii,ishdf5,isbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Xt,MAT_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,2);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,3);
  PetscCheckSameComm(v,1,Xt,2);
  PetscCheckSameComm(v,1,y,3);

  PetscCall(PetscObjectTypeCompare((PetscObject) v,PETSCVIEWERASCII,&isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject) v,PETSCVIEWERHDF5,&ishdf5));
  PetscCall(PetscObjectTypeCompare((PetscObject) v,PETSCVIEWERBINARY,&isbinary));

  if (isascii) {
    PetscCall(DatasetLoad_SVMLight(Xt,y,v));
  } else if (ishdf5 || isbinary) {
    PetscCall(DatasetLoad_Binary(Xt,y,v));
  } else {
    PetscCall(PetscObjectGetComm((PetscObject) v,&comm));
    PetscCall(PetscObjectGetType((PetscObject) v,&type_name));

    SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported for PetscViewerLoadDataset",type_name);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
