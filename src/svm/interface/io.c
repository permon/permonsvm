#include <petsc/private/petscimpl.h>
#include <permonsvmio.h>

#include <limits.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif

#define PERMON_DARRAY_INIT_CAPACITY 4
#define PERMON_DARRAY_GROW_FACTOR 2.0

#define PermonDynamicArray_(type) \
  {                               \
    type      *data;              \
    PetscInt  capacity;           \
    PetscInt  size;               \
    PetscReal grow_factor;        \
  }

#define PermonDynamicArrayInit(a,_capacity_,_grow_factor_)      \
  TRY( PetscMalloc(_capacity_ * sizeof(*(a.data)),&(a.data)) ); \
  a.size = 0;                                                   \
  a.capacity = _capacity_;                                      \
  a.grow_factor = _grow_factor_;

#define PermonDynamicArrayClear(a) \
  TRY( PetscFree(a.data) );        \
  a.capacity = 0;                  \
  a.size = 0;

#define PermonDynamicArrayPushBack(a,v)                         \
  if (a.capacity == a.size) PermonDynamicArrayResize(a);        \
  a.data[a.size] = v;                                       \
  ++a.size;

#define PermonDynamicArrayResize(a) \
  a.capacity = a.grow_factor * a.capacity; \
  TRY( PetscRealloc(a.capacity * sizeof(*(a.data)),&a.data) ); \

#define PermonDynamicArrayAddValue(a,v) \
  PetscInt i; \
  for (i = 0; i < a.size; ++i) a.data[i] += v;

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
#define __FUNCT__ "PermonSVMParseBuffer"
PetscErrorCode PermonSVMParseBuffer(MPI_Comm comm,char *buff,struct ArrInt *i,struct ArrInt *j,struct ArrReal *a,struct ArrInt *k,struct ArrReal *y,PetscInt *N) {
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
  array_init_capacity = PERMON_DARRAY_INIT_CAPACITY;
  array_grow_factor = PERMON_DARRAY_GROW_FACTOR;

  TRY( PetscOptionsGetInt(NULL,NULL,"-svm_io_darray_init_size",&array_init_capacity,NULL) );
  if (array_init_capacity <= 0) FLLOP_SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Initial size of dynamic array must be greater than zero");

  TRY( PetscOptionsGetReal(NULL,NULL,"-svm_io_darray_grow_factor",&array_grow_factor,NULL) );
  if (array_grow_factor <= 1.) FLLOP_SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Grow factor of dynamic array must be greater than one");

  PermonDynamicArrayInit(i_in,array_init_capacity,array_grow_factor);
  PermonDynamicArrayPushBack(i_in,0);

  if (buff) {
    PermonDynamicArrayInit(y_in,array_init_capacity,array_grow_factor);
    PermonDynamicArrayInit(k_in,array_init_capacity,array_grow_factor);

    PermonDynamicArrayInit(j_in,array_init_capacity,array_grow_factor);
    PermonDynamicArrayInit(a_in,array_init_capacity,array_grow_factor);

    N_in = 0;
#if (_POSIX_VERSION >= 200112L)
    line = strtok_r(buff,"\n",&ptr_line);

    while (line) {
      word = strtok_r(line," ",&ptr_word);
      yi = (PetscReal) atoi(word);

      PermonDynamicArrayPushBack(k_in,y_in.size);
      PermonDynamicArrayPushBack(y_in,yi);

      while ((word = strtok_r(NULL," ",&ptr_word))) {
        key = strtok(word,":");
        col = (PetscInt) atoi(key);

        if (col > N_in) N_in = col;
        col -= 1; /*Column indices start from 1 in SVMLight format*/

        PermonDynamicArrayPushBack(j_in,col);

        v = strtok(NULL,":");
        value = (PetscReal) atof(v);
        PermonDynamicArrayPushBack(a_in,value);
      }
      PermonDynamicArrayPushBack(i_in,a_in.size);

      line = strtok_r(NULL,"\n",&ptr_line);
    }
#else
    TRY( PetscTokenCreate(buff,'\n',&token_line) );
    while(PETSC_TRUE) {
      TRY( PetscTokenFind(token_line,&line) );
      if (line) {
        TRY( PetscTokenCreate(line,' ',&token_word) );

        TRY( PetscTokenFind(token_word,&word) );
        yi = (PetscReal) atof(word);

        PermonDynamicArrayPushBack(k_in,y_in.size);
        PermonDynamicArrayPushBack(y_in,yi);

        while (PETSC_TRUE) {
          TRY( PetscTokenFind(token_word,&word) );
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
      TRY( PetscTokenDestroy(&token_word) );
    }
    TRY( PetscTokenDestroy(&token_line) );
#endif
  } else {
    y_in.data = NULL;
    k_in.data = NULL;
    j_in.data = NULL;
    a_in.data = NULL;
  }

  TRY( MPI_Allreduce(MPI_IN_PLACE,&N_in,1,MPIU_INT,MPI_MAX,comm) );

  *i = i_in;
  *j = j_in;
  *a = a_in;
  *k = k_in;
  *y = y_in;
  *N = N_in;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMPAsseblyMatVec"
PetscErrorCode PermonSVMAsseblyMatVec(MPI_Comm comm,char *buff,Mat *Xt,Vec *labels) {
  struct ArrInt  i,j,k;
  struct ArrReal a,y;

  PetscInt offset,m,N;

  PetscFunctionBegin;
  TRY( PermonSVMParseBuffer(comm,buff,&i,&j,&a,&k,&y,&N) );

  m = (buff) ? i.size - 1 : 0;
  /*local to global: label vector indices*/
  offset = (k.data) ? k.size : 0;
  TRY( MPI_Scan(MPI_IN_PLACE,&offset,1,MPIU_INT,MPI_SUM,comm) );
  if (k.data) {
    offset -= k.size;
    PermonDynamicArrayAddValue(k,offset);
  }

  TRY( MatCreateMPIAIJWithArrays(comm,m,PETSC_DETERMINE,PETSC_DETERMINE,N,i.data,j.data,a.data,Xt) );
  TRY( MatAssemblyBegin(*Xt,MAT_FINAL_ASSEMBLY) );
  TRY( MatAssemblyEnd(*Xt,MAT_FINAL_ASSEMBLY) );
  
  TRY( MatCreateVecs(*Xt,NULL,labels) );
  if (y.data) TRY( VecSetValues(*labels,y.size,k.data,y.data,INSERT_VALUES) );
  TRY( VecAssemblyBegin(*labels) );
  TRY( VecAssemblyEnd(*labels) );
  
  if (y.data) PermonDynamicArrayClear(y);
  if (k.data) PermonDynamicArrayClear(k);
  PermonDynamicArrayClear(i);
  if (j.data) PermonDynamicArrayClear(j);
  if (a.data) PermonDynamicArrayClear(a);

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
