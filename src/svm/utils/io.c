#include <petsc/private/petscimpl.h>
#include <permonsvmio.h>

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
  TRY( PetscMalloc(_capacity_ * sizeof(*(a.data)),&(a.data)) ); \
  a.size = 0;                                                   \
  a.capacity = _capacity_;                                      \
  a.grow_factor = _grow_factor_;

#define DynamicArrayClear(a) \
  TRY( PetscFree(a.data) ); \
  a.capacity = 0;           \
  a.size = 0;

#define DynamicArrayPushBack(a,v) \
  if (a.capacity == a.size) DynamicArrayResize(a); \
  a.data[a.size] = v;                              \
  ++a.size;

#define DynamicArrayResize(a) \
  a.capacity = a.grow_factor * a.capacity; \
  TRY( PetscRealloc(a.capacity * sizeof(*(a.data)),&a.data) ); \

#define DynamicArrayAddValue(a,v) \
  PetscInt i; \
  for (i = 0; i < a.size; ++i) a.data[i] += v;

struct ArrInt  DynamicArray_(PetscInt);
struct ArrReal DynamicArray_(PetscReal);

#undef __FUNCT__
#define __FUNCT__ "SVMLoadBuffer"
PetscErrorCode SVMReadBuffer(MPI_Comm comm,const char *filename,char **chunk_buff) {
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
#define __FUNCT__ "SVMParseBuffer"
PetscErrorCode SVMParseBuffer(MPI_Comm comm,char *buff,struct ArrInt *i,struct ArrInt *j,struct ArrReal *a,struct ArrInt *k,struct ArrReal *y,PetscInt *N) {
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

  TRY( PetscOptionsGetInt(NULL,NULL,"-svm_io_darray_init_size",&array_init_capacity,NULL) );
  if (array_init_capacity <= 0) FLLOP_SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Initial size of dynamic array must be greater than zero");

  TRY( PetscOptionsGetReal(NULL,NULL,"-svm_io_darray_grow_factor",&array_grow_factor,NULL) );
  if (array_grow_factor <= 1.) FLLOP_SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Grow factor of dynamic array must be greater than one");

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
#define __FUNCT__ "SVMPAsseblyMatVec"
PetscErrorCode SVMAsseblyMatVec(MPI_Comm comm,char *buff,Mat *Xt,Vec *labels) {
  struct ArrInt  i,j,k;
  struct ArrReal a,y;

  PetscInt offset,m,N;

  PetscFunctionBegin;
  TRY( SVMParseBuffer(comm,buff,&i,&j,&a,&k,&y,&N) );

  m = (buff) ? i.size - 1 : 0;
  /*local to global: label vector indices*/
  offset = (k.data) ? k.size : 0;
  TRY( MPI_Scan(MPI_IN_PLACE,&offset,1,MPIU_INT,MPI_SUM,comm) );
  if (k.data) {
    offset -= k.size;
    DynamicArrayAddValue(k,offset);
  }

  TRY( MatCreate(comm,Xt) );
  TRY( MatSetType(*Xt,MATAIJ) );
  TRY( MatSetSizes(*Xt,m,PETSC_DECIDE,PETSC_DECIDE,N) );

  TRY( MatSeqAIJSetPreallocationCSR(*Xt,i.data,j.data,a.data) );
  TRY( MatMPIAIJSetPreallocationCSR(*Xt,i.data,j.data,a.data) );

  TRY( MatCreateVecs(*Xt,NULL,labels) );
  if (y.data) TRY( VecSetValues(*labels,y.size,k.data,y.data,INSERT_VALUES) );
  TRY( VecAssemblyBegin(*labels) );
  TRY( VecAssemblyEnd(*labels) );

  if (y.data) DynamicArrayClear(y);
  if (k.data) DynamicArrayClear(k);
  DynamicArrayClear(i);
  if (j.data) DynamicArrayClear(j);
  if (a.data) DynamicArrayClear(a);

  PetscFunctionReturn(0);
}

PetscErrorCode SVMDatasetInfo(Mat,Vec,PetscInt,PetscViewer);
PetscErrorCode SVMViewIO(SVM,const char *,const char *,PetscViewer);

#undef __FUNCT__
#define __FUNCT__ "SVMLoadData"
PetscErrorCode SVMLoadData(SVM svm,const char *filename,Mat *Xt,Vec *y) {
  MPI_Comm          comm;

  char              *chunk_buff = NULL;
  Mat               Xt_inner,Xt_biased;

  PetscInt          svm_mod;
  PetscReal         bias;

  PetscFunctionBegin;
  PetscValidPointer(Xt,3);
  PetscValidPointer(y,4);

  TRY( PetscObjectGetComm((PetscObject) svm,&comm) );

  TRY( SVMReadBuffer(comm,filename,&chunk_buff) );
  TRY( SVMAsseblyMatVec(comm,chunk_buff,&Xt_inner,y) );

  if (chunk_buff) TRY( PetscFree(chunk_buff) );

  TRY( SVMGetMod(svm,&svm_mod) );
  if (svm_mod == 2) {
    TRY( SVMGetBias(svm,&bias) );
    TRY( MatCreate_Biased(Xt_inner,bias,&Xt_biased) );
    Xt_inner = Xt_biased;
  }

  *Xt = Xt_inner;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMLoadTrainingDataset"
PetscErrorCode SVMLoadTrainingDataset(SVM svm,const char *filename)
{
  Mat Xt_training;
  Vec y_training;

  PetscBool view;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  TRY( SVMLoadData(svm,filename,&Xt_training,&y_training) );
  TRY( PetscObjectSetName((PetscObject) Xt_training,"Xt_training") );
  TRY( PetscObjectSetName((PetscObject) y_training,"y_training") );
  TRY( SVMSetTrainingDataset(svm,Xt_training,y_training) );

  TRY( PetscOptionsHasName(NULL,NULL,"-svm_view_io",&view) );
  if (view) {
    PetscViewer v = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject) svm) );
    TRY( SVMViewIO(svm,SVM_TRAINING_DATASET,filename,v) );
  }

  TRY( MatDestroy(&Xt_training) );
  TRY( VecDestroy(&y_training) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMLoadTestDataset"
PetscErrorCode SVMLoadTestDataset(SVM svm,const char *filename)
{
  Mat Xt_test;
  Vec y_test;

  PetscBool view;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  TRY( SVMLoadData(svm,filename,&Xt_test,&y_test) );
  TRY( PetscObjectSetName((PetscObject) Xt_test,"Xt_test") );
  TRY( PetscObjectSetName((PetscObject) y_test,"y_test") );
  TRY( SVMSetTestDataset(svm,Xt_test,y_test) );

  TRY( PetscOptionsHasName(NULL,NULL,"-svm_view_io",&view) );
  if (view) {
    PetscViewer v = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject) svm) );
    TRY( SVMViewIO(svm,SVM_TEST_DATASET,filename,v) );
  }

  TRY( MatDestroy(&Xt_test) );
  TRY( VecDestroy(&y_test) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMViewIO"
PetscErrorCode SVMViewIO(SVM svm,const char *dataset_type,const char *filename,PetscViewer v)
{
  Mat Xt;
  Vec y;

  PetscInt  svm_mod;

  PetscBool iseq;
  PetscBool isascii;

  PetscFunctionBegin;
  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&isascii) );

  if (isascii) {
    TRY( SVMGetMod(svm,&svm_mod) );

    TRY( PetscViewerASCIIPrintf(v, "=====================\n") );
    TRY( PetscObjectPrintClassNamePrefixType((PetscObject) svm,v) );

    TRY( PetscViewerASCIIPushTab(v) );
    TRY( PetscViewerASCIIPrintf(v,"%s from source \"%s\" was loaded successfully!\n",dataset_type,filename) );
    TRY( PetscViewerASCIIPrintf(v,"Dataset contains:\n") );

    TRY( PetscStrcmp(dataset_type,SVM_TRAINING_DATASET,&iseq) );
    if (iseq) {
      TRY( SVMGetTrainingDataset(svm,&Xt,&y) );
    }
    TRY( PetscStrcmp(dataset_type,SVM_TEST_DATASET,&iseq) );
    if (iseq) {
      TRY( SVMGetTestDataset(svm,&Xt,&y) );
    }

    TRY( PetscViewerASCIIPushTab(v) );
    TRY( SVMDatasetInfo(Xt,y,svm_mod,v) );
    TRY( PetscViewerASCIIPopTab(v) );

    TRY(PetscViewerASCIIPopTab(v));
    TRY(PetscViewerASCIIPrintf(v,"=====================\n"));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMViewDatasetInfo"
PetscErrorCode SVMDatasetInfo(Mat Xt,Vec y,PetscInt svm_mod,PetscViewer v)
{
  PetscInt    M,N,M_plus,M_minus;
  PetscReal   max,per_plus,per_minus;
  Vec         y_max;
  IS          is_plus;

  PetscFunctionBegin;
  TRY( MatGetSize(Xt,&M,&N) );

  TRY( VecDuplicate(y,&y_max) );
  TRY( VecMax(y,NULL,&max) );
  TRY( VecSet(y_max,max) );

  TRY( VecWhichEqual(y,y_max,&is_plus) );
  TRY( ISGetSize(is_plus,&M_plus) );

  per_plus = ((PetscReal) M_plus / (PetscReal) M) * 100.;
  M_minus = M - M_plus;
  per_minus = 100 - per_plus;

  if (svm_mod == 2) N -= 1;

  TRY( PetscViewerASCIIPushTab(v) );
  TRY( PetscViewerASCIIPrintf(v,"samples\t%5D\n",M) );
  TRY( PetscViewerASCIIPrintf(v,"samples+\t%5D (%.2f%%)\n",M_plus,per_plus) );
  TRY( PetscViewerASCIIPrintf(v,"samples-\t%5D (%.2f%%)\n",M_minus,per_minus) );
  TRY( PetscViewerASCIIPrintf(v,"features\t%5D\n",N) );
  TRY( PetscViewerASCIIPopTab(v) );

  TRY( VecDestroy(&y_max) );
  TRY( ISDestroy(&is_plus) );
  PetscFunctionReturn(0);
}
