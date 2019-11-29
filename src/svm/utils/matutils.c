
#include <petsc/private/matimpl.h>
#include <permonsvm.h>

typedef struct {
    Mat       inner;
    PetscReal bias;
} MatCtx;

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Biased"
PetscErrorCode MatDestroy_Biased(Mat A)
{
  void   *ptr = NULL;
  MatCtx *ctx = NULL;

  PetscFunctionBegin;
  TRY( MatShellGetContext(A,&ptr) );
  ctx = (MatCtx *) ptr;

  TRY( MatDestroy(&ctx->inner) );
  TRY( PetscFree(ctx) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Biased"
PetscErrorCode MatMult_Biased(Mat A,Vec x,Vec y)
{
  MPI_Comm          comm;
  PetscMPIInt       comm_size,comm_rank;
  MPI_Request       req;

  PetscInt          n;
  PetscInt          low,hi;
  PetscScalar       a,b;

  IS                is_col;
  Vec               x_sub;
  const PetscScalar *x_arr;

  void              *ptr = NULL;
  MatCtx            *ctx = NULL;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject) A,&comm) );
  TRY( MPI_Comm_size(comm,&comm_size) );
  TRY( MPI_Comm_rank(comm,&comm_rank) );

  TRY( MatShellGetContext(A,&ptr) );
  ctx = (MatCtx *) ptr;

  if (comm_rank == comm_size - 1) {
    TRY( VecGetLocalSize(x,&n) );
    TRY( VecGetArrayRead(x,&x_arr) );
    a = x_arr[n - 1];
    TRY( VecRestoreArrayRead(y,&x_arr) );
  }
  TRY( MPI_Ibcast(&a,1,MPIU_SCALAR,comm_size - 1,comm,&req) );
  TRY( MPI_Wait(&req,MPI_STATUS_IGNORE) );

  TRY( VecGetOwnershipRange(x,&low,&hi) );
  if (comm_rank == comm_size - 1) hi -= 1;
  TRY( ISCreateStride(comm,hi - low,low,1,&is_col) );

  TRY( VecGetSubVector(x,is_col,&x_sub) );
  TRY( MatMult(ctx->inner,x_sub,y) );
  TRY( VecRestoreSubVector(x,is_col,&x_sub) );

  b = ctx->bias * a;
  TRY( VecShift(y,b) );

  /* Free memory */
  TRY( ISDestroy(&is_col) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Biased"
PetscErrorCode MatMultTranspose_Biased(Mat A,Vec x,Vec y)
{
  MPI_Comm    comm;
  PetscMPIInt comm_size,comm_rank;

  PetscInt    n;
  PetscInt    low,hi;
  PetscScalar a;

  IS          is_col;
  Vec         y_sub;

  void        *ptr = NULL;
  MatCtx      *ctx = NULL;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject) A,&comm) );
  TRY( MPI_Comm_size(comm,&comm_size) );
  TRY( MPI_Comm_rank(comm,&comm_rank) );

  TRY( MatShellGetContext(A,&ptr) );
  ctx = (MatCtx *) ptr;

  TRY( VecGetOwnershipRange(y,&low,&hi) );
  if (comm_rank == comm_size - 1) hi -= 1;
  TRY( ISCreateStride(comm,hi - low,low,1,&is_col) );

  TRY( VecGetSubVector(y,is_col,&y_sub) );
  TRY( MatMultTranspose(ctx->inner,x,y_sub) );
  TRY( VecRestoreSubVector(y,is_col,&y_sub) );

  TRY( VecSum(x,&a) );
  if (comm_rank == comm_size - 1) {
    TRY( VecGetSize(y,&n) );
    a *= ctx->bias;
    TRY( VecSetValue(y,n - 1,a,INSERT_VALUES) );
  }
  TRY( VecAssemblyBegin(y) );
  TRY( VecAssemblyEnd(y) );

  TRY( ISDestroy(&is_col) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetOwnershipIS_Biased"
PetscErrorCode MatGetOwnershipIS_Biased(Mat mat,IS *rows,IS *cols)
{
  void   *ptr;
  MatCtx *ctx;

  PetscFunctionBegin;
  TRY( MatShellGetContext(mat,&ptr) );
  ctx = (MatCtx *) ptr;
  TRY( MatGetOwnershipIS(ctx->inner,rows,cols) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateSubMatrix_Biased"
PetscErrorCode MatCreateSubMatrix_Biased(Mat A,IS isrow,IS iscol,MatReuse cll,Mat *out)
{
  MPI_Comm    comm;
  PetscMPIInt comm_size,comm_rank;

  Mat         A_sub;
  PetscInt    m,n,M,N;
  Mat         mat_inner;

  void        *ptr     = NULL;
  MatCtx      *ctx     = NULL;
  MatCtx      *new_ctx = NULL;

  PetscFunctionBegin;
  TRY( MatShellGetContext(A,&ptr) );
  ctx = (MatCtx *) ptr;

  TRY( PetscObjectGetComm((PetscObject) ctx->inner,&comm) );
  TRY( MPI_Comm_size(comm,&comm_size) );
  TRY( MPI_Comm_rank(comm,&comm_rank) );

  TRY( MatCreateSubMatrix(ctx->inner,isrow,iscol,cll,&A_sub) );

  TRY( PetscNew(&new_ctx) );
  new_ctx->inner = A_sub;
  new_ctx->bias  = ctx->bias;

  TRY( MatGetLocalSize(A_sub,&m,&n) );
  TRY( MatGetSize(A_sub,&M,&N) );

  if (comm_rank == comm_size - 1) {
      n += 1;
  }
  N += 1;

  TRY( MatCreateShell(comm,m,n,M,N,new_ctx,&mat_inner) );
  /* Set shell matrix functions */
  TRY( MatShellSetOperation(mat_inner,MATOP_DESTROY         ,(void(*)(void))MatDestroy_Biased) );
  TRY( MatShellSetOperation(mat_inner,MATOP_MULT            ,(void(*)(void))MatMult_Biased) );
  TRY( MatShellSetOperation(mat_inner,MATOP_MULT_TRANSPOSE  ,(void(*)(void))MatMultTranspose_Biased) );
  TRY( MatShellSetOperation(mat_inner,MATOP_CREATE_SUBMATRIX,(void(*)(void))MatCreateSubMatrix_Biased) );
  TRY( PetscObjectComposeFunction((PetscObject) mat_inner,"MatGetOwnershipIS_C",MatGetOwnershipIS_Biased) );

  *out = mat_inner;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatBiasedGetInnerMat"
/*
  MatBiasedGetInnerMat - Get inner (original) matrix

  Not Collective

  Input Parameter:
. A - biased matrix context

  Output Parameter:
. inner - original (inner) matrix context

  Notes:
  This routine does not return a new copy of an inner (original) matrix. It actually returns a pointer to the same matrix that is passed in creating routine namely MatBiasedCreate().

  Do not call MatDestroy() after using of this object.

  Level: advanced

.seealso MatBiasedCreate()
*/
PetscErrorCode MatBiasedGetInnerMat(Mat A,Mat *inner)
{
  void   *ptr;
  MatCtx *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(inner,2);

  TRY( MatShellGetContext(A,&ptr) );
  ctx = (MatCtx *) ptr;

  *inner = ctx->inner;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatBiasedGetBias"
/*@
  MatBiasedGetBias - Gets a real value of bias.

  Not Collective

  Input Parameter:
. A - mat context

  Output Parameter:
. bias - value of bias

  Level: beginner

.seealso MatBiasedCreate()
@*/
PetscErrorCode MatBiasedGetBias(Mat A,PetscReal *bias)
{
  void   *ptr;
  MatCtx *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(bias,2);

  TRY( MatShellGetContext(A,&ptr) );
  ctx = (MatCtx *) ptr;

  *bias = ctx->bias;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatBiasedCreate"
/*@
  MatBiasedCreate - Creates biased matrix, x_i <- [x_i, bias].

  Collective on Mat

  Input Parameters:
+ A - mat to be biased
- bias - value of bias

  Output Parameter:
. A_biased - mat biased context

  Notes:
  Mat type of created mat is shell. Currently, we do not plan to create its own matrix type.

  Level: advanced

.seealso MatBiasedGetBias(), MatBiasedGetInnerMat()
@*/
PetscErrorCode MatBiasedCreate(Mat A,PetscReal bias,Mat *A_biased)
{
  MPI_Comm    comm;
  PetscMPIInt comm_size,comm_rank;

  Mat         A_biased_inner;
  PetscInt    m,n,M,N;
  MatCtx      *ctx = NULL;

  const char  *A_name,*A_prefix;
  char        A_name_inner[50];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(A,bias,2);
  PetscValidPointer(A_biased,3);

  PetscObjectGetComm((PetscObject) A,&comm);
  TRY( MPI_Comm_size(comm,&comm_size) );
  TRY( MPI_Comm_rank(comm,&comm_rank) );

  /* Create matrix context */
  TRY( PetscNew(&ctx) );
  ctx->inner    = A;
  ctx->bias = bias;

  TRY( MatGetLocalSize(A,&m,&n) );
  TRY( MatGetSize(A,&M,&N) );

  if (comm_rank == comm_size - 1) n += 1;

  TRY( MatCreateShell(comm,m,n,M,N+1,ctx,&A_biased_inner) );
  /* Set shell matrix functions */
  TRY( MatShellSetOperation(A_biased_inner,MATOP_DESTROY         ,(void(*)(void))MatDestroy_Biased) );
  TRY( MatShellSetOperation(A_biased_inner,MATOP_MULT            ,(void(*)(void))MatMult_Biased) );
  TRY( MatShellSetOperation(A_biased_inner,MATOP_MULT_TRANSPOSE  ,(void(*)(void))MatMultTranspose_Biased) );
  TRY( MatShellSetOperation(A_biased_inner,MATOP_CREATE_SUBMATRIX,(void(*)(void))MatCreateSubMatrix_Biased) );
  TRY( PetscObjectComposeFunction((PetscObject) A_biased_inner,"MatGetOwnershipIS_C",MatGetOwnershipIS_Biased) );

  /* Set name of biased mat */
  TRY( PetscObjectGetName((PetscObject) A,&A_name) );
  TRY( PetscStrcpy(A_name_inner,A_name) );
  TRY( PetscStrcat(A_name_inner,"_biased") );
  TRY( PetscObjectSetName((PetscObject) A_biased_inner,A_name_inner) );
  /* Set prefix of biased mat */
  TRY( PetscObjectGetOptionsPrefix((PetscObject) A,&A_prefix) );
  TRY( PetscObjectSetOptionsPrefix((PetscObject) A_biased_inner,A_prefix) );
  TRY( PetscObjectAppendOptionsPrefix((PetscObject) A_biased_inner,"biased_") );

  *A_biased = A_biased_inner;
  PetscFunctionReturn(0);
}
