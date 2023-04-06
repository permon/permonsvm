
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
  PetscCall(MatShellGetContext(A,&ptr));
  ctx = (MatCtx *) ptr;

  PetscCall(MatDestroy(&ctx->inner));
  PetscCall(PetscFree(ctx));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatGetOwnershipIS_C",NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(PetscObjectGetComm((PetscObject) A,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&comm_size));
  PetscCallMPI(MPI_Comm_rank(comm,&comm_rank));

  PetscCall(MatShellGetContext(A,&ptr));
  ctx = (MatCtx *) ptr;

  if (comm_rank == comm_size - 1) {
    PetscCall(VecGetLocalSize(x,&n));
    PetscCall(VecGetArrayRead(x,&x_arr));
    a = x_arr[n - 1];
    PetscCall(VecRestoreArrayRead(y,&x_arr));
  }
  PetscCallMPI(MPI_Ibcast(&a,1,MPIU_SCALAR,comm_size - 1,comm,&req));
  PetscCallMPI(MPI_Wait(&req,MPI_STATUS_IGNORE));

  PetscCall(VecGetOwnershipRange(x,&low,&hi));
  if (comm_rank == comm_size - 1) hi -= 1;
  PetscCall(ISCreateStride(comm,hi - low,low,1,&is_col));

  PetscCall(VecGetSubVector(x,is_col,&x_sub));
  PetscCall(MatMult(ctx->inner,x_sub,y));
  PetscCall(VecRestoreSubVector(x,is_col,&x_sub));

  b = ctx->bias * a;
  PetscCall(VecShift(y,b));

  /* Free memory */
  PetscCall(ISDestroy(&is_col));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(PetscObjectGetComm((PetscObject) A,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&comm_size));
  PetscCallMPI(MPI_Comm_rank(comm,&comm_rank));

  PetscCall(MatShellGetContext(A,&ptr));
  ctx = (MatCtx *) ptr;

  PetscCall(VecGetOwnershipRange(y,&low,&hi));
  if (comm_rank == comm_size - 1) hi -= 1;
  PetscCall(ISCreateStride(comm,hi - low,low,1,&is_col));

  PetscCall(VecGetSubVector(y,is_col,&y_sub));
  PetscCall(MatMultTranspose(ctx->inner,x,y_sub));
  PetscCall(VecRestoreSubVector(y,is_col,&y_sub));

  PetscCall(VecSum(x,&a));
  if (comm_rank == comm_size - 1) {
    PetscCall(VecGetSize(y,&n));
    a *= ctx->bias;
    PetscCall(VecSetValue(y,n - 1,a,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));

  PetscCall(ISDestroy(&is_col));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetOwnershipIS_Biased"
PetscErrorCode MatGetOwnershipIS_Biased(Mat mat,IS *rows,IS *cols)
{
  void   *ptr;
  MatCtx *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ptr));
  ctx = (MatCtx *) ptr;
  PetscCall(MatGetOwnershipIS(ctx->inner,rows,cols));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  VecType     vtype;

  void        *ptr     = NULL;
  MatCtx      *ctx     = NULL;
  MatCtx      *new_ctx = NULL;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&ptr));
  ctx = (MatCtx *) ptr;

  PetscCall(PetscObjectGetComm((PetscObject) ctx->inner,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&comm_size));
  PetscCallMPI(MPI_Comm_rank(comm,&comm_rank));

  PetscCall(MatCreateSubMatrix(ctx->inner,isrow,iscol,cll,&A_sub));

  PetscCall(PetscNew(&new_ctx));
  new_ctx->inner = A_sub;
  new_ctx->bias  = ctx->bias;

  PetscCall(MatGetLocalSize(A_sub,&m,&n));
  PetscCall(MatGetSize(A_sub,&M,&N));

  if (comm_rank == comm_size - 1) {
      n += 1;
  }
  N += 1;

  PetscCall(MatCreateShell(comm,m,n,M,N,new_ctx,&mat_inner));
  /* Set shell matrix functions */
  PetscCall(MatShellSetOperation(mat_inner,MATOP_DESTROY         ,(void(*)(void))MatDestroy_Biased));
  PetscCall(MatShellSetOperation(mat_inner,MATOP_MULT            ,(void(*)(void))MatMult_Biased));
  PetscCall(MatShellSetOperation(mat_inner,MATOP_MULT_TRANSPOSE  ,(void(*)(void))MatMultTranspose_Biased));
  PetscCall(MatShellSetOperation(mat_inner,MATOP_CREATE_SUBMATRIX,(void(*)(void))MatCreateSubMatrix_Biased));
  PetscCall(PetscObjectComposeFunction((PetscObject) mat_inner,"MatGetOwnershipIS_C",MatGetOwnershipIS_Biased));

  /* Set the default vector type for the shell to be the same as for the matrix A */
  PetscCall(MatGetVecType(A,&vtype));
  PetscCall(MatShellSetVecType(mat_inner,vtype));

  *out = mat_inner;
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscCall(MatShellGetContext(A,&ptr));
  ctx = (MatCtx *) ptr;

  *inner = ctx->inner;
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscCall(MatShellGetContext(A,&ptr));
  ctx = (MatCtx *) ptr;

  *bias = ctx->bias;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  VecType     vtype;

  const char  *A_name,*A_prefix;
  char        A_name_inner[50];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(A,bias,2);
  PetscValidPointer(A_biased,3);

  PetscCall(PetscObjectGetComm((PetscObject) A,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&comm_size));
  PetscCallMPI(MPI_Comm_rank(comm,&comm_rank));

  /* Create matrix context */
  PetscCall(PetscNew(&ctx));
  ctx->inner    = A;
  ctx->bias = bias;

  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCall(MatGetSize(A,&M,&N));

  if (comm_rank == comm_size - 1) n += 1;

  PetscCall(MatCreateShell(comm,m,n,M,N+1,ctx,&A_biased_inner));
  /* Set shell matrix functions */
  PetscCall(MatShellSetOperation(A_biased_inner,MATOP_DESTROY         ,(void(*)(void))MatDestroy_Biased));
  PetscCall(MatShellSetOperation(A_biased_inner,MATOP_MULT            ,(void(*)(void))MatMult_Biased));
  PetscCall(MatShellSetOperation(A_biased_inner,MATOP_MULT_TRANSPOSE  ,(void(*)(void))MatMultTranspose_Biased));
  PetscCall(MatShellSetOperation(A_biased_inner,MATOP_CREATE_SUBMATRIX,(void(*)(void))MatCreateSubMatrix_Biased));
  PetscCall(PetscObjectComposeFunction((PetscObject) A_biased_inner,"MatGetOwnershipIS_C",MatGetOwnershipIS_Biased));

  /* Set the default vector type for the shell to be the same as for the matrix A */
  PetscCall(MatGetVecType(A,&vtype));
  PetscCall(MatShellSetVecType(A_biased_inner,vtype));

  /* Set name of biased mat */
  PetscCall(PetscObjectGetName((PetscObject) A,&A_name));
  PetscCall(PetscStrncpy(A_name_inner,A_name,sizeof(A_name_inner)));
  PetscCall(PetscStrlcat(A_name_inner,"_biased",sizeof(A_name_inner)));
  PetscCall(PetscObjectSetName((PetscObject) A_biased_inner,A_name_inner));
  /* Set prefix of biased mat */
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) A,&A_prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) A_biased_inner,A_prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject) A_biased_inner,"biased_"));

  *A_biased = A_biased_inner;
  PetscFunctionReturn(PETSC_SUCCESS);
}
