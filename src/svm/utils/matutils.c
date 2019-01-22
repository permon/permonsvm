
#include <petsc/private/matimpl.h>
#include <permonsvm.h>

typedef struct {
    Mat         Xt;
    PetscReal bias;
} MatBiasedCtx;

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Biased"
PetscErrorCode MatDestroy_Biased(Mat A)
{
  void         *ptr = NULL;
  MatBiasedCtx *ctx = NULL;

  PetscFunctionBegin;
  TRY( MatShellGetContext(A,&ptr) );
  ctx = (MatBiasedCtx *) ptr;

  TRY( MatDestroy(&ctx->Xt) );
  TRY( PetscFree(ctx) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Biased"
PetscErrorCode MatMult_Biased(Mat A,Vec x,Vec y)
{
  MPI_Comm    comm;
  PetscMPIInt comm_size,comm_rank;
  MPI_Request req;

  PetscInt    n;
  PetscInt    ilow,ihi;
  PetscScalar a,b;
  const PetscScalar *x_arr;

  void         *ptr = NULL;
  MatBiasedCtx *ctx = NULL;

  IS           is_col;
  Vec          x_sub;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject) A,&comm) );
  TRY( MPI_Comm_size(comm,&comm_size) );
  TRY( MPI_Comm_rank(comm,&comm_rank) );

  TRY( MatShellGetContext(A,&ptr) );
  ctx = (MatBiasedCtx *) ptr;

  if (comm_rank == comm_size - 1) {
    TRY( VecGetLocalSize(x,&n) );
    TRY( VecGetArrayRead(x,&x_arr) );
    a = x_arr[n-1];
    TRY( VecRestoreArrayRead(y,&x_arr) );
  }

  TRY( MPI_Ibcast(&a,1,MPIU_SCALAR,comm_size-1,comm,&req) );
  TRY( MPI_Wait(&req,MPI_STATUS_IGNORE) );

  TRY( VecGetOwnershipRange(x,&ilow,&ihi) );
  if (comm_rank == comm_size - 1) ihi -= 1;
  TRY( ISCreateStride(comm,ihi-ilow,ilow,1,&is_col) );

  TRY( VecGetSubVector(x,is_col,&x_sub) );
  TRY( MatMult(ctx->Xt,x_sub,y) );
  TRY( VecRestoreSubVector(x,is_col,&x_sub) );

  TRY( ISDestroy(&is_col) );

  b = ctx->bias * a;
  TRY( VecShift(y,b) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Biased"
PetscErrorCode MatMultTranspose_Biased(Mat A,Vec x,Vec y)
{
  MPI_Comm    comm;
  PetscMPIInt comm_size,comm_rank;

  PetscInt    n;
  PetscInt    ilow,ihi;
  PetscScalar a;

  void         *ptr = NULL;
  MatBiasedCtx *ctx = NULL;

  IS           is_col;
  Vec          y_sub;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject) A,&comm) );
  TRY( MPI_Comm_size(comm,&comm_size) );
  TRY( MPI_Comm_rank(comm,&comm_rank) );

  TRY( MatShellGetContext(A,&ptr) );
  ctx = (MatBiasedCtx *) ptr;

  TRY( VecGetOwnershipRange(y,&ilow,&ihi) );
  if (comm_rank == comm_size - 1) ihi -= 1;
  TRY( ISCreateStride(comm,ihi-ilow,ilow,1,&is_col) );

  TRY( VecGetSubVector(y,is_col,&y_sub) );
  TRY( MatMultTranspose(ctx->Xt,x,y_sub) );
  TRY( VecRestoreSubVector(y,is_col,&y_sub) );

  TRY( ISDestroy(&is_col) );

  /* Change to MPI_Reduce */
  TRY( VecSum(x,&a) );
  if (comm_rank == comm_size - 1) {
    TRY( VecGetSize(y,&n) );
    a *= ctx->bias;
    TRY( VecSetValue(y,n-1,a,INSERT_VALUES) );
  }
  TRY( VecAssemblyBegin(y) );
  TRY( VecAssemblyEnd(y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateSubMatrix_Biased"
PetscErrorCode MatCreateSubMatrix_Biased(Mat mat,IS isrow,IS iscol,MatReuse cll,Mat *newmat)
{
  MPI_Comm     comm;
  PetscMPIInt  comm_size,comm_rank;

  Mat          Xt_sub;
  PetscInt     m,n,M,N;

  Mat          newmat_inner;

  void         *ptr = NULL;
  MatBiasedCtx *ctx = NULL;
  MatBiasedCtx *newctx = NULL;

  PetscFunctionBegin;
  TRY( MatShellGetContext(mat,&ptr) );
  ctx = (MatBiasedCtx *) ptr;

  PetscObjectGetComm((PetscObject) ctx->Xt,&comm);
  TRY( MPI_Comm_size(comm,&comm_size) );
  TRY( MPI_Comm_rank(comm,&comm_rank) );

  TRY( MatCreateSubMatrix(ctx->Xt,isrow,NULL,cll,&Xt_sub) );

  TRY( PetscNew(&newctx) );
  newctx->Xt   = Xt_sub;
  newctx->bias = ctx->bias;

  TRY( MatGetLocalSize(Xt_sub,&m,&n) );
  TRY( MatGetSize(Xt_sub,&M,&N) );

  if (comm_rank == comm_size - 1) {
      n += 1;
  }
  N += 1;

  TRY( MatCreateShell(comm,m,n,M,N,newctx,&newmat_inner) );
  TRY( MatShellSetOperation(newmat_inner,MATOP_DESTROY,(void(*)(void))MatDestroy_Biased) );
  TRY( MatShellSetOperation(newmat_inner,MATOP_MULT,(void(*)(void))MatMult_Biased) );
  TRY( MatShellSetOperation(newmat_inner,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Biased) );
  TRY( MatShellSetOperation(newmat_inner,MATOP_CREATE_SUBMATRIX,(void(*)(void))MatCreateSubMatrix_Biased) );

  *newmat = newmat_inner;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_Biased"
PetscErrorCode MatCreate_Biased(Mat Xt,PetscReal bias,Mat *Xt_biased)
{
  MPI_Comm    comm;
  PetscMPIInt comm_size,comm_rank;

  Mat      Xt_biased_inner;
  PetscInt m,n,M,N;

  MatBiasedCtx *ctx = NULL;

  PetscFunctionBegin;
  PetscObjectGetComm((PetscObject) Xt,&comm);
  TRY( MPI_Comm_size(comm,&comm_size) );
  TRY( MPI_Comm_rank(comm,&comm_rank) );

  TRY( PetscNew(&ctx) );
  ctx->Xt   = Xt;
  ctx->bias = bias;

  TRY( MatGetLocalSize(Xt,&m,&n) );
  TRY( MatGetSize(Xt,&M,&N) );

  if (comm_rank == comm_size - 1) n += 1;

  TRY( MatCreateShell(comm,m,n,M,N+1,ctx,&Xt_biased_inner) );
  TRY( MatShellSetOperation(Xt_biased_inner,MATOP_DESTROY,(void(*)(void))MatDestroy_Biased) );
  TRY( MatShellSetOperation(Xt_biased_inner,MATOP_MULT,(void(*)(void))MatMult_Biased) );
  TRY( MatShellSetOperation(Xt_biased_inner,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Biased) );
  TRY( MatShellSetOperation(Xt_biased_inner,MATOP_CREATE_SUBMATRIX,(void(*)(void))MatCreateSubMatrix_Biased) );

  *Xt_biased = Xt_biased_inner;
  PetscFunctionReturn(0);
}