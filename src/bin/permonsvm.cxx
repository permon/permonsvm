#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>

#include <mpi.h>
#include <petscksp.h>
#include <fllopqps.h>

extern char Fllop_input_dir[FLLOP_MAX_PATH_LEN], Fllop_output_dir[FLLOP_MAX_PATH_LEN];

static MPI_Comm     comm;
static PetscMPIInt  rank, commsize;
static PetscLogStage loadStage, prepStage, solvStage, postStage, saveStage;
static PetscReal C = 1.0;

template<typename T=int>
bool ParseSourceFileLine(std::string &line, std::vector<T> &cols, T &yi) {
    using namespace std;
    using namespace boost;

    regex r("\\s+|:");

    vector<string> parsed_values;
    int col;

    algorithm::split_regex(parsed_values, line, r);

    yi = boost::lexical_cast<T>(parsed_values[0]) ? 1 : -1;

    cols.clear();
    for (int i = 1; i < parsed_values.size(); i += 2) {
        try {
            col = boost::lexical_cast<T>(parsed_values[i]);
        } catch (bad_lexical_cast &e) {
            cout << "Caught bad lexical cast with error " << e.what() << endl;
            return (false);
        } catch (...) {
            cout << "Unknown exception caught!" << endl;
            return (false);
        }
        cols.push_back(col);
    }

    return (true);
}

template<typename T=int>
bool ParseSourceFileLine(std::string &line, std::vector<T> &cols) {
    using namespace std;
    using namespace boost;

    regex r("\\s+|:");

    vector<string> parsed_values;
    int col;

    algorithm::split_regex(parsed_values, line, r);

    cols.clear();
    for (int i = 1; i < parsed_values.size(); i += 2) {
        try {
            col = boost::lexical_cast<T>(parsed_values[i]);
        } catch (bad_lexical_cast &e) {
            cout << "Caught bad lexical cast with error " << e.what() << endl;
            return (false);
        } catch (...) {
            cout << "Unknown exception caught!" << endl;
            return (false);
        }
        cols.push_back(col);
    }

    return (true);
}

template<typename T=int>
bool GetMatrixStructure(const std::string &filename, std::vector<T> &nnz_per_row, T &nnz_max, T &num_cols, int n_examples=-1) {
    using namespace std;

    ifstream fl_hldr;
    string tmp_line; vector<T> tmp_vec;
    T tmp_cols, nnz;
    int i;

    fl_hldr.open(filename.c_str());

    num_cols = 0;
    nnz_max = 0;
    i = 0;
    if (fl_hldr.good()) {
        while (getline(fl_hldr, tmp_line)) {
            ParseSourceFileLine(tmp_line, tmp_vec);
            tmp_cols = tmp_vec.back();
            if (tmp_cols > num_cols) {
                num_cols = tmp_cols;
            }
            nnz = tmp_vec.size();
            if (nnz > nnz_max) {
              nnz_max = nnz;
            }
            nnz_per_row.push_back(nnz);
            i++;
            if (n_examples != -1 && i == n_examples) break;
        }
    } else {
        return (false);
    }

    return (true);
}

#undef __FUNCT__
#define __FUNCT__ "PermonExcapeSetValues"
PetscErrorCode PermonExcapeSetValues(Mat Xt, Vec y, PetscInt nnz_max, const std::string &filename) {
    using namespace std;

    ifstream fl_hldr;
    string tmp_line; vector<PetscInt> idxn;
    PetscInt yi, m;
    PetscScalar ones[nnz_max];

    PetscFunctionBeginI;
    TRY( MatGetLocalSize(Xt, &m, NULL) );
    fl_hldr.open(filename.c_str());
    TRY( fl_hldr.fail() );

    std::fill_n(ones, nnz_max, 1.0);

    PetscInt i = 0;
    while (getline(fl_hldr, tmp_line) && i < m) {
      ParseSourceFileLine(tmp_line, idxn, yi);
      std::transform(idxn.begin(), idxn.end(), idxn.begin(), [](PetscInt a) { return a-1; });  // decrement 1-based values of idxn to be 0-based
      TRY( MatSetValues(Xt, 1, &i, idxn.size(), &idxn[0], ones, INSERT_VALUES) );
      TRY( VecSetValue(y,i,yi,INSERT_VALUES) );
      i++;
    }

    TRY( MatAssemblyBegin(Xt, MAT_FINAL_ASSEMBLY) );
    TRY( MatAssemblyEnd(Xt, MAT_FINAL_ASSEMBLY) );
    TRY( VecAssemblyBegin(y) );
    TRY( VecAssemblyEnd(y) );
    PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonExcapeLoadData"
PetscErrorCode PermonExcapeLoadData(MPI_Comm comm, const char *data_file_name, PetscInt n_examples, Mat *Xt_new, Vec *y_new)
{
  using namespace std;
  vector<PetscInt> nnz_per_row;
  PetscInt n_attributes, nnz_max;
  Mat Xt;
  Vec y;

  PetscFunctionBeginI;
  TRY( !GetMatrixStructure(string(data_file_name), nnz_per_row, nnz_max, n_attributes, n_examples) );
  if (n_examples == -1) n_examples = nnz_per_row.size();

  TRY( MatCreate(comm, &Xt) );
  TRY( MatSetSizes(Xt, n_examples, n_attributes, PETSC_DECIDE, PETSC_DECIDE) );
  TRY( MatSetOptionsPrefix(Xt, "Xt_") );
  TRY( MatSetFromOptions(Xt) );
  TRY( MatSeqAIJSetPreallocation(Xt, -1, &nnz_per_row[0]) );
  //TODO MatMPIAIJSetPreallocation

  TRY( VecCreate(comm,&y) );
  TRY( VecSetSizes(y, n_examples, PETSC_DECIDE) );
  TRY( VecSetOptionsPrefix(y, "y_") );
  TRY( VecSetFromOptions(y) );

  TRY( PermonExcapeSetValues(Xt, y, nnz_max, data_file_name) );

  *Xt_new = Xt;
  *y_new = y;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonExcapeDataToQP"
PetscErrorCode PermonExcapeDataToQP(Mat Xt, Vec y, QP *qp_new)
{
  MPI_Comm comm;
  QP qp;
  Mat X,H;
  Vec e,lb,ub;
  Mat BE;
  PetscReal norm;

  PetscFunctionBeginI;
  TRY( PetscObjectGetComm((PetscObject)Xt, &comm) );
  TRY( QPCreate(comm,&qp) );

  TRY( FllopMatTranspose(Xt,MAT_TRANSPOSE_CHEAPEST,&X) );
  TRY( MatCreateNormal(X,&H) );
  TRY( MatDiagonalScale(H,y,y) );
  TRY( QPSetOperator(qp,H) );

  TRY( VecDuplicate(y,&e) );
  TRY( VecSet(e,1.0) );
  TRY( QPSetRhs(qp, e) );

  TRY( MatCreateOneRow(y,&BE) );
  TRY( VecNorm(y, NORM_2, &norm) );
  TRY( MatScale(BE,1.0/norm) );
  TRY( QPSetEq(qp, BE, NULL));

  TRY( VecDuplicate(y,&lb) );
  TRY( VecDuplicate(y,&ub) );
  TRY( VecSet(lb, 0.0) );
  TRY( VecSet(ub, C) );
  TRY( QPSetBox(qp, lb, ub) );

  TRY( MatDestroy(&X) );
  TRY( MatDestroy(&H) );
  TRY( VecDestroy(&e) );
  TRY( MatDestroy(&BE) );
  TRY( VecDestroy(&lb) );
  TRY( VecDestroy(&ub) );
  *qp_new = qp;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSVMSetC"
PetscErrorCode PermonSVMSetC(PetscReal C_)
{
  PetscFunctionBegin;
  C = C_;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "testQPS_files"
PetscErrorCode testQPS_files()
{
  QP             qp  = NULL;
  QPS            qps = NULL;
  const QPSType  qpstype;
  Mat            Xt;
  Vec            y, z, w, Xtw;
  PetscScalar    b;
  char           filename[PETSC_MAX_PATH_LEN] = "dummy.txt";
  PetscInt       n_examples = -1;  /* -1 means all */

  PetscFunctionBeginI;
  TRY( PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),NULL) );
  TRY( PetscOptionsGetInt(NULL,NULL,"-n_examples",&n_examples,NULL));

  /* ------------------------------------------------------------------------ */
  TRY( PetscLogStageRegister("load data", &loadStage) );
  TRY( PetscLogStagePush(loadStage) );
  {
    TRY( PermonExcapeLoadData(comm, filename, n_examples, &Xt, &y) );
    TRY( PermonExcapeDataToQP(Xt, y, &qp) );
  }
  TRY( PetscLogStagePop() );

  /* ------------------------------------------------------------------------ */  
  TRY( PetscLogStageRegister("preprocessing", &prepStage) );
  TRY( PetscLogStagePush(prepStage) );
  {
    TRY( QPTFromOptions(qp) );

    TRY( QPSCreate(comm, &qps) );
    TRY( QPSSetQP(qps, qp) );
    TRY( QPSSetAutoPostSolve(qps, PETSC_FALSE) );
    TRY( QPSSetFromOptions(qps) );

    TRY( QPSSetUp(qps) );
    TRY( QPSGetType(qps, &qpstype) );
    TRY( PetscInfo1(fllop, "Using automatically chosen QPS type %s\n", qpstype) );
  }
  TRY( PetscLogStagePop() );

  /* ------------------------------------------------------------------------ */  
  TRY( PetscLogStageRegister("solve", &solvStage) );
  TRY( PetscLogStagePush(solvStage) );
  {
    TRY( QPSSolve(qps) );
  }
  TRY( PetscLogStagePop() );

  /* ------------------------------------------------------------------------ */
  TRY( PetscLogStageRegister("postsolve", &postStage) );
  TRY( PetscLogStagePush(postStage) );
  {
    TRY( QPSPostSolve(qps) );

    /* reconstruct w from dual solution z */
    {
      Vec Yz;

      TRY( QPGetSolutionVector(qp, &z) );
      TRY( VecDuplicate(z, &Yz));

      TRY( VecPointwiseMult(Yz, y, z) );
      TRY( MatCreateVecs(Xt, &w, NULL) );
      TRY( MatMultTranspose(Xt, Yz, w) );

      TRY( VecDestroy(&Yz) );
    }

    /* reconstruct b from dual solution z */
    {
      IS is_sv;
      Vec o, y_sv, Xtw_sv, t;
      PetscInt len_sv;

      TRY( VecDuplicate(z, &o) );
      TRY( VecZeroEntries(o) );
      TRY( MatCreateVecs(Xt, NULL, &Xtw));

      TRY( VecWhichGreaterThan(z, o, &is_sv) );
      TRY( ISGetSize(is_sv, &len_sv) );
      TRY( MatMult(Xt, w, Xtw) );
      TRY( VecGetSubVector(y, is_sv, &y_sv) );
      TRY( VecGetSubVector(Xtw, is_sv, &Xtw_sv) );
      TRY( VecDuplicate(y_sv, &t) );
      TRY( VecWAXPY(t, -1.0, Xtw_sv, y_sv) );
      TRY( VecRestoreSubVector(y, is_sv, &y_sv) );
      TRY( VecRestoreSubVector(Xtw, is_sv, &Xtw_sv) );
      TRY( VecSum(t, &b) );
      b /= len_sv;

      TRY( ISDestroy(&is_sv) );
      TRY( VecDestroy(&o) );
      TRY( VecDestroy(&t) );
    }

    /* test */
    {
      Vec my_y;
      PetscInt i,m;
      const PetscScalar *Xtw_arr;
      PetscScalar *my_y_arr;
      PetscReal norm;

      TRY( VecDuplicate(Xtw, &my_y) );
      TRY( VecGetLocalSize(Xtw, &m) );

      TRY( VecGetArrayRead(Xtw, &Xtw_arr) );
      TRY( VecGetArray(my_y, &my_y_arr) );
      for (i=0; i<m; i++) {
        if (Xtw_arr[i] + b > 0) {
          my_y_arr[i] = 1.0;
        } else {
          my_y_arr[i] = -1.0;
        }
      }
      TRY( VecRestoreArrayRead(Xtw, &Xtw_arr) );
      TRY( VecRestoreArray(my_y, &my_y_arr) );

      //TRY( VecView(my_y,PETSC_VIEWER_STDOUT_WORLD) );
      //TRY( VecView(y,PETSC_VIEWER_STDOUT_WORLD) );

      TRY( VecAXPY(my_y,-1.0,y) );
      TRY( VecNorm(my_y,NORM_2,&norm) );
      TRY( PetscPrintf(comm, "||y - my_y|| = %f\n",norm) );

      TRY( VecDestroy(&my_y) );
    }
  }
  TRY( PetscLogStagePop() );

  /* ------------------------------------------------------------------------ */  
  TRY( PetscLogStageRegister("save data", &saveStage) );
  TRY( PetscLogStagePush(saveStage) );
  //
  TRY( PetscLogStagePop() );

  TRY( MatDestroy(&Xt) );
  TRY( VecDestroy(&y) );
  TRY( VecDestroy(&w) );
  TRY( VecDestroy(&Xtw) );
  TRY( QPSDestroy(&qps) );
  TRY( QPDestroy(&qp) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char** argv)
{
  FllopInitialize(&argc, &argv, (char*) 0);

  PetscFunctionBegin;
  comm = PETSC_COMM_WORLD;
  TRY( MPI_Comm_rank(comm, &rank) );
  TRY( MPI_Comm_size(comm, &commsize) );

  FLLOP_ASSERT(commsize==1,"currently only sequential");

  TRY( PetscPrintf(comm, "PETSC_DIR:\t" PETSC_DIR "\n") );
  TRY( PetscPrintf(comm, "PETSC_ARCH:\t" PETSC_ARCH "\n") );
#ifdef PETSC_RELEASE_DATE
#define DATE PETSC_RELEASE_DATE
#else
#define DATE PETSC_VERSION_DATE
#endif
  TRY( PetscPrintf(comm, "PETSc version:\t%d.%d.%d patch %d (%s)\n", PETSC_VERSION_MAJOR, PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR, PETSC_VERSION_PATCH, DATE) );
#undef DATE
  TRY( PetscPrintf(comm,"Input dir:\t%s\n", Fllop_input_dir) );
  TRY( PetscPrintf(comm,"Output dir:\t%s\n", Fllop_output_dir) );

  TRY( testQPS_files() );
  TRY( FllopFinalize() );
  PetscFunctionReturn(0);
}
