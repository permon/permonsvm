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

template<typename T=int>
bool ParseSourceFileLine(std::string &line, std::vector<T> &cols, T &yi) {
    using namespace std;
    using namespace boost;

    regex r("\\s+|:");

    vector<string> parsed_values;
    int col;

    algorithm::split_regex(parsed_values, line, r);

    yi = boost::lexical_cast<T>(parsed_values[0]);

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
bool GetMatrixStructure(const std::string &filename, std::vector<T> &nnz_per_row, T &num_cols) {
    using namespace std;

    ifstream fl_hldr;
    string tmp_line; vector<T> tmp_vec;

    fl_hldr.open(filename.c_str());

    num_cols = 0;
    if (fl_hldr.good()) {
        while (getline(fl_hldr, tmp_line)) {
            ParseSourceFileLine(tmp_line, tmp_vec);
            if (tmp_vec.back() > num_cols) {
                num_cols = tmp_vec.back();
            }
            nnz_per_row.push_back(tmp_vec.size());
        }
    } else {
        return (false);
    }

    return (true);
}

#undef __FUNCT__
#define __FUNCT__ "PermonExcapeLoadData"
static PetscErrorCode PermonExcapeLoadData(MPI_Comm comm,const char *data_file_name,QP qp)
{
  using namespace std;
  vector<int> y;
  vector<int> nnz_per_row;
  int num_cols;

  PetscFunctionBeginI;
  TRY( !GetMatrixStructure(string(data_file_name), nnz_per_row, num_cols) );

  cout << endl << "nnz_per_row:" << endl;
  for (auto i = nnz_per_row.begin(); i != nnz_per_row.end(); ++i)
    std::cout << *i << ' ';
  cout << endl << "num_cols: " << num_cols << endl;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "testQPS_files"
PetscErrorCode testQPS_files()
{
  QP             qp  = NULL;
  QPS            qps = NULL;
  const QPSType  qpstype;
  
  PetscFunctionBeginI;
  /* ------------------------------------------------------------------------ */
  TRY( PetscLogStageRegister("load data", &loadStage) );
  TRY( PetscLogStagePush(loadStage) );
  {
    TRY( QPCreate(comm,&qp) );
    TRY( PermonExcapeLoadData(comm,"gene_GSK3B_level6_1000_1_1_1_train.txt",qp) );
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
  }
  TRY( PetscLogStagePop() );
  
  /* ------------------------------------------------------------------------ */  
  TRY( PetscLogStageRegister("save data", &saveStage) );
  TRY( PetscLogStagePush(saveStage) );
  //
  TRY( PetscLogStagePop() );  
  
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

  TRY( FllopSetTrace(PETSC_TRUE) );
  TRY( FllopSetDebug(PETSC_TRUE) );
  TRY( FllopSetObjectInfo(PETSC_TRUE) );

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
