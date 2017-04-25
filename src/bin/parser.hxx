#ifndef PARSER_HXX
#define PARSER_HXX
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace excape {
    template<typename T1=PetscInt, typename T2=PetscScalar>
    class DataParser_ {
        private:
            std::string filename;
            T1 numbering_base;
            
            bool ParseSourceFileLine(std::string &, std::vector<T1> &, std::vector<T2> &, T1 &);
            bool ParseSourceFileLine(std::string &, std::vector<T1> &);
            bool GetMatrixStructure(std::vector<T1> &, T1 &, T1 &, T1 n_examples=-1);
            PetscErrorCode SetValues(Mat Xt, Vec y, PetscInt nnz_max);
        public:
            void SetInputFileName(const std::string &);
            PetscErrorCode GetData(MPI_Comm, PetscInt, Mat *, Vec *);
            void SetNumberingBase(T1 incides_from_zero = 0);
            
            DataParser_(void);
    };
    
    template<typename T1, typename T2>
    DataParser_<T1, T2>::DataParser_(void) {
        SetNumberingBase();
    }
    
    template<typename T1, typename T2>
    void DataParser_<T1, T2>::SetNumberingBase(T1 numbering_base) {
        this->numbering_base = numbering_base;
    }
    
    template<typename T1, typename T2>
    bool DataParser_<T1, T2>::ParseSourceFileLine(std::string &line, std::vector<T1> &cols, std::vector<T2> &vals, T1 &yi) {
        using namespace std;
        string l, pair, k, v;
        string delimiter(":");
        istringstream stream(line.c_str());

        stream >> l;
        yi = std::atoi(l.c_str());

        cols.clear();
        vals.clear();
        while (stream >> pair) {
          size_t pos = pair.find(delimiter);

          k = pair.substr(0, pos);
          v = pair.substr(pos + 1, string::npos);

          cols.push_back((T1) std::atoi(k.c_str()));
          vals.push_back((T2) std::strtod(v.c_str(), NULL));
        }
        
        return (true);
    }
    
    template<typename T1, typename T2>
    bool DataParser_<T1, T2>::ParseSourceFileLine(std::string &line, std::vector<T1> &cols) {
        using namespace std;
        string l, pair, k, v;
        string delimiter(":");
        stringstream stream(line.c_str());

        stream >> l;

        cols.clear();
        while (stream >> pair) {
          size_t pos = pair.find(delimiter);

          k = pair.substr(0, pos);
          cols.push_back((T1) std::atoi(k.c_str()));
        }

        return (true);
    }
    
#undef __FUNCT__
#define __FUNCT__ "GetMatrixStructure"
    template<typename T1, typename T2>
    bool DataParser_<T1, T2>::GetMatrixStructure(std::vector<T1> &nnz_per_row, T1 &nnz_max, T1 &num_cols, T1 n_examples) {
        using namespace std;

        ifstream fl_hldr;
        string tmp_line; vector<T1> tmp_vec;
        T1 tmp_cols, nnz;
        T1 i;

        fl_hldr.open(this->filename.c_str());

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
                if (n_examples > 0 && i == n_examples) break;
            }
            num_cols = num_cols - numbering_base + 1;
            if (n_examples > i) {
              FLLOP_SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"specified n_examples %d is greater than number of rows in data file %d!\n",n_examples,i);
            } else if (n_examples >= 0) {
              FLLOP_ASSERT2(n_examples == i, "n_examples == i (%d != %d)",n_examples,i);
            }
        } else {
            return (false);
        }

        return (true);
    }
    
#undef __FUNCT__
#define __FUNCT__ "SetValues"
    template<typename T1, typename T2>
    PetscErrorCode DataParser_<T1, T2>::SetValues(Mat Xt, Vec y, PetscInt nnz_max) {
        using namespace std;

        ifstream fl_hldr;
        string tmp_line; vector<PetscInt> idxn; vector<PetscScalar> vals;
        PetscInt yi, m;
        PetscInt nb = this->numbering_base;

        PetscFunctionBeginI;
        TRY( MatGetLocalSize(Xt, &m, NULL) );
        fl_hldr.open(filename.c_str());
        TRY( fl_hldr.fail() );

        PetscInt i = 0;
        while (getline(fl_hldr, tmp_line) && i < m) {
          ParseSourceFileLine(tmp_line, idxn, vals, yi);
          std::transform(idxn.begin(), idxn.end(), idxn.begin(), [nb](PetscInt a) { return a-nb; });  // decrement 1-based values of idxn to be 0-based
          TRY( MatSetValues(Xt, 1, &i, idxn.size(), &idxn[0], &vals[0], INSERT_VALUES) );
          TRY( VecSetValue(y,i,yi,INSERT_VALUES) );
          i++;
        }

        TRY( MatAssemblyBegin(Xt, MAT_FINAL_ASSEMBLY) );
        TRY( MatAssemblyEnd(Xt, MAT_FINAL_ASSEMBLY) );
        TRY( VecAssemblyBegin(y) );
        TRY( VecAssemblyEnd(y) );
        PetscFunctionReturnI(0);
    }
    
    template<typename T1, typename T2>
    void DataParser_<T1, T2>::SetInputFileName(const std::string &filename) {
        this->filename = filename;
    }
    
#undef __FUNCT__
#define __FUNCT__ "GetData"
    template<typename T1, typename T2>
    PetscErrorCode DataParser_<T1, T2>::GetData(MPI_Comm comm, PetscInt n_examples, Mat *Xt_new, Vec *y_new)
    {
        using namespace std;
        vector<PetscInt> nnz_per_row;
        PetscInt n_attributes, nnz_max;
        Mat Xt;
        Vec y;

        PetscFunctionBeginI;
        TRY( !this->GetMatrixStructure(nnz_per_row, nnz_max, n_attributes, n_examples) );
        if (n_examples == PETSC_DECIDE || n_examples == PETSC_DEFAULT) {
          n_examples = nnz_per_row.size();
        } else if (n_examples < 0) {
          FLLOP_SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"n_examples must be nonnegative");
        }

        TRY( PetscPrintf(comm,"### PermonSVM: GetData %d attributes, %d examples\n", n_attributes, n_examples) );

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

        TRY( this->SetValues(Xt, y, nnz_max) );

        *Xt_new = Xt;
        *y_new = y;
        PetscFunctionReturnI(0);
    }
   
    typedef DataParser_<> DataParser;
}

#endif /* PARSER_HXX */

