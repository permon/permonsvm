#ifndef PARSER_HXX
#define PARSER_HXX
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>


namespace excape {
    template<typename T=int>
    class DataParser_ {
        private:
            std::string filename;
            
            bool ParseSourceFileLine(std::string &, std::vector<T> &, T &);
            bool ParseSourceFileLine(std::string &, std::vector<T> &);
            bool GetMatrixStructure(std::vector<T> &, T &, T &, int n_examples=-1);
            PetscErrorCode SetValues(Mat Xt, Vec y, PetscInt nnz_max);
        public:
            void SetInputFileName(const std::string &);
            PetscErrorCode GetData(MPI_Comm, PetscInt, Mat *, Vec *);
    };
    
    template<typename T>
    bool DataParser_<T>::ParseSourceFileLine(std::string &line, std::vector<T> &cols, T &yi) {
        using namespace std;
        using namespace boost;

        regex r("\\s+|:");

        vector<string> parsed_values;
        int col;

        algorithm::split_regex(parsed_values, line, r);

        yi = boost::lexical_cast<T>(parsed_values[0]) ? 1 : -1;

        cols.clear();
        for (unsigned int i = 1; i < parsed_values.size(); i += 2) {
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
    
    template<typename T>
    bool DataParser_<T>::ParseSourceFileLine(std::string &line, std::vector<T> &cols) {
        using namespace std;
        using namespace boost;

        regex r("\\s+|:");

        vector<string> parsed_values;
        int col;

        algorithm::split_regex(parsed_values, line, r);

        cols.clear();
        for (unsigned int i = 1; i < parsed_values.size(); i += 2) {
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
    
    template<typename T>
    bool DataParser_<T>::GetMatrixStructure(std::vector<T> &nnz_per_row, T &nnz_max, T &num_cols, int n_examples) {
        using namespace std;

        ifstream fl_hldr;
        string tmp_line; vector<T> tmp_vec;
        T tmp_cols, nnz;
        int i;

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
                if (n_examples != -1 && i == n_examples) break;
            }
        } else {
            return (false);
        }

        return (true);
    }
    
    template<typename T>
    PetscErrorCode DataParser_<T>::SetValues(Mat Xt, Vec y, PetscInt nnz_max) {
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
    
    template<typename T>
    void DataParser_<T>::SetInputFileName(const std::string &filename) {
        this->filename = filename;
    }
    
    template<typename T>
    PetscErrorCode DataParser_<T>::GetData(MPI_Comm comm, PetscInt n_examples, Mat *Xt_new, Vec *y_new)
    {
        using namespace std;
        vector<PetscInt> nnz_per_row;
        PetscInt n_attributes, nnz_max;
        Mat Xt;
        Vec y;

        PetscFunctionBeginI;
        TRY( !this->GetMatrixStructure(nnz_per_row, nnz_max, n_attributes, n_examples) );
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

        TRY( this->SetValues(Xt, y, nnz_max) );

        *Xt_new = Xt;
        *y_new = y;
        PetscFunctionReturnI(0);
    }
   
    typedef DataParser_<> DataParser;
}

#endif /* PARSER_HXX */

