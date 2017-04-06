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
    template<typename T1=int, typename T2=PetscScalar>
    class DataParser_ {
        private:
            std::string filename;
            bool numbering_base;
            
            bool ParseSourceFileLine(std::string &, std::vector<T1> &, std::vector<T2> &, T1 &);
            bool ParseSourceFileLine(std::string &, std::vector<T1> &);
            bool GetMatrixStructure(std::vector<T1> &, T1 &, T1 &, int n_examples=-1);
            PetscErrorCode SetValues(Mat Xt, Vec y, PetscInt nnz_max);
        public:
            void SetInputFileName(const std::string &);
            PetscErrorCode GetData(MPI_Comm, PetscInt, Mat *, Vec *);
            void SetNumberingBase(bool incides_from_zero = false);
            
            DataParser_(void);
    };
    
    template<typename T1, typename T2>
    DataParser_<T1, T2>::DataParser_(void) {
        this->numbering_base = false;
    }
    
    template<typename T1, typename T2>
    void DataParser_<T1, T2>::SetNumberingBase(bool numbering_base) {
        this->numbering_base = numbering_base;
    }
    
    template<typename T1, typename T2>
    bool DataParser_<T1, T2>::ParseSourceFileLine(std::string &line, std::vector<T1> &cols, std::vector<T2> &vals, T1 &yi) {
        using namespace std;
        using namespace boost;

        regex r("\\s+|:");

        vector<string> parsed_values;
        T1 col;
        T2 val;
        
        algorithm::split_regex(parsed_values, line, r);        
        yi = boost::lexical_cast<T1>(parsed_values[0]); //? 1 : -1;

        cols.clear();
        vals.clear();
        for (unsigned int i = 1; i < parsed_values.size(); i += 2) {
            try {
                col = boost::lexical_cast<T1>(parsed_values[i]);
                val = boost::lexical_cast<T2>(parsed_values[i+1]);
            } catch (bad_lexical_cast &e) {
                cout << "Caught bad lexical cast with error " << e.what() << endl;
                return (false);
            } catch (...) {
                cout << "Unknown exception caught!" << endl;
                return (false);
            }
            cols.push_back(col);
            vals.push_back(val);
        }

        return (true);
    }
    
    template<typename T1, typename T2>
    bool DataParser_<T1, T2>::ParseSourceFileLine(std::string &line, std::vector<T1> &cols) {
        using namespace std;
        using namespace boost;

        regex r("\\s+|:");

        vector<string> parsed_values;
        int col;

        algorithm::split_regex(parsed_values, line, r);

        cols.clear();
        for (unsigned int i = 1; i < parsed_values.size(); i += 2) {
            try {
                col = boost::lexical_cast<T1>(parsed_values[i]);
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
    
#undef __FUNCT__
#define __FUNCT__ "GetMatrixStructure"
    template<typename T1, typename T2>
    bool DataParser_<T1, T2>::GetMatrixStructure(std::vector<T1> &nnz_per_row, T1 &nnz_max, T1 &num_cols, int n_examples) {
        using namespace std;

        ifstream fl_hldr;
        string tmp_line; vector<T1> tmp_vec;
        T1 tmp_cols, nnz;
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
                   num_cols = tmp_cols - this->numbering_base + 1;
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
#define __FUNCT__ "SetValues"
    template<typename T1, typename T2>
    PetscErrorCode DataParser_<T1, T2>::SetValues(Mat Xt, Vec y, PetscInt nnz_max) {
        using namespace std;

        ifstream fl_hldr;
        string tmp_line; vector<PetscInt> idxn; vector<PetscScalar> vals;
        PetscInt yi, m;

        PetscFunctionBeginI;
        TRY( MatGetLocalSize(Xt, &m, NULL) );
        fl_hldr.open(filename.c_str());
        TRY( fl_hldr.fail() );

        PetscInt i = 0;
        while (getline(fl_hldr, tmp_line) && i < m) {
          ParseSourceFileLine(tmp_line, idxn, vals, yi);
          std::transform(idxn.begin(), idxn.end(), idxn.begin(), [](PetscInt a) { return a-this->numbering_base; });  // decrement 1-based values of idxn to be 0-based
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

