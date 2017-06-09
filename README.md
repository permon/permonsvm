PermonSVM - the PERMON SVM classifier
====================================

PERMON project homepage: <http://permon.it4i.cz>  
PermonSVM homepage: <http://permon.it4i.cz/permonsvm.htm>  
GitHub: <https://github.com/It4innovations/permonsvm>

Please use [GitHub](https://github.com/It4innovations/permonsvm) for issues and pull requests.

Quick guide to PermonSVM installation
-------------------------------------

1. install [PermonQP](https://github.com/It4innovations/permon)
2. set `PERMON_SVM_DIR` variable pointing to the PermonSVM directory (probably this file's parent directory)
3. build PermonSVM simply using makefile (makes use of PETSc buildsystem):
`make`
4. if the build is successful, there is a new subdirectory named `$PETSC_ARCH` with the program library `$PETSC_ARCH/lib/libpermonsvm.{so,a}` and the executable `$PETSC_ARCH/bin/permonsvmfile`
   - shared library (.so) is built just if PETSc has been configured with option `--with-shared-libraries`
   - all compiler settings are inherited from PETSc

Example of PermonSVM usage
--------------------------

1. running PermonSVM on 2 MPI processes with default settings (L2-norm loss function, C = 10)  
   `./runsvmmpi 2 -f examples/heart_scale -f_test examples/heart_scale.t`
2. running PermonSVM on 2 MPI processes with grid search/cross-validation  
   `./runsvmmpi 2 -f examples/heart_scale -f_test examples/heart_scale.t -svm_C -1`
3. running PermonSVM on 2 MPI processes with grid search (C = {0.1, 1, 10, 100}) and cross-validation on 4 folds  
   `./runsvmmpi 2 -f examples/heart_scale -f_test examples/heart_scale.t -svm_C -1 -svm_nfolds 4 -svm_logC_base 10 -svm_logC_min -1 -svm_logC_max 2`
4. running PermonSVM on 2 MPI processes with C = 100  
   `./runsvmmpi 2 -f examples/heart_scale -f_test examples/heart_scale.t -svm_C 100`
5. running PermonSVM on 2 MPI processes with C = 0.01 and L1-norm loss function  
   `./runsvmmpi 2 -f examples/heart_scale -f_test examples/heart_scale.t -svm_loss_type L1 -svm_C 1e-2`

The source of the files `examples/heart_scale` and `examples/heart_scale.t` is [LIBSVM dataset page](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#heart).

Currently supported PETSc versions
----------------------------------
* 3.6.\*
* 3.7.\*
