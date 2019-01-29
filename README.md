PermonSVM - the PERMON SVM classifier
====================================

PERMON project homepage: <http://permon.vsb.cz>  
PermonSVM homepage: <http://permon.vsb.cz/permonsvm.htm>
GitHub: <https://github.com/permon/permonsvm>

Please use [GitHub](https://github.com/permon/permonsvm) for issues and pull requests.

Features
--------

- The scalable parallel solution for the linear C-SVM 
- Supported classifications:	
	- binary classification
	- no-bias binary classification
- Misclassification error quantification:
	- _l1_ hinge-loss function
	- _l2_ hinge-loss function
- Bias classification solvers: 
	-  SMALXE + MPRGP (active-set method for bound constrained problems) 
	-  SMALXE + [The Toolkit for Advance Optimization](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Tao/index.html) (TAO) solvers for minimization with bound constraints
- No-bias classification solvers:
	- MPRGP
	- TAO solvers for minimization with bound constraints 
- Warm start  
- Grid search
- Cross validation types:
	- k-fold 
	- stratified k-fold
- Model perfomance scores:
	- accuracy
	- sensitivity
	- specifity
	- F1
	- Matthews correlation coefficient 
	- Area Under Curve (AUC) Receiver Operating Characteristics (ROC)
	- Gini coefficient
- Parallel SVMLight format loader

Quick guide to PermonSVM installation
-------------------------------------

1. install [PermonQP](https://github.com/permon/permon)
2. set `PERMON_SVM_DIR` variable pointing to the PermonSVM directory (probably this file's parent directory)
3. build PermonSVM simply using makefile (makes use of PETSc buildsystem):
   `make`
4. if the build is successful, there is a new subdirectory named `$PETSC_ARCH` with the program library `$PETSC_ARCH/lib/libpermonsvm.{so,a}` and the executable `$PETSC_ARCH/bin/permonsvmfile`
   - shared library (.so) is built just if PETSc has been configured with option `--with-shared-libraries`
   - all compiler settings are inherited from PETSc

Example of PermonSVM usage
--------------------------

1. running SVM on 2 MPI processes with default settings (_l1_ hinge loss function, C = 1, no-bias classification)
   `./runsvmmpi 2 -f_training examples/heart_scale -f_test examples/heart_scale.t`
2. running PermonSVM on 2 MPI processes with grid search and k-fold cross validation
   `./runsvmmpi 2 -f_training examples/heart_scale -f_test examples/heart_scale.t -svm_C -1`
3. running PermonSVM on 2 MPI processes with grid search (C = {0.1, 1, 10, 100}) and cross validation on 4 folds with warm start
   `./runsvmmpi 2 -f_training examples/heart_scale -f_test examples/heart_scale.t -svm_C -1 -svm_nfolds 4 -svm_logC_base 10 -svm_logC_min -1 -svm_logC_max 2 -cross_svm_warm_start 1`
4. running PermonSVM on 2 MPI processes with grid search (C = {0.1, 1, 10, 100}) and stratified k-fold cross validation on 4 folds with warm start
   `./runsvmmpi 2 -f_training examples/heart_scale -f_test examples/heart_scale.t -svm_C -1 -svm_nfolds 4 -svm_logC_base 10 -svm_logC_min -1 -svm_logC_max 2 -cross_svm_warm_start 1 -svm_cv_type stratified_kfold`   
5. running PermonSVM on 2 MPI processes with C = 100
   `./runsvmmpi 2 -f_training examples/heart_scale -f_test examples/heart_scale.t -svm_C 100`
6. running PermonSVM on 2 MPI processes with C = 0.01 and _l2_ hinge loss function
   `./runsvmmpi 2 -f_training examples/heart_scale -f_test examples/heart_scale.t -svm_loss_type L2 -svm_C 1e-2`
7. running PermonSVM on 2 MPI processes with C = 0.01 and _l2_ hinge loss function (bias (standard) classification)
	`./runsvmmpi 2 -f_training examples/heart_scale -f_test examples/heart_scale.t -svm_loss_type L2 -svm_C 1e-2 -svm_binary_mod 1`

The training dataset `examples/heart_scale` and testing dataset `examples/heart_scale.t` have been obtained by splitting the `heart_scale` dataset from the [LIBSVM dataset page](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#heart).

Currently supported PERMON/PETSc versions
----------------------------------
PERMON tries to support newest versions of PETSc as soon as possible. The [releases](https://github.com/It4innovations/permonsvm/releases) are tagged with major.minor.sub-minor numbers. The major.minor numbers correspond to the major.minor release numbers of the supported PERMON/PETSc version.
