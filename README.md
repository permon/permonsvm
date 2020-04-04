PermonSVM - the PERMON SVM classifier
====================================

PERMON project homepage: <http://permon.vsb.cz>  
PermonSVM homepage: <http://permon.vsb.cz/permonsvm.htm>

Please use [GitHub](https://github.com/permon/permonsvm) for issues and pull requests.

Feature overview
-----------------

- Scalable (parallel) solution for the linear C-SVM 
- Supported binary classifications:	
	- standard classification (linear and bound constraints)
	- relaxed-bias classification (bound constraints)
- Misclassification error quantification:
	- _l1_ hinge-loss function
	- _l2_ hinge-loss function
- Standard classification solvers: 
	-  SMALXE + MPRGP (active-set method for bound constrained problems) 
	-  SMALXE + [The Toolkit for Advance Optimization](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Tao/index.html) (TAO) solvers for minimization with bound constraints
- Relaxed-bias classification solvers:
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
- Parallel data loaders:
	- PETSc binary
	- HDF5 (AIJ and dense matrices)
	- SVMLight

Quick installation guide
-------------------------------------

1. install [PermonQP](https://github.com/permon/permon) (follow instructions in its own [README.md](https://github.com/permon/permon/blob/master/README.md))
2. set `PERMON_SVM_DIR` variable pointing to the PermonSVM directory (probably this file's parent directory)
3. build PermonSVM simply using makefile (makes use of PETSc buildsystem):
   `make`
4. if the build is successful, there is a new subdirectory named `$PETSC_ARCH` with the program library `$PETSC_ARCH/lib/libpermonsvm.{dylib,so,a}` and the executable `$PETSC_ARCH/bin/permonsvmfile`
   - shared library (.so) is built just if PETSc has been configured with option `--with-shared-libraries`
   - all compiler settings are inherited from PETSc and PermonQP

Tutorials
--------------------------

* Tutorials illustrating basic functionality of the package are located in [`src/tutorials`](https://github.com/permon/permonsvm/tree/master/src/tutorials).
* We also provide the bash script [runsvmmpi](https://github.com/permon/permonsvm/tree/master/runsvmmpi) in the root directory of PermonSVM to easily run minimal working example [`src/bin/permonsvmfile.c`](https://github.com/permon/permonsvm/tree/master/src/bin/permonsvmfile.c).
* Several training and test datasets are located in `DATA_DIR=src/tutorials/data`.
* Please set the `DATA_DIR` variable before running following examples.

### Using different classification methods

1. running PermonSVM on 2 MPI processes with default settings (relaxed-bias classification, _l1_ hinge loss, C = 1, B = 1)
   
 	```bash 
 	./runsvmmpi 2 -f_training $DATA_DIR/heart_scale.bin -f_test $DATA_DIR/heart_scale.t.bin
 	```
  
2. running PermonSVM on 2 MPI processes with penalty parameter C = 100 
	
	```bash
	./runsvmmpi 2 -f_training $DATA_DIR/heart_scale.bin -f_test $DATA_DIR/heart_scale.t.bin \
	  -svm_C 100
	```
   
3. running PermonSVM on 2 MPI processes with C = 0.01 and _l2_ hinge loss

	```bash
	./runsvmmpi 2 -f_training $DATA_DIR/heart_scale.bin -f_test $DATA_DIR/heart_scale.t.bin \
	  -svm_loss_type L2 -svm_C 1e-2
	```
  
4. running PermonSVM on 2 MPI processes solving standard classification problem (binary mod 1), missclassification error quantification by _l2_ hinge loss, and C = 0.01
	
	```bash
	./runsvmmpi 2 -f_training $DATA_DIR/heart_scale.bin -f_test $DATA_DIR/heart_scale.t.bin \
	  -svm_loss_type L2 -svm_C 1e-2 -svm_binary_mod 1
	```
	
### Hyperparameter optimization

1. running PermonSVM on 2 MPI processes with hyperparameter optimization with default settings (_l1_ hinge loss function, relaxed-bias classification, grid-search log2C = [-2:1:2], k-fold cross validation on 5 folds)

	```bash
	./runsvmmpi 2 -f_training $DATA_DIR/heart_scale.bin -f_test $DATA_DIR/heart_scale.t.bin \
	  -svm_hyperopt 1
	```
   
2. running PermonSVM on 2 MPI processes with grid-search on C = {0.1, 1, 10, 100} combined with cross validation on 3 folds that reuses a previous solution (warm start)

	```bash
	./runsvmmpi 2 -f_training $DATA_DIR/heart_scale.bin -f_test $DATA_DIR/heart_scale.t.bin \
	  -svm_hyperopt 1 -svm_gs_logC_base 10 -svm_gs_logC_stride 1,2,1 -svm_nfolds 3 -cross_svm_warm_start 1
	```
  
3. running PermonSVM on 2 MPI processes with grid search on C = {0.1, 1, 10, 100} and stratified k-fold cross validation on 3 folds with warm start

	```bash
	./runsvmmpi 2 -f_training $DATA_DIR/heart_scale.bin -f_test $DATA_DIR/heart_scale.t.bin \
	  -svm_hyperopt 1 -svm_gs_logC_base 10 -svm_gs_logC_stride 1,2,1 -svm_nfolds 3 -cross_svm_warm_start 1 \
	  -svm_cv_type stratified_kfold
	```
   
### Using precomputed Gramian matrix

PermonSVM uses an implicit representation of the Gramian matrix by default.
Sometimes, it is reasonable to compute inner products related to the Gramian explicitly, typically, when a number of features is disproportionately larger than a number of samples.
For such cases, PermonSVM provides functionality allowing to load precomputed Gramian matrix.

```bash
./runsvmmpi 2 -f_training $DATA_DIR/heart_scale.bin -f_test $DATA_DIR/heart_scale.t.bin \
  -f_kernel $DATA_DIR/heart_scale.kernel.bin
```

The training dataset `src/tutorials/data/heart_scale` and testing dataset `src/tutorials/data/heart_scale.t` have been obtained by splitting the `heart_scale` dataset from the [LIBSVM dataset page](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#heart).


Currently supported PERMON/PETSc versions
----------------------------------
PERMON tries to support newest versions of PETSc as soon as possible. The [releases](https://github.com/It4innovations/permonsvm/releases) are tagged with major.minor.sub-minor numbers. The major.minor numbers correspond to the major.minor release numbers of the supported PERMON/PETSc version.
