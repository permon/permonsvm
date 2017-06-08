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
4. if the build is successful, there is a new subdirectory named `$PETSC_ARCH` with the program library `$PETSC_ARCH/lib/libpermonsvm.{so,a}` and the executable `$PETSC_ARCH/bin/permonsvm`
   - shared library (.so) is built just if PETSc has been configured with option `--with-shared-libraries`
   - all compiler settings are inherited from PETSc

Example of PermonSVM usage
--------------------------



Currently supported PETSc versions
----------------------------------
* 3.6.\*
* 3.7.\*

