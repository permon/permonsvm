
#include <permon/private/svmimpl.h>

PetscClassId SVM_CLASSID;

#undef __FUNCT__
#define __FUNCT__ "SVMCreate"
/*@
  SVMCreate - Creates instance of Support Vector Machine classifier

  Input Parameter:
. comm - MPI comm

  Output Parameter:
. svm_out - pointer to created SVM

  Level: beginner
@*/
PetscErrorCode SVMCreate(MPI_Comm comm,SVM *svm_out)
{
  SVM svm;

  PetscFunctionBegin;
  PetscValidPointer(svm_out,2);

#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  TRY( SVMInitializePackage() );
#endif
  TRY( PetscHeaderCreate(svm,SVM_CLASSID,"SVM","SVM Classifier","SVM",comm,SVMDestroy,SVMView) );

  svm->C = 1.;

  svm->setupcalled          = PETSC_FALSE;
  svm->setfromoptionscalled = PETSC_FALSE;
  *svm_out = svm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMReset"
/*@
  SVMReset - Resets a SVM context

  Collective on SVM

  Input Parameter:
. svm - the SVM

  Level: beginner
@*/
PetscErrorCode SVMReset(SVM svm)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  TRY( (*svm->ops->reset)(svm) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMDestroyDefault"
/*@
   SVMDestroyDefault - Destroys SVM content

   Input parameter:
.  svm - instance of SVM

   Developers Note: This is PETSC_EXTERN because it may be used by user written plugin SVM implementations

   Level: developer
@*/
PetscErrorCode SVMDestroyDefault(SVM svm)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( PetscFree(svm->data) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMDestroy"
/*@
   SVMDestroy - Destroys SVM context

   Collective on SVM

   Input Parameters:
.  svm - SVM context

   Level: beginner
@*/
PetscErrorCode SVMDestroy(SVM *svm)
{

  PetscFunctionBegin;
  if (!*svm) PetscFunctionReturn(0);

  PetscValidHeaderSpecific(*svm,SVM_CLASSID,1);
  if (--((PetscObject) (*svm))->refct > 0) {
    *svm = 0;
    PetscFunctionReturn(0);
  }

  TRY( SVMReset(*svm) );
  if ((*svm)->ops->destroy) {
    TRY( (*(*svm)->ops->destroy)(*svm) );
  }

  TRY( PetscHeaderDestroy(svm) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetFromOptions"
/*@
  SVMSetFromOptions - Sets SVM options from the options database

  Input Parameter:
. svm - the SVM

  Level: beginner
@*/
PetscErrorCode SVMSetFromOptions(SVM svm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  ierr = PetscObjectOptionsBegin((PetscObject)svm);CHKERRQ(ierr);
  if (svm->ops->setfromoptions) {
    TRY( svm->ops->setfromoptions(PetscOptionsObject,svm) );
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMSetType"
/*@

@*/
PetscErrorCode SVMSetType(SVM svm,const SVMType type) 
{
  PetscErrorCode (*create_svm)(SVM);
  PetscBool issame = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  PetscValidCharPointer(type,2);

  TRY( PetscObjectTypeCompare((PetscObject) svm,type,&issame) );
  if (issame) PetscFunctionReturn(0);

  TRY( PetscFunctionListFind(SVMList,type,(void(**)(void))&create_svm) );
  if (!create_svm) FLLOP_SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested SVM type %s",type);

  /* Destroy the pre-existing private SVM context */
  if (svm->ops->destroy) svm->ops->destroy(svm);
  /* Reinitialize function pointers in SVMOps structure */
  TRY( PetscMemzero(svm->ops,sizeof(struct _SVMOps)) );

  TRY( (*create_svm)(svm) );
  TRY( PetscObjectChangeTypeName((PetscObject)svm,type) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMView"
/*@
   SVMView - Views classification model details

   Input Parameters:
+  svm - the SVM
-  v - visualization context

   Level: beginner
@*/
PetscErrorCode SVMView(SVM svm,PetscViewer v)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMTrain"
/*@
  SVMTrain - Creates a classification model on the basis of training samples

  Input Parameters:
. svm - the SVM

  Level: beginner
@*/
PetscErrorCode SVMTrain(SVM svm)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);
  TRY( svm->ops->train(svm) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVMTest"
/*@
  SVMTest - Tests quality of classification model

  Input Parameters:
+ svm - the SVM
. Xt_test - matrix of tested samples
- y_known - known labels of tested samples

  Output Parameters:
+ N_all - number of all tested samples
- N_eq  - number of right classified samples

  Level: beginner
@*/
PetscErrorCode SVMTest(SVM svm,Mat Xt_test,Vec y_known,PetscInt *N_all,PetscInt *N_eq)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svm,SVM_CLASSID,1);

  PetscFunctionReturn(0);
}
