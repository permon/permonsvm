
#if !defined(__PROBIMPL_H)
#define	__PROBIMPL_H

#include <permon/private/svmimpl.h>

typedef struct {
  Mat Xt_training;
  Vec y_training;

  Mat Xt_calib;
  Vec y_calib;


  SVM inner;
  Tao tao;

  PetscReal *target;
  PetscReal *deci;

  PetscInt  Np_calib;
  PetscInt  Nn_calib;

  PetscBool labels_to_target_probs;

} SVM_Probability;

#endif
