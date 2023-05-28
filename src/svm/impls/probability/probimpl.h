
#if !defined(__PROBIMPL_H)
#define	__PROBIMPL_H

#include <permon/private/svmimpl.h>

typedef struct {
  Mat       Xt_training;
  Vec       y_training;

  Mat       Xt_calib;
  Vec       y_calib;

  SVM       inner;
  Tao       tao;

  PetscReal *target; // TODO as Vec
  PetscReal *deci;   // TODO as Vec

  PetscReal sigmoid_params[2];
  PetscReal threshold;

  PetscInt  Np_calib;
  PetscInt  Nn_calib;

  PetscBool labels_to_target_probs;

  PetscInt  confusion_matrix[4]; // TODO move to _p_SVM
  PetscReal model_scores[15];    // TODO move to _p_SVM

} SVM_Probability;

#endif
