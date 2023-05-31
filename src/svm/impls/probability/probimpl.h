
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

  // PetscReal *target; // TODO remove
  // PetscReal *deci;   // TODO remove

  Vec       vec_dist;
  Vec       vec_targets;

  PetscReal sigmoid_params[2];
  PetscReal threshold;

  PetscInt  Np_calib;
  PetscInt  Nn_calib;

  PetscBool labels_to_target_probs;

  PetscInt  confusion_matrix[4];
  PetscReal model_scores[15];

} SVM_Probability;

#endif
