
#if !defined(__PROBIMPL_H)
#define	__PROBIMPL_H

#include <permon/private/svmimpl.h>

typedef struct {
  Mat       Xt_training;
  Vec       y_training;

  Mat       Xt_calib;
  Vec       y_calib;

  PetscInt  Np_calib;
  PetscInt  Nn_calib;

  SVM       inner;  // TODO change name to uncalibrated

  Vec       vec_dist;
  Vec       vec_targets;

  PetscBool labels_to_target_probs;

  Tao       tao;

  Vec       sub_work[2];
  Vec       *work;

  PetscReal sigmoid_params[2];
  PetscReal threshold;

  PetscInt  confusion_matrix[4];
  PetscReal model_scores[15];

} SVM_Probability;

#endif
