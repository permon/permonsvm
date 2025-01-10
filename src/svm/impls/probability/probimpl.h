#pragma once

#include <permon/private/svmimpl.h>

typedef struct {
  Mat Xt_training;
  Vec y_training;

  Mat Xt_calib;
  Vec y_calib;

  PetscInt Np_calib;
  PetscInt Nn_calib;

  SVM inner;

  Vec vec_dist;
  Vec vec_targets;

  PetscBool labels_to_target_probs;

  Tao tao;

  Vec *work_vecs;
  Vec  work_sub[2];

  PetscReal sigmoid_params[2];
  PetscReal threshold;

  PetscInt  confusion_matrix[4];
  PetscReal model_scores[16];

} SVM_Probability;
