#pragma once

#include <permon/private/svmimpl.h>

typedef struct {
  Mat Xt_training;
  Vec y_training;
  Vec y_inner;
  IS  is_p, is_n;

  PetscScalar y_map[2];
  Vec         diag;
  Mat         J;

  Mat G;

  Vec         w;
  PetscScalar b;

  PetscScalar hinge_loss, hinge_loss_p, hinge_loss_n;

  PetscReal norm_w;
  PetscReal margin;

  IS        is_sv;
  PetscInt  nsv;
  PetscReal chop_tol;

  QPS qps;

  PetscInt  confusion_matrix[4];
  PetscReal model_scores[16];

  /* Work vecs */
  Vec work[3]; /* xi, c, Xtw */

  /* Valuess of primal and dual objective functions */
  PetscReal primalObj, dualObj;
} SVM_Binary;

typedef enum {
  SVM_CONVERGED_DEFAULT,
  SVM_CONVERGED_DUALITY_GAP,
  SVM_CONVERGED_MAXIMAL_DUAL_VIOLATION
} SVMConvergedType;
PERMON_EXTERN const char *const SVMConvergedTypes[];

PERMON_EXTERN PetscErrorCode SVMCrossValidation_Binary(SVM, PetscReal[], PetscInt, PetscReal[]);
PERMON_EXTERN PetscErrorCode SVMKFoldCrossValidation_Binary(SVM, PetscReal[], PetscInt, PetscReal[]);
PERMON_EXTERN PetscErrorCode SVMStratifiedKFoldCrossValidation_Binary(SVM, PetscReal[], PetscInt, PetscReal[]);
