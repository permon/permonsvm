
#if !defined(__BINARYIMPL_H)
#define	__BINARYIMPL_H

#include <permon/private/svmimpl.h>

typedef struct {
    Mat         Xt_training;
    Vec         y_training;
    Vec         y_inner;
    IS          is_p,is_n;

    PetscScalar y_map[2];
    Vec         diag;
    Mat         D;

    Vec         w;
    PetscScalar b;

    PetscScalar hinge_loss,hinge_loss_p,hinge_loss_n;

    PetscReal   norm_w;
    PetscReal   margin;

    IS          is_sv;
    PetscInt    nsv;

    QPS         qps;
    PetscInt    svm_mod;

    PetscInt    confusion_matrix[4];
    PetscReal   model_scores[5];

    /* Work vecs */
    Vec         work[3]; /* xi, c, Xtw */

    /* Valuess of primal and dual objective functions */
    PetscReal   primalObj,dualObj;
} SVM_Binary;

FLLOP_EXTERN PetscErrorCode SVMCrossValidation_Binary(SVM,PetscReal [],PetscInt,PetscReal []);
FLLOP_EXTERN PetscErrorCode SVMKFoldCrossValidation_Binary(SVM,PetscReal [],PetscInt,PetscReal []);
FLLOP_EXTERN PetscErrorCode SVMStratifiedKFoldCrossValidation_Binary(SVM,PetscReal [],PetscInt,PetscReal []);
#endif
