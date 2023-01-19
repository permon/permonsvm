
#if !defined(__PROBIMPL_H)
#define	__PROBIMPL_H

#include <permon/private/svmimpl.h>

typedef struct {
  Mat Xt_training;
  Vec y_training;

  Mat Xt_calib;
  Vec y_calib;

  SVM svm_inner;

} SVM_Probability;

#endif
