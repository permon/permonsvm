#if !defined(__DYNARRAY_H)
#define	__DYNARRAY_H

#include <limits.h>

#define PERMON_DARRAY_INIT_CAPACITY 4
#define PERMON_DARRAY_GROW_FACTOR 2.0

#define PermonDynamicArray_(type) \
  {                               \
    type      *data;              \
    PetscInt  capacity;           \
    PetscInt  size;               \
    PetscReal grow_factor;        \
  }

#define PermonDynamicArrayInit(a,_capacity_,_grow_factor_)      \
  TRY( PetscMalloc(_capacity_ * sizeof(*(a.data)),&(a.data)) ); \
  a.size = 0;                                                   \
  a.capacity = _capacity_;                                      \
  a.grow_factor = _grow_factor_;

#define PermonDynamicArrayClear(a) \
  TRY( PetscFree(a.data) );        \
  a.capacity = 0;                  \
  a.size = 0;

#define PermonDynamicArrayPushBack(a,v)                         \
  if (a.capacity == a.size) PermonDynamicArrayResize(a);        \
  a.data[a.size] = v;                                       \
  ++a.size;

#define PermonDynamicArrayResize(a) \
  a.capacity = a.grow_factor * a.capacity; \
  TRY( PetscRealloc(a.capacity * sizeof(*(a.data)),&a.data) ); \

#define CheckCastToInt(x) (             \
  {                                     \
    int ret = 0;                        \
    if (x >= INT_MIN && x <= INT_MAX) { \
      ret = 1;                          \
    }                                   \
    ret;                                \
  }                                     \
)

#endif
