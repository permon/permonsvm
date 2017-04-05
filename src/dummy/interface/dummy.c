
#include <private/dummyimpl.h>

#undef __FUNCT__
#define __FUNCT__ "PermonDummy"
/*@
   PermonDummy - Dummy function.
   
   Not Collective
   
   Input Parameter:
+  a - integer a
-  b - integer b
   
   Output Parameter:
.  c - integer c = a + b 
   
   Level: beginner

.seealso: PetscInt
@*/
PetscErrorCode PermonDummy(PetscInt a, PetscInt b, PetscInt *c)
{
  PetscFunctionBeginI;
  *c = a + b;
  PetscFunctionReturnI(0);
}

