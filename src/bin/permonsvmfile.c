#include <permonsvm.h>

static MPI_Comm comm;

#define h5       "h5"
#define bin      "bin"
#define SVMLight "svmlight"

#undef __FUNCT__
#define __FUNCT__ "GetFilenameExtension"
PetscErrorCode GetFilenameExtension(const char *filename, char **extension)
{
  char *extension_inner;

  PetscFunctionBegin;
  PetscCall(PetscStrrchr(filename, '.', &extension_inner));
  *extension = extension_inner;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "SVMRunBinaryClassification"
PetscErrorCode SVMRunBinaryClassification()
{
  SVM svm;

  char      training_file[PETSC_MAX_PATH_LEN]        = "examples/data/heart_scale.bin";
  char      test_file[PETSC_MAX_PATH_LEN]            = "";
  char      kernel_file[PETSC_MAX_PATH_LEN]          = "";
  char      training_result_file[PETSC_MAX_PATH_LEN] = "";
  char      test_result_file[PETSC_MAX_PATH_LEN]     = "";
  PetscBool test_file_set = PETSC_FALSE, kernel_file_set = PETSC_FALSE;
  PetscBool training_result_file_set = PETSC_FALSE, test_result_file_set = PETSC_FALSE;

  char *extension = NULL;

  PetscViewer viewer;
  PetscBool   ishdf5, isbinary, issvmlight;

  PetscFunctionBeginI;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f_training", training_file, sizeof(training_file), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f_test", test_file, sizeof(test_file), &test_file_set));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f_kernel", kernel_file, sizeof(kernel_file), &kernel_file_set));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-f_training_predictions", training_result_file, sizeof(training_result_file), &training_result_file_set));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f_test_predictions", test_result_file, sizeof(test_result_file), &test_result_file_set));

  PetscCall(SVMCreate(comm, &svm));
  PetscCall(SVMSetType(svm, SVM_BINARY));
  PetscCall(SVMSetFromOptions(svm));

  /* Load training dataset */
  PetscCall(GetFilenameExtension(training_file, &extension));
  PetscCall(PetscStrcmp(extension, h5, &ishdf5));
  PetscCall(PetscStrcmp(extension, bin, &isbinary));
  PetscCall(PetscStrcmp(extension, SVMLight, &issvmlight));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscCall(PetscViewerHDF5Open(comm, training_file, FILE_MODE_READ, &viewer));
#else
    SETERRQ(comm, PETSC_ERR_SUP, "PETSc is not configured with HDF5");
#endif
  } else if (isbinary) {
    PetscCall(PetscViewerBinaryOpen(comm, training_file, FILE_MODE_READ, &viewer));
  } else if (issvmlight) {
    PetscCall(PetscViewerSVMLightOpen(comm, training_file, &viewer));
  } else {
    SETERRQ(comm, PETSC_ERR_SUP, "File type %s not supported", extension);
  }
  PetscCall(SVMLoadTrainingDataset(svm, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* Load precomputed Gramian (kernel matrix), i.e. phi(X^T) * phi(X) generally */
  if (kernel_file_set) {
    PetscCall(GetFilenameExtension(kernel_file, &extension));
    PetscCall(PetscStrcmp(extension, h5, &ishdf5));
    PetscCall(PetscStrcmp(extension, bin, &isbinary));

    if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
      PetscCall(PetscViewerHDF5Open(comm, kernel_file, FILE_MODE_READ, &viewer));
#else
      SETERRQ(comm, PETSC_ERR_SUP, "PETSc is not configured with HDF5");
#endif
    } else if (isbinary) {
      PetscCall(PetscViewerBinaryOpen(comm, kernel_file, FILE_MODE_READ, &viewer));
    } else {
      SETERRQ(comm, PETSC_ERR_SUP, "File type %s not supported", extension);
    }
    PetscCall(SVMLoadGramian(svm, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  /* Load test dataset */
  if (test_file_set) {
    PetscCall(GetFilenameExtension(test_file, &extension));
    PetscCall(PetscStrcmp(extension, h5, &ishdf5));
    PetscCall(PetscStrcmp(extension, bin, &isbinary));
    PetscCall(PetscStrcmp(extension, SVMLight, &issvmlight));
    if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
      PetscCall(PetscViewerHDF5Open(comm, test_file, FILE_MODE_READ, &viewer));
#else
      SETERRQ(comm, PETSC_ERR_SUP, "PETSc is not configured with HDF5");
#endif
    } else if (isbinary) {
      PetscCall(PetscViewerBinaryOpen(comm, test_file, FILE_MODE_READ, &viewer));
    } else if (issvmlight) {
      PetscCall(PetscViewerSVMLightOpen(comm, test_file, &viewer));
    } else {
      SETERRQ(comm, PETSC_ERR_SUP, "File type %s not supported", extension);
    }
    PetscCall(SVMLoadTestDataset(svm, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* Train SVM model */
  PetscCall(SVMTrain(svm));
  /* Test performance of SVM model */
  if (test_file_set) { PetscCall(SVMTest(svm)); }

  /* Save results */
  if (training_result_file_set) {
    PetscCall(GetFilenameExtension(training_result_file, &extension));
    PetscCall(PetscStrcmp(extension, h5, &ishdf5));
    PetscCall(PetscStrcmp(extension, bin, &isbinary));
    if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
      PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, training_result_file, FILE_MODE_WRITE, &viewer));
#else
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "PETSc is not configured with HDF5");
#endif
    } else {
      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, training_result_file, FILE_MODE_WRITE, &viewer));
    }
    PetscCall(SVMViewTrainingPredictions(svm, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  if (test_result_file_set) {
    PetscCall(GetFilenameExtension(test_result_file, &extension));
    PetscCall(PetscStrcmp(extension, h5, &ishdf5));
    PetscCall(PetscStrcmp(extension, bin, &isbinary));
    if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
      PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, test_result_file, FILE_MODE_WRITE, &viewer));
#else
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "PETSc is not configured with HDF5");
#endif
    } else {
      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, test_result_file, FILE_MODE_WRITE, &viewer));
    }
    PetscCall(SVMViewTestPredictions(svm, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(SVMDestroy(&svm));
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscFunctionBegin;
  PetscCall(PermonInitialize(&argc, &argv, (char *)0, (char *)0));
  comm = PETSC_COMM_WORLD;
  PetscCall(SVMRunBinaryClassification());
  PetscCall(PermonFinalize());
  return 0;
}
