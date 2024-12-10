static char help[] = "";

#include <permonsvm.h>

#define h5       "h5"
#define bin      "bin"
#define SVMLight "svmlight"

PetscErrorCode GetFilenameExtension(const char *filename,char **extension)
{
  char   *extension_inner;

  PetscFunctionBegin;
  PetscCall(PetscStrrchr(filename,'.',&extension_inner));
  *extension = extension_inner;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ViewerOpen(const char *filename,PetscFileMode file_mode,PetscViewer *viewer)
{
  PetscViewer  inner_viewer;

  char        *extension=NULL;
  PetscBool    ishdf5,isbinary,issvmlight;

  PetscFunctionBegin;
  PetscCall(GetFilenameExtension(filename,&extension));
  PetscCall(PetscStrcmp(extension,h5,&ishdf5));
  PetscCall(PetscStrcmp(extension,bin,&isbinary));
  PetscCall(PetscStrcmp(extension,SVMLight,&issvmlight));

  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,filename,file_mode,&inner_viewer));
#else
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PETSc is not configured with HDF5");
#endif
  } else if (isbinary) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,file_mode,&inner_viewer));
  } else if (issvmlight) {
    PetscCall(PetscViewerSVMLightOpen(PETSC_COMM_WORLD,filename,&inner_viewer));
  } else {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"File type %s not supported",extension);
  }

  *viewer = inner_viewer;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc,char **argv)
{
  SVM         svm;
  PetscViewer viewer;

  char training_file[PETSC_MAX_PATH_LEN]        = "data/heart_scale.bin";
  char calibration_file[PETSC_MAX_PATH_LEN]     = "data/heart_scale.bin";
  char test_file[PETSC_MAX_PATH_LEN]            = "data/heart_scale.t.bin";
  char training_result_file[PETSC_MAX_PATH_LEN] = "";
  char test_result_file[PETSC_MAX_PATH_LEN]     = "";

  PetscBool training_result_file_set = PETSC_FALSE;
  PetscBool test_result_file_set = PETSC_FALSE;

  PetscCall(PermonInitialize(&argc,&argv,(char *)0,help));

  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_training",training_file,sizeof(training_file),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_calib",calibration_file,sizeof(calibration_file),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_test",test_file,sizeof(test_file),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_training_predictions",training_result_file,sizeof(training_result_file),&training_result_file_set));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f_test_predictions",test_result_file,sizeof(test_result_file),&test_result_file_set));

  /* Create SVM object */
  PetscCall(SVMCreate(PETSC_COMM_WORLD,&svm));
  PetscCall(SVMSetType(svm,SVM_PROBABILITY));
  PetscCall(SVMSetFromOptions(svm));

  PetscCall(ViewerOpen(training_file,FILE_MODE_READ,&viewer));
  PetscCall(SVMLoadTrainingDataset(svm,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(ViewerOpen(calibration_file,FILE_MODE_READ,&viewer));
  PetscCall(SVMLoadCalibrationDataset(svm,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(ViewerOpen(test_file,FILE_MODE_READ,&viewer));
  PetscCall(SVMLoadTestDataset(svm,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(SVMTrain(svm));
  PetscCall(SVMTest(svm));

  if (training_result_file_set) {
    PetscCall(ViewerOpen(training_result_file,FILE_MODE_WRITE,&viewer));
    PetscCall(SVMViewTrainingPredictions(svm,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  if (test_result_file_set) {
    PetscCall(ViewerOpen(test_result_file,FILE_MODE_WRITE,&viewer));
    PetscCall(SVMViewTestPredictions(svm,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* Free memory */
  PetscCall(SVMDestroy(&svm));
  PetscCall(PermonFinalize());

  return 0;
}

/*TEST

  test:
    filter: grep -v MPI
    args: -uncalibrated_svm_loss_type L1 -svm_view_io
    args: -f_training $PERMON_SVM_DIR/src/tutorials/data/heart_scale.bin
    args: -f_calib $PERMON_SVM_DIR/src/tutorials/data/heart_scale.bin
    args: -f_test $PERMON_SVM_DIR/src/tutorials/data/heart_scale.t.bin
    args: -tao_type nls -tao_view -svm_view_report
    args: -svm_view_test_predictions
    args: -svm_threshold 0.47
    output_file: output/ex5.out
TEST*/
