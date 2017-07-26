set(CMAKE_C_COMPILER   ${PETSC_DIR}/${PETSC_ARCH}/bin/mpicc)
set(CMAKE_CXX_COMPILER ${PETSC_DIR}/${PETSC_ARCH}/bin/mpicxx)

set(LIBRARY_OUTPUT_PATH ${PERMON_SVM_DIR}/${PETSC_ARCH}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PERMON_SVM_DIR}/${PETSC_ARCH}/bin)

set(SOURCES_DIR_SVM_INTERFACE ${PERMON_SVM_DIR}/src/svm/interface)
set(INCLUDES_DIR_SVM ${PERMON_SVM_DIR}/include)
set(SOURCES_DIR_SVM_BIN ${PERMON_SVM_DIR}/src/bin)

set(SOURCES_LIBRARY_SVM ${SOURCES_DIR_SVM_INTERFACE}/crossval.c ${SOURCES_DIR_SVM_INTERFACE}/dlregissvm.c
        ${SOURCES_DIR_SVM_INTERFACE}/svm.c ${INCLUDES_DIR_SVM}/permonsvm.h ${INCLUDES_DIR_SVM}/permon/private/svmimpl.h)
set(SOURCES_BIN_SVM ${SOURCES_DIR_SVM_BIN}/parser.hxx ${SOURCES_DIR_SVM_BIN}/permonsvmfile.cxx)

include_directories(${INCLUDES_DIR_SVM})
include_directories(${PERMON_DIR}/include)

add_library(permonsvm SHARED ${SOURCES_LIBRARY_SVM})
target_link_libraries(permonsvm LINK_PUBLIC permonqp)
target_link_libraries(permonsvm LINK_PUBLIC ${PETSC_LIBRARIES})

add_executable(permonsvmfile ${SOURCES_BIN_SVM})
target_link_libraries(permonsvmfile LINK_PUBLIC permonqp)
target_link_libraries(permonsvmfile LINK_PUBLIC permonsvm)
target_link_libraries(permonsvmfile LINK_PUBLIC ${PETSC_LIBRARIES})
target_compile_options(permonsvmfile PRIVATE "-std=c++11")