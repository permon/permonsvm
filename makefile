#
# This is the makefile for installing PermonDummy, the PERMON template package <http://permon.it4i.cz/>.
#

ALL: permonsvm-all
LOCDIR = .
DIRS   = src include

include lib/permonsvm/conf/permonsvm_variables
include lib/permonsvm/conf/permonsvm_rules

permonsvm-all: permon-all-legacy

permonsvm:  ${PERMON_SVM_DIR}/${PETSC_ARCH}/bin/permonsvm

src/bin/permonsvm.o: src/bin/permonsvm.cxx
	${PETSC_CXXCOMPILE_SINGLE} -std=c++11 `pwd`/$<

${PERMON_SVM_DIR}/${PETSC_ARCH}/bin/permonsvm: src/bin/permonsvm.o ${PERMON_SVM_LIB_DIR}/lib*
	@${MKDIR} $(dir $@)
	@${CLINKER} -o $@ $< -lboost_regex ${PERMON_SVM_LIB}
	-@${RM} $<
	-@echo executable $@ created
	-@echo "========================================="

cleanbin:                          
	-@${RM} ${PERMON_SVM_DIR}/${PETSC_ARCH}/bin/*      
