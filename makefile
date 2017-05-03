#
# This is the makefile for installing PermonDummy, the PERMON template package <http://permon.it4i.cz/>.
#

ALL: permonsvm-all
all: permonsvm-all
LOCDIR = .
DIRS   = src include

include lib/permonsvm/conf/permonsvm_variables
include lib/permonsvm/conf/permonsvm_rules

# turn off gmake build explicitly as it is not implemented in this PERMON package
MAKE_IS_GNUMAKE :=

permonsvm-all: permon-all

permonsvmfile:  ${PERMON_SVM_DIR}/${PETSC_ARCH}/bin/permonsvmfile

src/bin/permonsvmfile.o: src/bin/permonsvmfile.cxx
	${PETSC_CXXCOMPILE_SINGLE} -std=c++11 `pwd`/$<

${PERMON_SVM_DIR}/${PETSC_ARCH}/bin/permonsvmfile: src/bin/permonsvmfile.o
	@${MKDIR} $(dir $@)
	@${CLINKER} -o $@ $< ${PERMON_SVM_LIB}
	-@${RM} $<
	-@echo executable $@ created
	-@echo "========================================="

cleanbin:
	-@${RM} ${PERMON_SVM_DIR}/${PETSC_ARCH}/bin/*

clean:: allclean

