#
# This is the makefile for installing PermonDummy, the PERMON template package <http://permon.it4i.cz/>.
#

ALL: all
all: permonsvm-all permonsvmfile
LOCDIR = .
DIRS   = src include

include lib/permonsvm/conf/permonsvm_variables
include lib/permonsvm/conf/permonsvm_rules

# turn off gmake build explicitly as it is not implemented in this PERMON package
MAKE_IS_GNUMAKE :=

permonsvm-all: chk_permon_dir permon-all

cleanbin:
	-@${RM} ${PERMON_SVM_DIR}/${PETSC_ARCH}/bin/*

clean:: allclean

#
# Check if PERMON_DIR variable specified is valid
#
chk_permon_dir:
	@if [ -z ${PERMON_DIR} ]; then \
          printf ${PETSC_TEXT_HILIGHT}"*************************ERROR**************************************\n"; \
	  echo "PERMON_DIR not specified!"; \
          printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n"; \
    false; fi
	@if [ ! -f ${PERMON_DIR}/${PERMON_MAIN_HEADER} ]; then \
          printf ${PETSC_TEXT_HILIGHT}"*************************ERROR**************************************\n"; \
	  echo "Incorrect PERMON_DIR specified: ${PERMON_DIR}!"; \
    echo 'File ${PERMON_DIR}/${PERMON_MAIN_HEADER} does not exist.'; \
	  echo "Note: You need to use / to separate directories, not \\"; \
          printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n"; \
	  false; fi

