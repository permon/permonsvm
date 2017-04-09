#
# This is the makefile for installing PermonDummy, the PERMON template package <http://permon.it4i.cz/>.
#

ALL: permonsvm-all
LOCDIR = .
DIRS   = src include

include lib/permonsvm/conf/permonsvm_variables
include lib/permonsvm/conf/permonsvm_rules

permonsvm-all: permon-all-legacy

