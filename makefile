#
# This is the makefile for installing PermonDummy, the PERMON template package <http://permon.it4i.cz/>.
#

ALL: PermonDummy-all
LOCDIR = .
DIRS   = src include

include lib/PermonDummy/conf/permon_dummy_variables
include lib/PermonDummy/conf/permon_dummy_rules

PermonDummy-all: permon-all-legacy

