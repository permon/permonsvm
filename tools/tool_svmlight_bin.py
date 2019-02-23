import sys
import os.path

from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file

import petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

mem = Memory("./mycache")

@mem.cache
def get_data(file_libsvm):
    data = load_svmlight_file(file_libsvm)
    return data[0], data[1]


def ml_tool_svmlight_bin(file_libsvm, file_bin, path=""):
    X, y = get_data(file_libsvm)
    m = X.shape[0]
    n = X.shape[1]

    # create PETSc Mat from scipy matrix
    Xp = PETSc.Mat().create(PETSc.COMM_SELF)
    Xp.setSizes([m, n])
    Xp.setType('aij')
    Xp.setName('X')
    Xp.setFromOptions()

    ai = X.indptr
    aj = X.indices
    av = X.data

    Xp.setPreallocationCSR((ai, aj))
    Xp.setValuesCSR(ai, aj, av)
    Xp.assemble()

    # create PETSc Vec from numpy array
    yp = PETSc.Vec().create(PETSc.COMM_SELF)
    yp.setSizes(m)
    yp.setName('y')
    yp.setFromOptions()
    yp.setArray(y)

    # save PETSc Mat and Vec
    output_file_bin_path = os.path.join(path, file_bin)
    viewer = PETSc.Viewer().createBinary(output_file_bin_path, 'w')
    viewer(Xp)
    viewer(yp)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        ml_tool_svmlight_bin(sys.argv[1], sys.argv[2])

    if len(sys.argv) == 4:
        ml_tool_svmlight_bin(sys.argv[1], sys.argv[2], sys.argv[3])
