import sys
import os.path

from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file

import h5py

mem = Memory("./mycache")

@mem.cache
def get_data(file_libsvm):
    data = load_svmlight_file(file_libsvm)
    return data[0], data[1]


def ml_tool_libsvm_hdf5(file_libsvm, file_hdf5, path=""):
    Xs, y = get_data(file_libsvm)

    nfeatures = Xs.shape[1]

    output_file_h5_path = os.path.join(path, file_hdf5)
    with h5py.File(output_file_h5_path, 'w') as hf:
        hf.create_dataset("y", data=y)
        grp = hf.create_group("X")
        grp.attrs["MATLAB_sparse"] = nfeatures
        grp.create_dataset("data", data=Xs.data)
        grp.create_dataset("jc", data=Xs.indptr)
        grp.create_dataset("ir", data=Xs.indices)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        ml_tool_libsvm_hdf5(sys.argv[1], sys.argv[2])

    if len(sys.argv) == 4:
        ml_tool_libsvm_hdf5(sys.argv[1], sys.argv[2], sys.argv[3])
