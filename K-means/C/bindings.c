#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include "K-means.h"

static PyObject* py_kmeans_fit(PyObject* self, PyObject* args) {
    PyArrayObject* X;
    int k, max_iters;

    if (!PyArg_ParseTuple(args, "O!ii",
        &PyArray_Type, &X, &k, &max_iters)) {
        return NULL;
    }

    X = (PyArrayObject*) PyArray_FROM_OTF(
        (PyObject*)X,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY
    );

    if (!X) return NULL;

    int n_samples = PyArray_DIM(X, 0);
    int n_features = PyArray_DIM(X, 1);

    double* data = (double*) PyArray_DATA(X);

    // Allocate outputs
    npy_intp labels_dim[1] = { n_samples };
    PyArrayObject* labels =
        (PyArrayObject*) PyArray_SimpleNew(1, labels_dim, NPY_INT32);

    npy_intp centroids_dim[2] = { k, n_features };
    PyArrayObject* centroids =
        (PyArrayObject*) PyArray_SimpleNew(2, centroids_dim, NPY_DOUBLE);

    int* labels_data = (int*) PyArray_DATA(labels);
    double* centroids_data = (double*) PyArray_DATA(centroids);

    // Initialize centroids (simple random pick)
    for (int c = 0; c < k; c++) {
        int idx = rand() % n_samples;
        memcpy(
            &centroids_data[c * n_features],
            &data[idx * n_features],
            n_features * sizeof(double)
        );
    }

    // 🚀 Release GIL (important!)
    Py_BEGIN_ALLOW_THREADS
    kmeans_fit(
        data,
        n_samples,
        n_features,
        k,
        max_iters,
        labels_data,
        centroids_data
    );
    Py_END_ALLOW_THREADS

    Py_DECREF(X);

    return Py_BuildValue("NN", labels, centroids);
}
static PyMethodDef KMeansMethods[] = {
    {
        "fit",
        py_kmeans_fit,
        METH_VARARGS,
        "Run K-Means clustering (C implementation)"
    },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "kmeans_c",
    NULL,
    -1,
    KMeansMethods,
    NULL, //m_slots
    NULL, //m_traverse
    NULL, //m_clear
    NULL, //m_free
};

PyMODINIT_FUNC PyInit_kmeans_c(void) {
    import_array();
    return PyModule_Create(&kmeansmodule);
}
