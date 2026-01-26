#ifndef KMEANS_H
#define KMEANS_H

void kmeans_fit(
    const double* data,
    int n_samples,
    int n_features,
    int k,
    int max_iters,
    int* labels,
    double* centroids
);

#endif
