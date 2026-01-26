#include <stdlib.h>
#include <float.h>
#include <string.h>
#include "K-means.h"

static double squared_distance(
    const double* a,
    const double* b,
    int n_features
) {
    double dist = 0.0;
    for (int i = 0; i < n_features; i++) {
        double diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

void kmeans_fit(
    const double* data,
    int n_samples,
    int n_features,
    int k,
    int max_iters,
    int* labels,
    double* centroids
) {
    double* centroid_sums = malloc(k * n_features * sizeof(double));
    int* counts = malloc(k * sizeof(int));

    for (int iter = 0; iter < max_iters; iter++) {
        for (int i = 0; i < n_samples; i++) {
            double min_dist = DBL_MAX;
            int best = 0;

            for (int c = 0; c < k; c++) {
                double dist = squared_distance(
                    &data[i * n_features],
                    &centroids[c * n_features],
                    n_features
                );

                if (dist < min_dist) {
                    min_dist = dist;
                    best = c;
                }
            }
            labels[i] = best;
        }
        memset(centroid_sums, 0, k * n_features * sizeof(double));
        memset(counts, 0, k * sizeof(int));

        for (int i = 0; i < n_samples; i++) {
            int c = labels[i];
            counts[c]++;
            for (int f = 0; f < n_features; f++) {
                centroid_sums[c * n_features + f] +=
                    data[i * n_features + f];
            }
        }

        for (int c = 0; c < k; c++) {
            if (counts[c] == 0) continue;
            for (int f = 0; f < n_features; f++) {
                centroids[c * n_features + f] =
                    centroid_sums[c * n_features + f] / counts[c];
            }
        }
    }

    free(centroid_sums);
    free(counts);
}
