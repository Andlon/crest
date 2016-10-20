#pragma once

#include <cstdlib>
#include <cstdint>

extern "C"
{
    typedef struct flat_mesh_data
    {
        // Vertices stored as [x1 y1 x2 y2 x3 y3 ... ]
        double * vertices;
        // Elements stored as [e1_1 e1_2 e1_3 e2_1 e2_2 e2_3 ... ]
        int32_t * elements;
        size_t vertices_size;
        size_t elements_size;
    } flat_mesh_data;

    void delete_flat_mesh_data(const flat_mesh_data * data);
    void delete_flat_mesh_data_array(const flat_mesh_data ** data_array, size_t size);

    flat_mesh_data * bisect_to_tolerance(const flat_mesh_data * mesh_data, double tolerance);

    /**
     * Refines the initial mesh in accordance with the Threshold algorithm.
     * @param initial_mesh
     * @param tolerance
     * @param corner_indices
     * @param corner_radians
     * @param num_corners
     * @return A two-element array of flat_mesh_data pointers.
     *         Must be explicitly deleted with delete_flat_mesh_data_array.
     */
    flat_mesh_data ** threshold(const flat_mesh_data * initial_mesh,
                               double tolerance,
                               const int32_t * corner_indices,
                               const double * corner_radians,
                               size_t num_corners);
}