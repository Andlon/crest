#pragma once

#include <cstdlib>

extern "C"
{
    typedef struct flat_mesh_data
    {
        // Vertices stored as [x1 y1 x2 y2 x3 y3 ... ]
        double * vertices;
        // Elements stored as [e1_1 e1_2 e1_3 e2_1 e2_2 e2_3 ... ]
        int * elements;
        size_t vertices_size;
        size_t elements_size;
    } flat_mesh_data;

    void delete_flat_mesh_data(const flat_mesh_data * data);

    flat_mesh_data * bisect_to_tolerance(const flat_mesh_data * mesh_data, double tolerance);
}