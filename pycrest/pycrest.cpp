#include "pycrest.h"

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/geometry/refinement.hpp>

namespace
{
    crest::IndexedMesh<double, int32_t> mesh_from_flat_data(const flat_mesh_data * mesh_data)
    {
        assert(mesh_data->vertices_size % 2 == 0);
        assert(mesh_data->elements_size % 3 == 0);

        const auto num_vertices = mesh_data->vertices_size / 2;
        const auto num_elements = mesh_data->elements_size / 3;

        using crest::Vertex;
        using crest::Element;

        std::vector<crest::Vertex<double>> vertices;
        vertices.reserve(num_vertices);
        for (size_t v = 0; v < num_vertices; ++v)
        {
            const auto x = mesh_data->vertices[2 * v + 0];
            const auto y = mesh_data->vertices[2 * v + 1];
            vertices.emplace_back(Vertex<double>(x, y));
        }

        std::vector<crest::Element<int32_t>> elements;
        elements.reserve(num_elements);
        for (size_t e = 0; e < num_elements; ++e)
        {
            const auto a = mesh_data->elements[3 * e + 0];
            const auto b = mesh_data->elements[3 * e + 1];
            const auto c = mesh_data->elements[3 * e + 2];
            elements.emplace_back(Element<int32_t>({a, b, c}));
        }

        return crest::IndexedMesh<double, int32_t>(vertices, elements);
    };

    flat_mesh_data * flat_data_from_mesh(crest::IndexedMesh<double, int32_t> mesh)
    {
        const size_t num_vertex_entries = static_cast<size_t>(2 * mesh.num_vertices());
        const size_t num_element_entries = static_cast<size_t>(3 * mesh.num_elements());

        double * vertex_buffer = new double[num_vertex_entries];
        int32_t * index_buffer = new int32_t[num_element_entries];

        for (size_t v = 0; v < static_cast<size_t>(mesh.num_vertices()); ++v)
        {
            const auto vertex = mesh.vertices()[v];
            vertex_buffer[2 * v + 0] = vertex.x;
            vertex_buffer[2 * v + 1] = vertex.y;
        }

        for (size_t e = 0; e < static_cast<size_t>(mesh.num_elements()); ++e)
        {
            const auto indices = mesh.elements()[e].vertex_indices;
            index_buffer[3 * e + 0] = indices[0];
            index_buffer[3 * e + 1] = indices[1];
            index_buffer[3 * e + 2] = indices[2];
        }

        return new flat_mesh_data {
                vertex_buffer,
                index_buffer,
                num_vertex_entries,
                num_element_entries
        };
    }
}

extern "C"
{
    flat_mesh_data * bisect_to_tolerance(const flat_mesh_data * mesh_data, double tolerance)
    {
        const auto mesh = crest::bisect_to_tolerance(mesh_from_flat_data(mesh_data), tolerance);
        return flat_data_from_mesh(std::move(mesh));
    }

    void delete_flat_mesh_data(const flat_mesh_data * data)
    {
        delete[] data->vertices;
        delete[] data->elements;
        delete data;
    }
}


