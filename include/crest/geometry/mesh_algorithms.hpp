#pragma once

#include <vector>
#include <unordered_map>
#include <set>
#include <queue>
#include <algorithm>
#include <limits>

#include <crest/geometry/indexed_mesh.hpp>

namespace crest {

    template <typename Scalar, typename Index>
    std::vector<Index> patch_for_element(const IndexedMesh<Scalar, Index> & mesh,
                                         Index element,
                                         unsigned int max_distance)
    {
        assert(element >= 0 && element < mesh.num_elements());

        std::set<Index> patch;

        // We wish to do a BFS, but it's a bit tricky, because in the context of IndexedMesh,
        // the "neighbors" of a triangle correspond to triangles which share an edge with the given triangle,
        // but in this case we're rather interested in all triangles which share a vertex with the given triangle.
        // To overcome this hurdle, we define the "distance" of a vertex from the root element to be the
        // minimum number of edges between the vertex and any of the vertices contained in the root element.
        std::unordered_map<Index, unsigned int> vertex_distances;

        // Below we implement a modified breadth-first search to find all elements in the patch
        const auto root_vertices = mesh.elements()[element].vertex_indices;
        for (const auto v : root_vertices) vertex_distances[v] = 0;

        std::queue<Index> queue;
        queue.push(element);
        patch.insert(element);

        while (!queue.empty())
        {
            const auto current = queue.front();
            queue.pop();

            for (const auto neighbor : mesh.neighbors_for(current))
            {
                if (neighbor != mesh.sentinel())
                {
                    const auto vertices = mesh.elements()[neighbor].vertex_indices;
                    unsigned int min_vertex_dist = std::numeric_limits<unsigned int>::max();
                    unsigned int offset = 0;

                    for (const auto v : vertices)
                    {
                        const auto it = vertex_distances.find(v);
                        if (it != vertex_distances.end()) {
                            const auto v_dist = it->second;
                            if (v_dist <= min_vertex_dist) min_vertex_dist = v_dist;
                            else offset = 1;
                        }
                        else offset = 1;
                    }

                    // Define the distance of 'neighbor' to be the maximum of the distances of its vertices,
                    // which is either min_vertex_dist or (min_vertex_dist + 1)
                    const auto dist = min_vertex_dist + offset;
                    for (const auto v : vertices)
                    {
                        // Update the distance of the vertices which do not have a recorded distance already.
                        // This distance coincides with the distance of the element.
                        if (vertex_distances.count(v) == 0) vertex_distances[v] = dist;
                    }

                    if (patch.count(neighbor) == 0 && dist <= max_distance)
                    {
                        queue.push(neighbor);
                        patch.insert(neighbor);
                    }
                }
            }
        }

        return std::vector<Index>(patch.begin(), patch.end());
    };

    template <typename Scalar, typename Index>
    std::vector<Index> patch_vertices(const IndexedMesh<Scalar, Index> & mesh,
                                      const std::vector<Index> & patch)
    {
        assert(std::is_sorted(patch.cbegin(), patch.cend()));
        std::vector<Index> vertices_in_patch;
        vertices_in_patch.reserve(3 * patch.size());
        for (const auto t : patch)
        {
            const auto vertices = mesh.elements()[t].vertex_indices;
            std::copy(vertices.cbegin(), vertices.cend(), std::back_inserter(vertices_in_patch));
        }
        std::sort(vertices_in_patch.begin(), vertices_in_patch.end());
        vertices_in_patch.erase(std::unique(vertices_in_patch.begin(), vertices_in_patch.end()),
                                vertices_in_patch.end());
        return vertices_in_patch;
    };

    template <typename Scalar, typename Index>
    std::vector<Index> patch_interior(const IndexedMesh<Scalar, Index> & mesh,
                                      const std::vector<Index> & patch)
    {
        assert(std::is_sorted(patch.cbegin(), patch.cend()));

        std::vector<Index> interior;
        for (const auto t : patch)
        {
            const auto neighbors = mesh.neighbors_for(t);
            const auto vertices = mesh.elements()[t].vertex_indices;

            const auto edge_has_neighbor_in_patch = [&] (auto edge_index)
            {
                const auto neighbor = neighbors[edge_index];
                return std::binary_search(patch.cbegin(), patch.cend(), neighbor);
            };

            // Recall that for a triangle (z0, z1, z2), the neighbors are defined as the neighboring triangle
            // associated with edges (z0, z1), (z1, z2), (z2, z0).
            if (edge_has_neighbor_in_patch(2) && edge_has_neighbor_in_patch(0))
            {
                interior.push_back(vertices[0]);
            }

            if (edge_has_neighbor_in_patch(0) && edge_has_neighbor_in_patch(1))
            {
                interior.push_back(vertices[1]);
            }

            if (edge_has_neighbor_in_patch(1) && edge_has_neighbor_in_patch(2))
            {
                interior.push_back(vertices[2]);
            }
        }
        std::sort(interior.begin(), interior.end());
        interior.erase(std::unique(interior.begin(), interior.end()), interior.end());
        return interior;
    }

}
