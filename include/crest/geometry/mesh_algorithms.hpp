#pragma once

#include <vector>
#include <unordered_map>
#include <set>
#include <queue>
#include <algorithm>

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

        // Define the distance of an already visited element to be the maximum of the distances of its vertices
        const auto element_dist = [&vertex_distances, &mesh] (auto e)
        {
            const auto vertices = mesh.elements()[e].vertex_indices;
            const auto distances = {
                    vertex_distances[vertices[0]],
                    vertex_distances[vertices[1]],
                    vertex_distances[vertices[2]]
            };
            return *std::max_element(distances.begin(), distances.end());
        };

        // Below we implement a breadth-first search to find all elements in the patch
        const auto root_vertices = mesh.elements()[element].vertex_indices;
        for (const auto v : root_vertices) vertex_distances[v] = 0;

        std::queue<Index> queue;
        queue.push(element);
        patch.insert(element);

        while (!queue.empty())
        {
            const auto current = queue.front();
            queue.pop();
            //)const auto current_dist = element_dist(current);

            for (const auto neighbor : mesh.neighbors_for(current))
            {
                if (neighbor != mesh.sentinel())
                {
                    const auto vertices = mesh.elements()[neighbor].vertex_indices;
                    unsigned int min_vertex_dist = std::numeric_limits<unsigned int>::max();

                    for (const auto v : vertices)
                    {
                        const auto it = vertex_distances.find(v);
                        if (it != vertex_distances.end()) min_vertex_dist = std::min(min_vertex_dist, it->second);
                    }

                    for (const auto v : vertices)
                    {
                        if (vertex_distances.count(v) == 0) vertex_distances[v] = min_vertex_dist + 1;
                    }

                    if (patch.count(neighbor) == 0 && element_dist(neighbor) <= max_distance)
                    {
                        queue.push(neighbor);
                        patch.insert(neighbor);
                    }
                }
            }
        }

        return std::vector<Index>(patch.begin(), patch.end());
    };

}
