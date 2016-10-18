#pragma once

#include <vector>
#include <unordered_map>
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

        std::vector<Index> patch;

        // We wish to do a BFS, but it's a bit tricky, because in the context of IndexedMesh,
        // the "neighbors" of a triangle correspond to triangles which share an edge with the given triangle,
        // but in this case we're rather interested in all triangles which share a vertex with the given triangle.
        // To overcome this hurdle, we define the "distance" of a vertex from the root element to be the
        // minimum number of edges between the vertex and any of the vertices contained in the root element.
        std::unordered_map<Index, unsigned int> vertex_distances;

        // Define the distance of an already visited element to be the minimum of the distances of its vertices
        const auto element_dist = [&vertex_distances, &mesh] (auto e)
        {
            const auto vertices = mesh.elements()[e].vertex_indices;
            const auto distances = {
                    vertex_distances[vertices[0]],
                    vertex_distances[vertices[1]],
                    vertex_distances[vertices[2]]
            };
            return *std::min_element(distances.begin(), distances.end());
        };

        // Below we implement a breadth-first search to find all elements in the patch
        const auto root_vertices = mesh.elements()[element].vertex_indices;
        for (const auto vertex : root_vertices)
        {
            vertex_distances[vertex] = 0;
        }

        std::queue<Index> queue;
        queue.push(element);
        patch.push_back(element);

        while (!queue.empty())
        {
            const auto current = queue.front();
            queue.pop();
            const auto current_dist = element_dist(current);

            for (const auto neighbor : mesh.neighbors_for(current))
            {
                if (neighbor != mesh.sentinel())
                {
                    const auto vertices = mesh.elements()[neighbor].vertex_indices;
                    bool visited = true;
                    for (const auto vertex_index : vertices)
                    {
                        // Assume that we've already visited this neighbor, unless it has a vertex
                        // for which a distance has not been recorded, in which case it means we have never
                        // before visited it.
                        auto it = vertex_distances.find(vertex_index);
                        if (it == vertex_distances.end())
                        {
                            visited = false;
                            vertex_distances[vertex_index] = current_dist + 1;
                        }
                    }

                    if (!visited && current_dist + 1 <= max_distance)
                    {
                        queue.push(neighbor);
                        patch.push_back(neighbor);
                    }
                }
            }
        }

        std::sort(patch.begin(), patch.end());
        return patch;
    };

}
