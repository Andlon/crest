#pragma once

#include <crest/geometry/vertex.hpp>
#include <crest/geometry/triangle.hpp>
#include <crest/util/algorithms.hpp>

#include <array>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <ostream>
#include <limits>
#include <cassert>

namespace crest {
    typedef uint32_t default_index_type;
    typedef double default_scalar_type;

    template <typename Index=default_index_type>
    struct Element
    {
        std::array<Index, 3> vertex_indices;

        explicit Element(std::array<Index, 3> indices) : vertex_indices(std::move(indices)) {}
    };

    template <typename Index=default_index_type>
    struct Neighbors
    {
        std::array<Index, 3> indices;

        explicit Neighbors(std::array<Index, 3> indices) : indices(std::move(indices)) {}
    };

    /**
     * IndexedMesh represents a 2D triangulation by an indexed set of vertices and a set of elements which holds
     * indices into the indexed set of vertices.
     */
    template <typename Scalar=default_scalar_type, typename Index=default_index_type>
    class IndexedMesh
    {
    public:
        typedef crest::Vertex<Scalar> Vertex;
        typedef crest::Element<Index> Element;
        typedef crest::Neighbors<Index> Neighbors;
        typedef Index SentinelType;

        IndexedMesh() {}
        explicit IndexedMesh(std::vector<Vertex> vertices,
                             std::vector<Element> indices);

        const std::vector<Vertex> &     vertices() const { return _vertices; }
        const std::vector<Element> &    elements() const { return _elements; }
        const std::vector<Neighbors> &  neighbors() const { return _neighbors; }

        std::array<Index, 3>    neighbors_for(Index element_index) const;
        Triangle<Scalar>        triangle_for(Index element_index) const;
        Index                   ancestor_for(Index element_index) const;

        Index num_vertices() const { return static_cast<Index>(_vertices.size()); }
        Index num_elements() const { return static_cast<Index>(_elements.size()); }

        Index num_boundary_vertices() const { return static_cast<Index>(_boundary.size()); }
        Index num_interior_vertices() const { return static_cast<Index>(num_vertices() - num_boundary_vertices()); };

        const std::vector<Index> & boundary_vertices() const { return _boundary; }

        // TODO: Consider rewriting interior indices as lazy iterators to avoid the potentially
        // huge allocation of a vector of mostly consecutive integers.
        std::vector<Index> compute_interior_vertices() const;

        /**
         * Use newest-vertex-bisection (NVB) to bisect all triangles indicated by their corresponding indices
         * in the supplied vector of indices.
         * @param marked
         */
        void bisect_marked(std::vector<Index> marked);

        /**
         * Attempts to compress the data structure so that it consumes less memory. Only use this when you are sure
         * that you will not further refine the mesh, as it may negatively impact the performance of subsequent
         * refinement.
         */
        void compress();

        /**
         * Effectively makes every triangle its own ancestor.
         *
         * If one represents the mesh as a forest of trees, this would take every leaf node and build a new
         * forest in which each lofe node becomes a standalone tree.
         */
        void reset_ancestry();

        /**
         * Returns the sentinel value used when a sentinel value is required. This is a special value at the extreme
         * of the range of valid values in the type, and is used to indicate special properties, such as when
         * a triangle has no neighbor (the index of the neighbor is then given by a sentinel value).
         */
        constexpr static SentinelType sentinel() { return std::numeric_limits<Index>::max(); }

    private:
        std::vector<Vertex> _vertices;
        std::vector<Element> _elements;
        std::vector<Neighbors> _neighbors;
        std::vector<Index> _boundary;
        std::vector<Index> _ancestors;
    };

    class conformance_error : public std::logic_error
    {
    public:
        explicit conformance_error(const std::string & what) : std::logic_error(what) {}
    };

    /*
     * IMPLEMENTATION
     */

    namespace detail {
        template <typename I>
        I find_neighbor_from_edge_vertices(I element_index,
                                           const std::vector<I> & a_vertices,
                                           const std::vector<I> & b_vertices)
        {
            // This algorithm is obviously N^2, but we except each vector to hold a very low amount of items,
            // and since then both vectors most likely fit in cache, it might in fact outperform more efficient
            // routines involving sorts (would require profiling to verify).

            constexpr I NO_NEIGHBOR = std::numeric_limits<I>::max();
            I neighbor_index = NO_NEIGHBOR;
            size_t num_matches = 0;

            for (auto i : a_vertices)
            {
                for (auto j : b_vertices)
                {
                    if (i == j && i != element_index)
                    {
                        neighbor_index = i;
                        ++num_matches;
                    }
                }
            }

            if (num_matches > 1) throw conformance_error("An edge shared by more than two triangles is not conforming.");

            return neighbor_index;
        }

        template <typename I>
        std::vector<Neighbors<I>> find_neighbors(I num_vertices, const std::vector<Element<I>> & elements)
        {
            std::vector<Neighbors<I>> element_neighbors;
            element_neighbors.resize(elements.size(), Neighbors<I>({0, 0, 0}));
            std::unordered_map<I, std::vector<I>> vertex_to_triangles;

            // Map each vertex to the set of triangles that contain it
            for (size_t element_index = 0; element_index < elements.size(); ++element_index)
            {
                const auto & element = elements[element_index];
                for (auto vertex_index : element.vertex_indices)
                {
                    if (vertex_index < 0 || vertex_index >= num_vertices)
                    {
                        throw std::out_of_range("Invalid vertex index in element.");
                    }
                    vertex_to_triangles[vertex_index].push_back(element_index);
                }
            }

            // For a conforming triangulation, each edge defined by two connected vertices
            // can only be shared by at most two triangles.
            for (size_t element_index = 0; element_index < elements.size(); ++element_index)
            {
                const auto & element = elements[element_index];
                auto & neighbors = element_neighbors[element_index];

                // for an element defined by the vertex indices (z0, z1, z2),
                // we have the convention that the neighbors are ordered in the order of the edges
                // (z0, z1), (z1, z2), (z2, z0),
                // so that the first neighbor shares edge (z0, z1) and so on.
                auto z0_triangles = vertex_to_triangles[element.vertex_indices[0]];
                auto z1_triangles = vertex_to_triangles[element.vertex_indices[1]];
                auto z2_triangles = vertex_to_triangles[element.vertex_indices[2]];

                neighbors.indices[0] = find_neighbor_from_edge_vertices<I>(element_index, z0_triangles, z1_triangles);
                neighbors.indices[1] = find_neighbor_from_edge_vertices<I>(element_index, z1_triangles, z2_triangles);
                neighbors.indices[2] = find_neighbor_from_edge_vertices<I>(element_index, z2_triangles, z0_triangles);
            }

            return element_neighbors;
        };

        template <typename T, typename I>
        std::vector<I> determine_new_boundary_vertices(const IndexedMesh<T, I> & mesh)
        {
            const auto NO_NEIGHBOR = mesh.sentinel();
            std::vector<I> boundary;

            auto push_boundary_edge = [&boundary] (auto from, auto to)
            {
                boundary.push_back(from);
                boundary.push_back(to);
            };

            for (I element_index = 0; element_index < mesh.num_elements(); ++element_index)
            {
                const auto vertex_indices = mesh.elements()[element_index].vertex_indices;
                const auto neighbors = mesh.neighbors_for(element_index);

                const auto z0 = vertex_indices[0];
                const auto z1 = vertex_indices[1];
                const auto z2 = vertex_indices[2];

                // Recall that edges are ordered (z0, z1), (z1, z2), (z2, z0).
                if (neighbors[0] == NO_NEIGHBOR) push_boundary_edge(z0, z1);
                if (neighbors[1] == NO_NEIGHBOR) push_boundary_edge(z1, z2);
                if (neighbors[2] == NO_NEIGHBOR) push_boundary_edge(z2, z0);
            }

            std::sort(boundary.begin(), boundary.end());
            boundary.erase(std::unique(boundary.begin(), boundary.end()), boundary.end());
            return boundary;
        };
    }

    template <typename I>
    inline bool operator==(const Element<I> & left, const Element<I> & right) {
        return std::equal(left.vertex_indices.cbegin(), left.vertex_indices.cend(), right.vertex_indices.cbegin());
    }

    template <typename I>
    inline std::ostream & operator<<(std::ostream & o, const Element<I> & element)
    {
        o << "[ "
          << element.vertex_indices[0] << ", "
          << element.vertex_indices[1] << ", "
          << element.vertex_indices[2]
          << " ]";
        return o;
    }

    template <typename T, typename I>
    inline IndexedMesh<T, I>::IndexedMesh(std::vector<IndexedMesh<T, I>::Vertex> vertices,
                                          std::vector<IndexedMesh<T, I>::Element> elements)
            :   _vertices(std::move(vertices)), _elements(std::move(elements)),
                _neighbors(detail::find_neighbors(static_cast<I>(_vertices.size()), _elements))
    {
        _boundary = detail::determine_new_boundary_vertices(*this);
        _ancestors.reserve(num_elements());
        for (I i = 0; i < num_elements(); ++i)
        {
            // Make every triangle an ancestor of itself
            _ancestors.push_back(i);
        }
    }

    template <typename T, typename I>
    inline std::array<I, 3> IndexedMesh<T, I>::neighbors_for(I element_index) const
    {
        assert(element_index >= 0 && element_index < num_elements());
        return _neighbors[element_index].indices;
    };

    template <typename T, typename I>
    inline Triangle<T> IndexedMesh<T, I>::triangle_for(I element_index) const
    {
        assert(element_index >= 0 && element_index < num_elements());
        const auto indices = _elements[static_cast<size_t>(element_index)].vertex_indices;
        const auto a = _vertices[indices[0]];
        const auto b = _vertices[indices[1]];
        const auto c = _vertices[indices[2]];
        return Triangle<T>(a, b, c);
    }

    template <typename T, typename I>
    inline I IndexedMesh<T, I>::ancestor_for(I element_index) const
    {
        assert(element_index >= 0 && element_index < num_elements());
        return _ancestors[element_index];
    }

    template <typename T, typename I>
    inline void IndexedMesh<T, I>::compress()
    {
        _neighbors.shrink_to_fit();
        _vertices.shrink_to_fit();
        _elements.shrink_to_fit();
        _boundary.shrink_to_fit();
    };

    template <typename T, typename I>
    inline std::vector<I> IndexedMesh<T, I>::compute_interior_vertices() const {
        std::vector<I> interior;
        interior.reserve(num_interior_vertices());

        const auto & boundary = boundary_vertices();
        size_t boundary_index = 0;
        for (I i = 0; i < num_vertices(); ++i)
        {
            if (boundary_index < boundary.size() && i == boundary[boundary_index])
            {
                ++boundary_index;
            } else
            {
                interior.push_back(i);
            }
        }

        return interior;
    };

    template <typename T, typename I>
    inline void IndexedMesh<T, I>::reset_ancestry()
    {
        for (I t = 0; t < num_elements(); ++t)
        {
            _ancestors[t] = t;
        }
    }

    template <typename T, typename I>
    inline void IndexedMesh<T, I>::bisect_marked(std::vector<I> marked)
    {
        // The implementation here relies heavily on certain important conventions.
        // First, for any element K defined by vertex indices (z0, z1, z2),
        // we define the *refinement edge* to be the edge between z2 and z0, denoted (z2, z0).

        // Furthermore, we introduce the convention that a triangle K has a hanging node on edge E
        // if another triangle T is a neighbor of K on E, but K is not a neighbor of T on E.

        // In other words, we may represent the neighbor relationships as a directed graph. In a conforming mesh,
        // any triangle K and a neighbor T will have bidirectional edges, whereas if the mesh has hanging nodes,
        // this is not the case. We use this property to determine if there are hanging nodes.
        // Hence, we are done once all neighbor edges are bidirectional after refinement.

        // As for the bisection, for a triangle K defined by the vertices (z0, z1, z2), and the midpoint z_mid
        // on the refinement edge (z2, z0), we define the 'left' and 'right' triangles as the two triangles defined
        // by the vertices (z1, midpoint_index, z0) and (z2, midpoint_index, z1) respectively. The names correspond
        // to the fact that if the vertices are defined in a counter-clockwise order, the 'left' triangle will
        // correspond to the left triangle when facing the refinement edge, and similarly for the right one.
        // Note also in particular that this choice of indices preserves the *winding order* of the original triangle.
        // This is important, because for some applications (such as computer graphics), the winding order defines
        // the orientation of the face associated with the triangle.

        constexpr I NO_NEIGHBOR = sentinel();
        std::vector<I> nonconforming;

        if (std::any_of(marked.cbegin(), marked.cend(), [this] (auto i) { return i >= this->num_elements(); }))
        {
            throw std::invalid_argument("Set of marked elements contains element indices out of bounds.");
        }

        do {
            // The algorithm may mark the same element multiple times in the course of a single iteration
            // so we need to remove duplicates first. We also sort it so we can use binary search later.
            // Using a hashset would provide better average complexity, but it would be less memory-efficient,
            // and since triangulations usually don't grow too big (more than a few million elements, perhaps),
            // std::sort is still extremely fast for integers. Of course, proper profiling would need to take place
            // in order to verify this.
            std::sort(marked.begin(), marked.end());
            marked.erase(std::unique(marked.begin(), marked.end()), marked.end());

            for (auto element_index : marked)
            {
                const auto & element = elements()[element_index];
                auto neighbors = neighbors_for(element_index);

                const auto edge_is_on_boundary = [this, neighbors] (I local_edge_index) {
                    return neighbors[local_edge_index] == NO_NEIGHBOR;
                };

                const auto edge_has_hanging_node = [this, neighbors, element_index] (I local_edge_index) {
                    auto neighbor = neighbors[local_edge_index];
                    return neighbor == NO_NEIGHBOR
                           ? false
                           : !algo::contains(neighbors_for(neighbor), element_index);
                };

                auto update_neighbor_of = [this, neighbors, element_index, edge_is_on_boundary, &nonconforming]
                        (I local_edge_index, I new_index) {
                    // Update the neighbor on the edge indicated by local_edge_index with the new index.
                    // This is of course only necessary if there exists a neighbor at all
                    // (i.e. we're not on the boundary)
                    if (!edge_is_on_boundary(local_edge_index))
                    {
                        const auto neighbor = neighbors[local_edge_index];
                        auto & neighbors_of_neighbor = _neighbors[neighbor].indices;
                        auto pos_of_element_in_neighbor =
                                std::find(neighbors_of_neighbor.begin(), neighbors_of_neighbor.end(), element_index);

                        if (pos_of_element_in_neighbor == neighbors_of_neighbor.end())
                        {
                            // Edge is not a boundary edge, yet there is no graph edge from the neighbor to the
                            // current element, so by the convention explained earlier, there must be a hanging node
                            // on the edge.
                            nonconforming.push_back(new_index);
                        } else
                        {
                            // No hanging node, so we need to update the neighbor's data
                            *pos_of_element_in_neighbor = new_index;
                        }
                    }

                };

                const auto left_index = element_index;
                const auto right_index = static_cast<I>(_elements.size());

                const auto z0 = element.vertex_indices[0];
                const auto z1 = element.vertex_indices[1];
                const auto z2 = element.vertex_indices[2];

                const auto v0 = vertices()[z0];
                const auto v2 = vertices()[z2];

                // Recall that we have the edges (z0, z1), (z1, z2), (z2, z0),
                // and the indexing of neighbors is defined in the same order.
                auto left_neighbors = Neighbors({right_index, NO_NEIGHBOR, neighbors[0] });
                auto right_neighbors = Neighbors({NO_NEIGHBOR, left_index, neighbors[1] });
                const auto refinement_edge_neighbor = neighbors[2];

                I midpoint_index = sentinel();
                if (!edge_has_hanging_node(2) || edge_is_on_boundary(2))
                {
                    // There is currently no hanging node on the refinement edge
                    midpoint_index = static_cast<I>(num_vertices());
                    _vertices.emplace_back(midpoint(v0, v2));

                    if (edge_is_on_boundary(2))
                    {
                        // Recall that _boundary must always be sorted. Since we already add new midpoint vertices
                        // to the end of the list of vertices, this invariant holds automatically when
                        // we add the midpoint to the list of boundary indices.
                        _boundary.push_back(midpoint_index);
                    } else if (!std::binary_search(marked.cbegin(), marked.cend(), refinement_edge_neighbor))
                    {
                        // If n20 exists and has not already been marked, add it as a nonconforming triangle,
                        // since we've added a midpoint on the shared edge. We must make sure to only add it as
                        // a nonconforming element if it isn't already marked, otherwise we might end up
                        // bisecting the left bisection of n20 twice.
                        nonconforming.push_back(refinement_edge_neighbor);
                    }
                } else
                {
                    const auto & n20 = refinement_edge_neighbor;

                    // There is already a hanging node on the refinement edge.
                    // This means that n20 is the 'left' triangle of the bisected original triangle,
                    // and as such it has vertices (z0', z_mid, z1'), so the middle vertex
                    // corresponds to the vertex.
                    midpoint_index = elements()[n20].vertex_indices[1];

                    const auto n20_left = n20;
                    const auto n20_right = neighbors_for(n20)[0];

                    // Connect left with n20_right and right with n20_left, and vice versa
                    _neighbors[n20_left].indices[1] = right_index;
                    _neighbors[n20_right].indices[0] = left_index;
                    left_neighbors.indices[1] = n20_right;
                    right_neighbors.indices[0] = n20_left;
                }

                const auto left = Element({z1, midpoint_index, z0});
                const auto right = Element({z2, midpoint_index, z1});

                _elements[left_index] = left;
                _elements.push_back(right);
                _neighbors[left_index] = left_neighbors;
                _neighbors.push_back(right_neighbors);

                if (_ancestors[left_index] == sentinel())
                {
                    // If the element being refined has no ancestor, we want to make this element the ancestor
                    // of the two new elements.
                    _ancestors[left_index] = element_index;
                    _ancestors.push_back(element_index);
                } else
                {
                    // If the element being refined has an ancestor, we want to keep this ancestor in the
                    // two new elements (and hence we do not need to change the ancestor of left_index).
                    _ancestors.push_back(_ancestors[left_index]);
                }

                update_neighbor_of(0, left_index);
                update_neighbor_of(1, right_index);

                assert(midpoint_index != sentinel());
            }

            // Note that this pattern minimizes reallocation
            marked.swap(nonconforming);
            nonconforming.clear();
        } while (!marked.empty());
    };

}
