#pragma once

#include <crest/geometry/indexed_mesh.hpp>

#include <vector>

namespace crest
{
    struct GenericPatchTag;

    /**
     * Represents a subset of elements in an IndexedMesh.
     *
     * The patch internally holds a reference to the mesh, and so it is the responsibility of the user to
     * make sure that the mesh outlives the patch.
     *
     * Note that any refinement to the mesh invalidates this patch, much the same way
     * some STL iterators are invalidated when the underlying data structure is modified.
     */
    template <typename Scalar, typename Index, typename Tag = GenericPatchTag>
    class Patch
    {
    public:
        typedef typename std::vector<Index>::const_iterator const_iterator;
        typedef Index value_type;
        typedef Index & reference;
        typedef const Scalar & const_reference;
        typedef typename std::vector<Index>::difference_type difference_type;
        typedef typename std::vector<Index>::size_type size_type;

        explicit Patch(const IndexedMesh<Scalar, Index> & mesh, std::vector<Index> element_indices)
            : _mesh(mesh), _element_indices(std::move(element_indices))
        {
            assert(std::is_sorted(_element_indices.cbegin(), _element_indices.cend()));
        }

        /**
         * Returns a const iterator pointing to the first element in the patch.
         * @return An iterator to the first element in the patch.
         */
        const_iterator begin() const
        {
            return _element_indices.begin();
        }

        /**
         * Returns a const iterator referring to the *past-the-end* element in the Patch.
         * @return
         */
        const_iterator end() const
        {
            return _element_indices.end();
        }

        /**
         * Returns a const iterator pointing to the first element in the patch.
         * @return An iterator to the first element in the patch.
         */
        const_iterator cbegin() const
        {
            return _element_indices.cbegin();
        }

        /**
         * Returns a const iterator referring to the *past-the-end* element in the Patch.
         * @return
         */
        const_iterator cend() const
        {
            return _element_indices.cend();
        }

        /**
         * Returns a vector of the indices of the vertices that are contained in the patch. This includes
         * boundary vertices.
         * @return
         */
        std::vector<Index> vertices() const;

        /**
         * Returns a vector of the indices of the vertices that are contained
         * in the patch and not on the patch boundary.
         * @return
         */
        std::vector<Index> interior() const;

        /**
         * Returns the number of elements in the patch.
         * @return
         */
        Index num_elements() const
        {
            return static_cast<Index>(size());
        }

        size_t size() const
        {
            return _element_indices.size();
        }

    private:
        const IndexedMesh<Scalar, Index> & _mesh;
        std::vector<Index> _element_indices;
    };

    template <typename Scalar, typename Index, typename Tag = GenericPatchTag>
    Patch<Scalar, Index, Tag> make_patch(const IndexedMesh<Scalar, Index> & mesh,
                                         std::vector<Index> element_indices)
    {
        return Patch<Scalar, Index, Tag>(mesh, std::move(element_indices));
    };

    template <typename Scalar, typename Index, typename Tag>
    Patch<Scalar, Index, Tag> patch_union(const Patch<Scalar, Index, Tag> & left,
                                          const Patch<Scalar, Index, Tag> & right)
    {
        // TODO: Implement this
    }

    template <typename Scalar, typename Index, typename Tag>
    Patch<Scalar, Index, Tag> patch_difference(const Patch<Scalar, Index, Tag> & left,
                                               const Patch<Scalar, Index, Tag> & right)
    {
        // TODO: Implement this
    }

    template <typename Scalar, typename Index, typename Tag>
    std::vector<Index> Patch<Scalar, Index, Tag>::vertices() const
    {
        std::vector<Index> vertices_in_patch;
        vertices_in_patch.reserve(3 * num_elements());
        for (const auto t : *this)
        {
            const auto vertices = _mesh.elements()[t].vertex_indices;
            std::copy(vertices.cbegin(), vertices.cend(), std::back_inserter(vertices_in_patch));
        }
        std::sort(vertices_in_patch.begin(), vertices_in_patch.end());
        vertices_in_patch.erase(std::unique(vertices_in_patch.begin(), vertices_in_patch.end()),
                                vertices_in_patch.end());
        return vertices_in_patch;
    };

    template <typename Scalar, typename Index, typename Tag>
    std::vector<Index> Patch<Scalar, Index, Tag>::interior() const
    {

        std::vector<Index> patch_vertices;
        std::vector<Index> boundary;
        patch_vertices.reserve(3 * num_elements());

        for (const auto t : *this)
        {
            const auto neighbors = _mesh.neighbors_for(t);
            const auto vertices = _mesh.elements()[t].vertex_indices;
            std::copy(vertices.begin(), vertices.end(), std::back_inserter(patch_vertices));

            const auto edge_is_on_patch_boundary = [&] (auto edge_index)
            {
                const auto neighbor = neighbors[edge_index];
                return !std::binary_search(_element_indices.cbegin(), _element_indices.cend(), neighbor);
            };

            for (size_t e = 0; e < 3; ++e)
            {
                if (edge_is_on_patch_boundary(e))
                {
                    // Denote the edge as (a, b). If the edge is on the boundary, then a and b are boundary vertices
                    boundary.push_back(vertices[e]);
                    boundary.push_back(vertices[(e + 1) % 3]);
                }
            }
        }

        patch_vertices = algo::sorted_unique(std::move(patch_vertices));
        boundary = algo::sorted_unique(std::move(boundary));
        std::vector<Index> interior;
        std::set_difference(patch_vertices.cbegin(), patch_vertices.cend(),
                            boundary.cbegin(), boundary.cend(),
                            std::back_inserter(interior));
        return interior;
    }


}
