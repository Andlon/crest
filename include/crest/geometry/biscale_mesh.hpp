#pragma once

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/geometry/patch.hpp>
#include <crest/geometry/mesh_algorithms.hpp>

#include <boost/range/iterator_range.hpp>

#include <cassert>

namespace crest
{
    namespace detail
    {
        template <typename Index>
        class DescendantMap
        {
        public:
            typedef typename std::vector<Index>::const_iterator          ConstIndexIterator;
            typedef typename boost::iterator_range<ConstIndexIterator>   ConstIndexIteratorRange;

            template <typename Scalar>
            explicit DescendantMap(const IndexedMesh<Scalar, Index> & coarse,
                                   const IndexedMesh<Scalar, Index> & fine)
            {
                _map = std::vector<std::vector<Index>>(coarse.num_elements());
                for (Index k = 0; k < fine.num_elements(); ++k)
                {
                    assert(fine.ancestor_for(k) < coarse.num_elements());
                    _map[fine.ancestor_for(k)].push_back(k);
                }
            }

            ConstIndexIteratorRange descendants_for(Index coarse_element) const
            {
                assert(coarse_element >= Index(0) && coarse_element <= static_cast<Index>(_map.size()));
                return boost::make_iterator_range(_map[coarse_element].begin(), _map[coarse_element].end());
            }

        private:
            std::vector<std::vector<Index>> _map;
        };
    }

    /**
     * Represents a triangulated domain by a "coarse" (usually quasi-uniform) mesh and
     * a "fine" mesh that is formed from the coarse mesh by bisection.
     */
    template <typename Scalar, typename Index>
    class BiscaleMesh
    {
    public:
        struct CoarseTag {};
        struct FineTag {};

        typedef Patch<Scalar, Index, CoarseTag> CoarsePatch;
        typedef Patch<Scalar, Index, FineTag>   FinePatch;

        typedef typename detail::DescendantMap<Index>::ConstIndexIteratorRange ConstIndexIteratorRange;

        explicit BiscaleMesh(IndexedMesh<Scalar, Index> coarse_mesh,
                             IndexedMesh<Scalar, Index> fine_mesh);

        CoarsePatch coarse_element_patch(Index coarse_element, unsigned int max_distance) const;
        FinePatch fine_patch_from_coarse(const CoarsePatch & coarse_patch) const;

        Index                       ancestor_for(Index fine_element) const;
        ConstIndexIteratorRange     descendants_for(Index coarse_element) const;

        const IndexedMesh<Scalar, Index> & coarse_mesh() const;
        const IndexedMesh<Scalar, Index> & fine_mesh() const;

    private:
        IndexedMesh<Scalar, Index>      _coarse;
        IndexedMesh<Scalar, Index>      _fine;
        detail::DescendantMap<Index>    _descendants;
    };

    template <typename Scalar, typename Index>
    BiscaleMesh<Scalar, Index>::BiscaleMesh(IndexedMesh<Scalar, Index> coarse_mesh,
                                            IndexedMesh<Scalar, Index> fine_mesh)
            :   _descendants(detail::DescendantMap<Index>(coarse_mesh, fine_mesh))
    {
        _coarse = std::move(coarse_mesh);
        _fine = std::move(fine_mesh);
        // Check that all coarse elements have at least one descendant, otherwise
        // the fine mesh cannot correspond to a refinement of the coarse mesh.
        for (Index k = 0; k < _coarse.num_elements(); ++k)
        {
            // Since this is a no-op in release mode, the compiler will optimize it away
            assert(!_descendants.descendants_for(k).empty());
        }
    };

    template <typename Scalar, typename Index>
    typename BiscaleMesh<Scalar, Index>::CoarsePatch
    BiscaleMesh<Scalar, Index>::coarse_element_patch(Index coarse_element, unsigned int max_distance) const
    {
        return patch_for_element<CoarseTag>(coarse_mesh(), coarse_element, max_distance);
    };

    template <typename Scalar, typename Index>
    typename BiscaleMesh<Scalar, Index>::FinePatch
    BiscaleMesh<Scalar, Index>::fine_patch_from_coarse(
            const BiscaleMesh<Scalar, Index>::CoarsePatch & coarse_patch) const
    {
        std::vector<Index> fine_patch;
        for (const auto coarse_element : coarse_patch)
        {
            for (const auto fine_element : descendants_for(coarse_element))
            {
                fine_patch.push_back(fine_element);
            }
        }
        fine_patch = algo::sorted_unique(std::move(fine_patch));
        return FinePatch(fine_mesh(), std::move(fine_patch));
    };

    template <typename Scalar, typename Index>
    Index BiscaleMesh<Scalar, Index>::ancestor_for(Index fine_element) const
    {
        return _fine.ancestor_for(fine_element);
    };

    template <typename Scalar, typename Index>
    typename BiscaleMesh<Scalar, Index>::ConstIndexIteratorRange
    BiscaleMesh<Scalar, Index>::descendants_for(Index coarse_element) const
    {
        return _descendants.descendants_for(coarse_element);
    };

    template <typename Scalar, typename Index>
    const IndexedMesh<Scalar, Index> & BiscaleMesh<Scalar, Index>::coarse_mesh() const
    {
        return _coarse;
    };

    template <typename Scalar, typename Index>
    const IndexedMesh<Scalar, Index> & BiscaleMesh<Scalar, Index>::fine_mesh() const
    {
        return _fine;
    };


}
