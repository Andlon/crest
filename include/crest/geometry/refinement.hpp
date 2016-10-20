#pragma once

#include <crest/geometry/indexed_mesh.hpp>

#include <cassert>
#include <algorithm>
#include <iterator>

namespace crest
{
    namespace detail {
        template <typename Scalar, typename Index>
        std::vector<Index> mark_refinement_candidates(const IndexedMesh<Scalar, Index> & mesh,
                                                        std::vector<Index> search_space,
                                                        Scalar tolerance)
        {
            // Instead of adding new marked elements, we remove
            // the ones which do not need further refinement from the search space and return the search space.
            // This way we can exploit std::remove_if and avoid any additional allocation.
            auto tolerance2 = tolerance * tolerance;
            auto logical_end = std::remove_if(search_space.begin(), search_space.end(), [&] (auto index) {
                const auto diameter2 = diameter_squared(mesh.triangle_for(index));
                return diameter2 < tolerance2;
            });
            search_space.erase(logical_end, search_space.end());
            return search_space;
        };
    }



    template <typename Scalar, typename Index>
    inline IndexedMesh<Scalar, Index> bisect_to_tolerance(IndexedMesh<Scalar, Index> mesh, Scalar tolerance)
    {
        // Mark all elements whose diameters are too big, refine marked elements. Repeat until all elements
        // have sufficiently small diameters.
        assert(tolerance > 0);
        std::vector<Index> marked(mesh.num_elements());

        // Mark all elements for inspection
        algo::fill_strided_integers<Index>(marked.begin(), marked.end());

        do {
            // Note that due to the nature of how refine_marked works, we know that all elements that are:
            // - not marked prior to calling refine_marked
            // - not added as a consequence of refine_marked
            // will remain unchanged. We can exploit this property to avoid having to re-scan every
            // element after each bisection.
            Index elements_before_refine = mesh.num_elements();
            marked = detail::mark_refinement_candidates(mesh, std::move(marked), tolerance);
            mesh.bisect_marked(marked);
            const auto num_refinements = mesh.num_elements() - elements_before_refine;

            // Keep the existing marked element indices (which now correspond to newly bisected elements),
            // and add the newly added element indices, resulting from the bisection.
            algo::fill_strided_integers_n(std::back_inserter(marked), num_refinements, elements_before_refine);
        } while(!marked.empty());
        return mesh;
    };

}