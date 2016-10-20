#pragma once

#include <crest/geometry/indexed_mesh.hpp>

#include <cassert>
#include <algorithm>
#include <iterator>
#include <cmath>

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

        template <typename Scalar, typename Index>
        std::vector<Index> mark_elements_in_circle(const IndexedMesh<Scalar, Index> & mesh,
                                                   std::vector<Index> search_space,
                                                   const crest::Vertex<Scalar> & center,
                                                   Scalar radius)
        {

            const auto radius2 = radius * radius;
            auto logical_end = std::remove_if(search_space.begin(), search_space.end(), [&] (auto index) {
                const auto triangle = mesh.triangle_for(index);
                // The element is too big. Is it sufficiently far away to escape refinement?
                const auto distance2 = distance_squared(triangle, center);
                return distance2 > radius2;
            });
            search_space.erase(logical_end, search_space.end());
            return search_space;
        };

        template <typename Scalar, typename Index>
        crest::IndexedMesh<Scalar, Index> graded_refinement_for_single_corner(IndexedMesh<Scalar, Index> mesh,
                                                                              Scalar tolerance,
                                                                              Index corner_index,
                                                                              Scalar corner_radians)
        {
            constexpr Scalar PI = 3.1415926535897932385;
            assert(tolerance > 0);
            assert(corner_index >= 0 && corner_index < mesh.num_vertices());
            assert(corner_radians > 0 && corner_radians < 2.0 * PI);

            const auto corner_vertex = mesh.vertices()[corner_index];

            // TODO: Take p as an argument?
            const Scalar p = 1.0;

            // The lambda factor determines the grading of the refinement
            const Scalar lambda = PI / (2.0 * corner_radians);

            // The K factor determines the number of iterations, and hence the grade of the refinement,
            // since it proceeds in smaller and smaller constricted circles.
            const auto K = static_cast<unsigned int>(std::ceil( (p + 1.0) / lambda * log2(1.0 / tolerance)));

            // Initially include all elements in the search space
            std::vector<Index> search_space;
            search_space.reserve(mesh.num_elements());
            algo::fill_strided_integers_n(std::back_inserter(search_space), mesh.num_elements());

            for (size_t k = 0; k < 2 * K; ++k)
            {
                const auto p1 = p + 1.0;
                const auto m = static_cast<Scalar>(k);
                const auto max_diameter = tolerance * pow(2.0, - m * (p1 - lambda) / (2.0 * p1));
                const auto radius = pow(2.0, - m / 2.0);

                // Narrow the search space to only elements within a circle around the corner
                search_space = mark_elements_in_circle(mesh, std::move(search_space), corner_vertex, radius);

                // Mark all elements within the circle whose diameter is too big for refinement
                const auto marked = mark_refinement_candidates(mesh, search_space, max_diameter);

                const auto index_of_first_new_element = mesh.num_elements();
                mesh.bisect_marked(std::move(marked));
                const auto num_new_elements = mesh.num_elements() - index_of_first_new_element;

                // We need to add the newly added triangles to the search space
                algo::fill_strided_integers_n(std::back_inserter(search_space),
                                              num_new_elements,
                                              index_of_first_new_element);
            }

            return mesh;
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

    template <typename Scalar, typename Index>
    struct TwoScaleMeshes
    {
        IndexedMesh<Scalar, Index> coarse;
        IndexedMesh<Scalar, Index> fine;

        explicit TwoScaleMeshes(IndexedMesh<Scalar, Index> coarse, IndexedMesh<Scalar, Index> fine)
            :   coarse(std::move(coarse)), fine(std::move(fine)) {}
    };

    template <typename Scalar, typename Index>
    struct ReentrantCorner
    {
        Index  vertex_index;
        Scalar radians;

        explicit ReentrantCorner(Index vertex_index, Scalar radians)
            :   vertex_index(vertex_index), radians(radians) {}
    };

    template <typename Scalar, typename Index>
    inline TwoScaleMeshes<Scalar, Index> threshold(IndexedMesh<Scalar, Index> initial_mesh,
                                                   Scalar tolerance,
                                                   const std::vector<ReentrantCorner<Scalar, Index>> & corners)
    {
        const auto coarse = bisect_to_tolerance(std::move(initial_mesh), tolerance);
        auto fine = coarse;
        fine.reset_ancestry();

        for (const auto & corner : corners)
        {
            fine = detail::graded_refinement_for_single_corner(std::move(fine),
                                                               tolerance,
                                                               corner.vertex_index,
                                                               corner.radians);
        }

        return TwoScaleMeshes<Scalar, Index>(std::move(coarse), std::move(fine));
    };

}