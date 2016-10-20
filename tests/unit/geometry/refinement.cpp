#include <crest/geometry/refinement.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <util/vertex_matchers.hpp>
#include <cmath>
#include <algorithm>

using Vertex = crest::Vertex<double>;
using Element = crest::Element<unsigned int>;

using ::testing::Pointwise;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::AnyOf;
using ::testing::DoubleEq;
using ::testing::Each;

template <typename T, typename I>
std::vector<T> compute_diameters(const crest::IndexedMesh<T, I> & mesh)
{
    std::vector<T> diam;
    diam.reserve(mesh.num_elements());
    for (size_t i = 0; i < mesh.num_elements(); ++i)
    {
        const auto triangle = mesh.triangle_for(static_cast<I>(i));
        const auto diameter = crest::diameter(triangle);
        diam.push_back(diameter);
    }
    return diam;
};

template <typename Integer>
bool is_even(Integer i) { return i % 2 == 0; }

class refinement_test : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        {
            std::vector<Vertex> vertices_unit_square = {
                    Vertex(0.0, 0.0),
                    Vertex(1.0, 0.0),
                    Vertex(1.0, 1.0),
                    Vertex(0.0, 1.0)
            };

            std::vector<Element> elements_unit_square = {
                    Element({3, 0, 1}),
                    Element({1, 2, 3})
            };
            unit_square_2_elements = crest::IndexedMesh<>(vertices_unit_square, elements_unit_square);
        }
    }

    crest::IndexedMesh<double, unsigned int> unit_square_2_elements;
};

TEST_F(refinement_test, sufficiently_fine_mesh_is_unchanged)
{
    const auto mesh = crest::bisect_to_tolerance(unit_square_2_elements, 2.0);

    EXPECT_THAT(mesh.elements(), Pointwise(Eq(), unit_square_2_elements.elements()));
    EXPECT_THAT(mesh.vertices(), Pointwise(VertexDoubleEq(), unit_square_2_elements.vertices()));
    EXPECT_THAT(mesh.neighbors_for(0), Pointwise(Eq(), unit_square_2_elements.neighbors_for(0)));
    EXPECT_THAT(mesh.neighbors_for(1), Pointwise(Eq(), unit_square_2_elements.neighbors_for(1)));
}

TEST_F(refinement_test, expected_single_bisection)
{
    const auto mesh = crest::bisect_to_tolerance(unit_square_2_elements, 1.1);

    std::vector<Vertex> expected_vertices {
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(1.0, 1.0),
            Vertex(0.0, 1.0),
            Vertex(0.5, 0.5)
    };

    std::vector<Element> expected_elements {
            Element({0, 4, 3}),
            Element({2, 4, 1}),
            Element({1, 4, 0}),
            Element({3, 4, 2})
    };

    EXPECT_THAT(mesh.vertices(), Pointwise(VertexDoubleEq(), expected_vertices));
    EXPECT_THAT(mesh.elements(), Pointwise(Eq(), expected_elements));
}

class refinement_test_with_tolerance : public refinement_test, public ::testing::WithParamInterface<double>
{

};

TEST_P(refinement_test_with_tolerance, all_elements_have_expected_diameter)
{
    const double tolerance =  GetParam();

    // The `unit_square_2_elements` mesh has the following useful property:
    // The number of bisections n necessary for all edges in the mesh to have diameters smaller than tolerance
    // is bounded by:
    //      n >= 1 - 2 log2(tolerance).
    // Furthermore, the length l of each edge is bounded by
    //
    // sqrt(2) / 2^((n + 1)/2) <= l <= 1 / 2^((n - 1)/2)    if n is odd
    //
    //           1 / 2^(n / 2) <= l <= sqrt(2) / 2^(n / 2)  if n is even
    //
    // Moreover, since any triangle in the mesh family consists of exactly one "diagonal" edge and two "straight"
    // edges, the diameter of any element is given by the upper bound.

    const unsigned int n = static_cast<unsigned int>(std::max(0.0, ceil(1 - 2.0 * std::log2(tolerance))));
    const double upper_bound = is_even(n)
                               ? sqrt(2.0) / pow(2.0, n / 2.0)
                               : 1.0 / pow(2.0, (n - 1.0) / 2.0);
    const double expected_diameter = upper_bound;

    const auto mesh = crest::bisect_to_tolerance(unit_square_2_elements, tolerance);
    const auto diameters = compute_diameters(mesh);

    EXPECT_THAT(diameters.size(), mesh.num_elements());
    EXPECT_THAT(diameters, Each(DoubleEq(expected_diameter)));
}

INSTANTIATE_TEST_CASE_P(various_tolerances,
                        refinement_test_with_tolerance,
                        ::testing::Values(1.4, 0.7, 0.35, 0.175));
