#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/geometry/mesh_algorithms.hpp>

#include <vector>

using crest::IndexedMesh;
using crest::patch_for_element;

using ::testing::ElementsAreArray;
using ::testing::ElementsAre;
using ::testing::Pointwise;
using ::testing::Eq;
using ::testing::IsEmpty;

typedef IndexedMesh<>::Vertex Vertex;
typedef IndexedMesh<>::Element Element;

class parameterized_patch_for_element_test : public ::testing::TestWithParam<unsigned int>
{

};

TEST_P(parameterized_patch_for_element_test, patch_for_single_triangle_contains_only_triangle_itself)
{
    const auto max_distance = GetParam();

    const auto vertices = std::vector<Vertex> {
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(0.0, 1.0)
    };

    const auto elements = std::vector<Element> {
            Element({0, 1, 2})
    };

    const auto mesh = IndexedMesh<>(vertices, elements);
    const auto patch = crest::patch_for_element(mesh, 0u, max_distance);

    EXPECT_THAT(patch, ElementsAre(0));
}

INSTANTIATE_TEST_CASE_P(small_maximum_distances,
                        parameterized_patch_for_element_test,
                        ::testing::Values(0, 1, 2, 3));


TEST(patch_for_element, patch_for_triangle_in_two_triangle_mesh)
{
    const auto vertices = std::vector<Vertex> {
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(1.0, 1.0),
            Vertex(0.0, 1.0)
    };

    const auto elements = std::vector<Element> {
            Element({0, 1, 3}),
            Element({1, 2, 3})
    };

    const auto mesh = IndexedMesh<>(vertices, elements);

    EXPECT_THAT(patch_for_element(mesh, 0u, 0u), ElementsAre(0));
    EXPECT_THAT(patch_for_element(mesh, 0u, 1u), ElementsAre(0, 1));
    EXPECT_THAT(patch_for_element(mesh, 0u, 2u), ElementsAre(0, 1));
    EXPECT_THAT(patch_for_element(mesh, 0u, 3u), ElementsAre(0, 1));

    EXPECT_THAT(patch_for_element(mesh, 1u, 0u), ElementsAre(1));
    EXPECT_THAT(patch_for_element(mesh, 1u, 1u), ElementsAre(0, 1));
    EXPECT_THAT(patch_for_element(mesh, 1u, 2u), ElementsAre(0, 1));
    EXPECT_THAT(patch_for_element(mesh, 1u, 3u), ElementsAre(0, 1));
}

// TODO: Write more tests