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


TEST(patch_for_element, patch_for_triangle_in_2_triangle_mesh)
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

TEST(patch_for_element, patch_for_triangles_in_16_element_mesh)
{
    const std::vector<Vertex> vertices {
            Vertex(0.0, 0.0),
            Vertex(0.5, 0.0),
            Vertex(1.0, 0.0),
            Vertex(0.75, 0.25),
            Vertex(0.25, 0.25),
            Vertex(0.0, 0.5),
            Vertex(0.5, 0.5),
            Vertex(1.0, 0.5),
            Vertex(0.75, 0.75),
            Vertex(0.25, 0.75),
            Vertex(0.0, 1.0),
            Vertex(0.5, 1.0),
            Vertex(1.0, 1.0)
    };

    const std::vector<Element> elements {
            Element({0, 4, 5}),
            Element({0, 1, 4}),
            Element({1, 3, 4}),
            Element({1, 2, 3}),
            Element({2, 7, 3}),
            Element({7, 8, 3}),
            Element({3, 8, 6}),
            Element({6, 4, 3}),
            Element({4, 6, 9}),
            Element({4, 9, 5}),
            Element({5, 9, 10}),
            Element({10, 9, 11}),
            Element({9, 8, 11}),
            Element({8, 12, 11}),
            Element({7, 12, 8}),
            Element({6, 8, 9})
    };

    const auto mesh = IndexedMesh<>(vertices, elements);

    EXPECT_THAT(patch_for_element(mesh, 0u, 0u), ElementsAre(0));
    EXPECT_THAT(patch_for_element(mesh, 0u, 1u), ElementsAre(0, 1, 2, 7, 8, 9, 10));
    EXPECT_THAT(patch_for_element(mesh, 0u, 2u), ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7 ,8, 9, 10, 11, 12, 15}));
    EXPECT_THAT(patch_for_element(mesh, 0u, 3u), ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7 ,8, 9, 10, 11, 12, 13, 14, 15}));
    EXPECT_THAT(patch_for_element(mesh, 0u, 4u), ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7 ,8, 9, 10, 11, 12, 13, 14, 15}));

    EXPECT_THAT(patch_for_element(mesh, 6u, 0u), ElementsAreArray({6}));
    EXPECT_THAT(patch_for_element(mesh, 6u, 1u), ElementsAreArray({2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15}));
    EXPECT_THAT(patch_for_element(mesh, 6u, 2u), ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7 ,8, 9, 10, 11, 12, 13, 14, 15}));
}

TEST(patch_vertices_test, patch_vertices_for_triangles_in_16_element_mesh)
{
    const std::vector<Vertex> vertices {
            Vertex(0.0, 0.0),
            Vertex(0.5, 0.0),
            Vertex(1.0, 0.0),
            Vertex(0.75, 0.25),
            Vertex(0.25, 0.25),
            Vertex(0.0, 0.5),
            Vertex(0.5, 0.5),
            Vertex(1.0, 0.5),
            Vertex(0.75, 0.75),
            Vertex(0.25, 0.75),
            Vertex(0.0, 1.0),
            Vertex(0.5, 1.0),
            Vertex(1.0, 1.0)
    };

    const std::vector<Element> elements {
            Element({0, 4, 5}),
            Element({0, 1, 4}),
            Element({1, 3, 4}),
            Element({1, 2, 3}),
            Element({2, 7, 3}),
            Element({7, 8, 3}),
            Element({3, 8, 6}),
            Element({6, 4, 3}),
            Element({4, 6, 9}),
            Element({4, 9, 5}),
            Element({5, 9, 10}),
            Element({10, 9, 11}),
            Element({9, 8, 11}),
            Element({8, 12, 11}),
            Element({7, 12, 8}),
            Element({6, 8, 9})
    };

    const auto mesh = IndexedMesh<>(vertices, elements);

    const auto vertices_in_patch = crest::patch_vertices(mesh, { 3, 5, 8});

    EXPECT_THAT(vertices_in_patch, ElementsAre(1, 2, 3, 4, 6, 7, 8, 9));
}

TEST(patch_interior_test, patch_interior_for_triangles_in_16_element_mesh)
{
    const std::vector<Vertex> vertices {
            Vertex(0.0, 0.0),
            Vertex(0.5, 0.0),
            Vertex(1.0, 0.0),
            Vertex(0.75, 0.25),
            Vertex(0.25, 0.25),
            Vertex(0.0, 0.5),
            Vertex(0.5, 0.5),
            Vertex(1.0, 0.5),
            Vertex(0.75, 0.75),
            Vertex(0.25, 0.75),
            Vertex(0.0, 1.0),
            Vertex(0.5, 1.0),
            Vertex(1.0, 1.0)
    };

    const std::vector<Element> elements {
            Element({0, 4, 5}),
            Element({0, 1, 4}),
            Element({1, 3, 4}),
            Element({1, 2, 3}),
            Element({2, 7, 3}),
            Element({7, 8, 3}),
            Element({3, 8, 6}),
            Element({6, 4, 3}),
            Element({4, 6, 9}),
            Element({4, 9, 5}),
            Element({5, 9, 10}),
            Element({10, 9, 11}),
            Element({9, 8, 11}),
            Element({8, 12, 11}),
            Element({7, 12, 8}),
            Element({6, 8, 9})
    };

    const auto mesh = IndexedMesh<>(vertices, elements);

    {
        const auto interior = crest::patch_interior(mesh, { });
        EXPECT_THAT(interior, IsEmpty());
    }


    {
        const auto interior = crest::patch_interior(mesh, { 5, 6 });
        EXPECT_THAT(interior, IsEmpty());
    }

    {
        const auto interior = crest::patch_interior(mesh, { 2, 3, 4, 5, 6, 7, 15 });
        EXPECT_THAT(interior, ElementsAre(3));
    }

    {
        const auto interior = crest::patch_interior(mesh, { 0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15 });
        EXPECT_THAT(interior, ElementsAre(4, 6, 8));
    }

}
