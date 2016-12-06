#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/geometry/patch.hpp>


using crest::IndexedMesh;

using ::testing::ElementsAreArray;
using ::testing::ElementsAre;
using ::testing::Pointwise;
using ::testing::Eq;
using ::testing::IsEmpty;

typedef IndexedMesh<>::Vertex Vertex;
typedef IndexedMesh<>::Element Element;

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
    const auto patch = crest::make_patch(mesh, { 3, 5, 8 });

    EXPECT_THAT(patch.vertices(), ElementsAre(1, 2, 3, 4, 6, 7, 8, 9));
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
        const auto interior = crest::make_patch(mesh, { }).interior();
        EXPECT_THAT(interior, IsEmpty());
    }


    {
        const auto interior = crest::make_patch(mesh, { 5, 6 }).interior();
        EXPECT_THAT(interior, IsEmpty());
    }

    {
        const auto interior = crest::make_patch(mesh, { 2, 3, 4, 5, 6, 7, 15 }).interior();
        EXPECT_THAT(interior, ElementsAre(3));
    }

    {
        const auto interior = crest::make_patch(mesh, { 0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15 }).interior();
        EXPECT_THAT(interior, ElementsAre(4, 6, 8));
    }

}
