#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <rapidcheck/gtest.h>
#include <rapidcheck/state.h>

#include <crest/geometry/indexed_mesh.hpp>

#include <util/vertex_matchers.hpp>
#include <util/test_generators.hpp>

using crest::IndexedMesh;

using ::testing::ElementsAreArray;
using ::testing::ElementsAre;
using ::testing::Pointwise;
using ::testing::Eq;
using ::testing::IsEmpty;

typedef IndexedMesh<>::Vertex Vertex;
typedef IndexedMesh<>::Element Element;

class indexed_mesh_test : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        vertices_unit_square = {
                Vertex(0.0, 0.0),
                Vertex(0.0, 1.0),
                Vertex(1.0, 1.0),
                Vertex(1.0, 0.0)
        };

        elements_unit_square = {
                Element({0, 1, 2}),
                Element({0, 2, 3})
        };

        vertices_diamond = {
                Vertex(0.0, 0.0),
                Vertex(1.0, 0.0),
                Vertex(0.0, 1.0),
                Vertex(-1.0, 0.0),
                Vertex(0.0, -1.0)
        };

        elements_diamond = {
                Element({0, 1, 2}),
                Element({0, 2, 3}),
                Element({0, 3, 4}),
                Element({0, 4, 1})
        };

        vertices_triangle = {
                Vertex(0.0, 0.0),
                Vertex(1.0, 0.0),
                Vertex(0.0, 1.0)
        };

        elements_triangle = {
                Element({2, 0, 1})
        };

        vertices_unit_square_with_interior = {
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

        elements_unit_square_with_interior = {
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

        vertices_unit_square_1_interior_node = {
                Vertex(0.0, 0.0),
                Vertex(1.0, 0.0),
                Vertex(1.0, 1.0),
                Vertex(0.0, 1.0),
                Vertex(0.5, 0.5)

        };

        elements_unit_square_1_interior_node = {
                Element({0, 1, 4}),
                Element({1, 2, 4}),
                Element({2, 3, 4}),
                Element({3, 0, 4})
        };
    }

    std::vector<Vertex>     vertices_unit_square;
    std::vector<Element>    elements_unit_square;
    std::vector<Vertex>     vertices_diamond;
    std::vector<Element>    elements_diamond;
    std::vector<Vertex>     vertices_triangle;
    std::vector<Element>    elements_triangle;

    // A triangulation of the unit square with a regular pattern consisting of 13 vertices and 16 triangles,
    // showcasing vertices both on the boundary and in the interior.
    std::vector<Vertex>     vertices_unit_square_with_interior;
    std::vector<Element>    elements_unit_square_with_interior;

    std::vector<Vertex>     vertices_unit_square_1_interior_node;
    std::vector<Element>    elements_unit_square_1_interior_node;

    constexpr static IndexedMesh<>::SentinelType NO_NEIGHBOR = IndexedMesh<>::sentinel();
};

class indexed_mesh_refine_marked_test : public indexed_mesh_test
{

};

TEST_F(indexed_mesh_test, has_expected_properties)
{
    // Check that when we give a set of vertices and elements as input to an IndexedMesh,
    // the returned properties of the mesh correspond to the input choice.
    const auto & vertices = vertices_unit_square;
    const auto & elements = elements_unit_square;

    const auto mesh = IndexedMesh<>(vertices, elements);
    EXPECT_EQ(4u, mesh.num_vertices());
    EXPECT_EQ(2u, mesh.num_elements());

    EXPECT_THAT(mesh.vertices(), Pointwise(VertexDoubleEq(), vertices));
    EXPECT_THAT(mesh.elements(), Pointwise(Eq(), elements));
}

TEST_F(indexed_mesh_test, neighbors_for_unit_square)
{
    const auto mesh = IndexedMesh<>(vertices_unit_square, elements_unit_square);
    EXPECT_EQ(2u, mesh.neighbors().size());

    EXPECT_THAT(mesh.neighbors_for(0), ElementsAreArray({NO_NEIGHBOR, NO_NEIGHBOR, 1u}));
    EXPECT_THAT(mesh.neighbors_for(1), ElementsAreArray({0u, NO_NEIGHBOR, NO_NEIGHBOR}));
}

TEST_F(indexed_mesh_test, neighbors_for_diamond) {
    const auto & vertices = vertices_diamond;
    const auto & elements = elements_diamond;

    const auto mesh = IndexedMesh<>(vertices, elements);
    EXPECT_EQ(4u, mesh.neighbors().size());

    EXPECT_THAT(mesh.neighbors_for(0), ElementsAreArray({3u, NO_NEIGHBOR, 1u}));
    EXPECT_THAT(mesh.neighbors_for(1), ElementsAreArray({0u, NO_NEIGHBOR, 2u}));
    EXPECT_THAT(mesh.neighbors_for(2), ElementsAreArray({1u, NO_NEIGHBOR, 3u}));
    EXPECT_THAT(mesh.neighbors_for(3), ElementsAreArray({2u, NO_NEIGHBOR, 0u}));
}

TEST_F(indexed_mesh_test, neighbors_for_unit_square_with_interior_vertices)
{
    const auto mesh = IndexedMesh<>(vertices_unit_square_with_interior, elements_unit_square_with_interior);

    EXPECT_EQ(16u, mesh.neighbors().size());

    EXPECT_THAT(mesh.neighbors_for(0), ElementsAreArray({1u, 9u, NO_NEIGHBOR}));
    EXPECT_THAT(mesh.neighbors_for(1), ElementsAreArray({NO_NEIGHBOR, 2u, 0u}));
    EXPECT_THAT(mesh.neighbors_for(2), ElementsAreArray({3u, 7u, 1u}));
    EXPECT_THAT(mesh.neighbors_for(3), ElementsAreArray({NO_NEIGHBOR, 4u, 2u}));
    EXPECT_THAT(mesh.neighbors_for(4), ElementsAreArray({NO_NEIGHBOR, 5u, 3u}));
    EXPECT_THAT(mesh.neighbors_for(5), ElementsAreArray({14u, 6u, 4u}));
    EXPECT_THAT(mesh.neighbors_for(6), ElementsAreArray({5u, 15u, 7u}));
    EXPECT_THAT(mesh.neighbors_for(7), ElementsAreArray({8u, 2u, 6u}));
    EXPECT_THAT(mesh.neighbors_for(8), ElementsAreArray({7u, 15u, 9u}));
    EXPECT_THAT(mesh.neighbors_for(9), ElementsAreArray({8u, 10u, 0u}));
    EXPECT_THAT(mesh.neighbors_for(10), ElementsAreArray({9u, 11u, NO_NEIGHBOR}));
    EXPECT_THAT(mesh.neighbors_for(11), ElementsAreArray({10u, 12u, NO_NEIGHBOR}));
    EXPECT_THAT(mesh.neighbors_for(12), ElementsAreArray({15u, 13u, 11u}));
    EXPECT_THAT(mesh.neighbors_for(13), ElementsAreArray({14u, NO_NEIGHBOR, 12u}));
    EXPECT_THAT(mesh.neighbors_for(14), ElementsAreArray({NO_NEIGHBOR, 13u, 5u}));
    EXPECT_THAT(mesh.neighbors_for(15), ElementsAreArray({6u, 12u, 8u}));
}

TEST_F(indexed_mesh_test, boundary_for_unit_square)
{
    const auto mesh = IndexedMesh<>(vertices_unit_square, elements_unit_square);

    EXPECT_EQ(4u, mesh.num_boundary_vertices());
    EXPECT_EQ(4u, mesh.boundary_vertices().size());
    EXPECT_THAT(mesh.boundary_vertices(), ElementsAreArray({0u, 1u, 2u, 3u}));
}

TEST_F(indexed_mesh_test, boundary_for_unit_square_with_interior_vertices)
{
    const auto mesh = IndexedMesh<>(vertices_unit_square_with_interior, elements_unit_square_with_interior);

    EXPECT_EQ(8u, mesh.num_boundary_vertices());
    EXPECT_EQ(8u, mesh.boundary_vertices().size());
    EXPECT_THAT(mesh.boundary_vertices(), ElementsAreArray({0u, 1u, 2u, 5u, 7u, 10u, 11u, 12u}));
}

TEST_F(indexed_mesh_test, interior_for_unit_square)
{
    const auto mesh = IndexedMesh<>(vertices_unit_square, elements_unit_square);
    const auto interior = mesh.compute_interior_vertices();

    EXPECT_THAT(mesh.num_interior_vertices(), Eq(0u));
    EXPECT_THAT(interior, IsEmpty());
}

TEST_F(indexed_mesh_test, interior_for_unit_square_with_interior_vertices)
{
    const auto mesh = IndexedMesh<>(vertices_unit_square_with_interior, elements_unit_square_with_interior);
    const auto interior = mesh.compute_interior_vertices();

    EXPECT_THAT(mesh.num_interior_vertices(), Eq(5u));
    EXPECT_THAT(interior.size(), 5u);
    EXPECT_THAT(interior, ElementsAreArray({3u, 4u, 6u, 8u, 9u}));
}

TEST_F(indexed_mesh_test, interior_for_unit_square_1_interior_node)
{
    // This test is interesting because it exhibits the case where there are interior indices which
    // are higher than that of the boundary. In a previous implementation of compute_interior_nodes(),
    // this was not taken into account, and so it caused memory corruption. This test effectively
    // checks for this case, but is only reliable as long as bounds checking is enabled in the compiler
    // for debug builds.

    const auto mesh = IndexedMesh<>(vertices_unit_square_1_interior_node, elements_unit_square_1_interior_node);
    const auto interior = mesh.compute_interior_vertices();

    EXPECT_THAT(mesh.num_interior_vertices(), Eq(1u));
    EXPECT_THAT(interior.size(), 1u);
    EXPECT_THAT(interior, ElementsAreArray({4u}));
}

TEST_F(indexed_mesh_test, ancestors_for_unit_square) {
    const auto mesh = IndexedMesh<>(vertices_unit_square, elements_unit_square);

    EXPECT_THAT(mesh.ancestor_for(0), Eq(0u));
    EXPECT_THAT(mesh.ancestor_for(1), Eq(1u));
}

TEST_F(indexed_mesh_test, ancestors_for_diamond) {
    const auto mesh = IndexedMesh<>(vertices_diamond, elements_diamond);

    EXPECT_THAT(mesh.ancestor_for(0), Eq(0u));
    EXPECT_THAT(mesh.ancestor_for(1), Eq(1u));
    EXPECT_THAT(mesh.ancestor_for(2), Eq(2u));
    EXPECT_THAT(mesh.ancestor_for(3), Eq(3u));
}

TEST_F(indexed_mesh_test, reset_ancestry) {
    std::vector<Vertex> vertices {
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(1.0, 1.0),
            Vertex(0.0, 1.0)
    };

    std::vector<Element> elements {
            Element({3, 0, 1}),
            Element({1, 2, 3})
    };

    auto mesh = IndexedMesh<>(vertices, elements);
    mesh.bisect_marked({0, 1});
    EXPECT_THAT(mesh.ancestor_for(0), Eq(0u));
    EXPECT_THAT(mesh.ancestor_for(1), Eq(1u));
    EXPECT_THAT(mesh.ancestor_for(2), Eq(0u));
    EXPECT_THAT(mesh.ancestor_for(3), Eq(1u));

    mesh.reset_ancestry();

    EXPECT_THAT(mesh.ancestor_for(0), Eq(0u));
    EXPECT_THAT(mesh.ancestor_for(1), Eq(1u));
    EXPECT_THAT(mesh.ancestor_for(2), Eq(2u));
    EXPECT_THAT(mesh.ancestor_for(3), Eq(3u));
}

TEST_F(indexed_mesh_refine_marked_test, single_triangle)
{
    const auto & vertices = vertices_triangle;
    const auto & elements = elements_triangle;

    std::vector<Vertex> expected_vertices = vertices;
    expected_vertices.push_back(Vertex(0.5, 0.5));

    std::vector<Element> expected_elements {
            Element({0, 3, 2}),
            Element({1, 3, 0})
    };

    auto mesh = IndexedMesh<>(vertices, elements);
    mesh.bisect_marked({0});

    EXPECT_EQ(4u, mesh.vertices().size());
    EXPECT_EQ(2u, mesh.elements().size());

    EXPECT_THAT(mesh.vertices(), Pointwise(VertexDoubleEq(), expected_vertices));
    EXPECT_THAT(mesh.elements(), Pointwise(Eq(), expected_elements));

    EXPECT_THAT(mesh.neighbors_for(0), ElementsAreArray({1u, NO_NEIGHBOR, NO_NEIGHBOR}));
    EXPECT_THAT(mesh.neighbors_for(1), ElementsAreArray({NO_NEIGHBOR, 0u, NO_NEIGHBOR}));

    EXPECT_THAT(mesh.boundary_vertices(), ElementsAreArray({0, 1, 2, 3}));

    EXPECT_THAT(mesh.ancestor_for(0), Eq(0u));
    EXPECT_THAT(mesh.ancestor_for(1), Eq(0u));
}

TEST_F(indexed_mesh_refine_marked_test, two_triangles_with_shared_refinement_edge)
{
    // A unit square with a shared refinement edge between the two triangles
    std::vector<Vertex> vertices {
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(1.0, 1.0),
            Vertex(0.0, 1.0)
    };

    std::vector<Element> elements {
            Element({3, 0, 1}),
            Element({1, 2, 3})
    };

    std::vector<Vertex> expected_vertices {
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(1.0, 1.0),
            Vertex(0.0, 1.0),
            Vertex(0.5, 0.5)
    };

    std::vector<Element> expected_elements_both_refined {
            Element({0, 4, 3}),
            Element({2, 4, 1}),
            Element({1, 4, 0}),
            Element({3, 4, 2})
    };

    std::vector<Element> expected_elements_0_refined = expected_elements_both_refined;

    std::vector<Element> expected_elements_1_refined {
            Element({0, 4, 3}),
            Element({2, 4, 1}),
            Element({3, 4, 2}),
            Element({1, 4, 0})
    };

    auto mesh_0_refined = IndexedMesh<>(vertices, elements);
    mesh_0_refined.bisect_marked({0});

    auto mesh_1_refined = IndexedMesh<>(vertices, elements);
    mesh_1_refined.bisect_marked({1});

    auto mesh_both_refined = IndexedMesh<>(vertices, elements);
    mesh_both_refined.bisect_marked({0, 1});

    const auto NO_NEIGHBOR = IndexedMesh<>::sentinel();
    EXPECT_THAT(mesh_0_refined.vertices(), Pointwise(VertexDoubleEq(), expected_vertices));
    EXPECT_THAT(mesh_0_refined.elements(), Pointwise(Eq(), expected_elements_0_refined));
    EXPECT_THAT(mesh_0_refined.boundary_vertices(), ElementsAreArray({0, 1, 2, 3}));
    EXPECT_THAT(mesh_0_refined.neighbors_for(0), ElementsAre(2, 3, NO_NEIGHBOR));
    EXPECT_THAT(mesh_0_refined.neighbors_for(1), ElementsAre(3, 2, NO_NEIGHBOR));
    EXPECT_THAT(mesh_0_refined.neighbors_for(2), ElementsAre(1, 0, NO_NEIGHBOR));
    EXPECT_THAT(mesh_0_refined.neighbors_for(3), ElementsAre(0, 1, NO_NEIGHBOR));
    EXPECT_THAT(mesh_0_refined.ancestor_for(0), Eq(0u));
    EXPECT_THAT(mesh_0_refined.ancestor_for(1), Eq(1u));
    EXPECT_THAT(mesh_0_refined.ancestor_for(2), Eq(0u));
    EXPECT_THAT(mesh_0_refined.ancestor_for(3), Eq(1u));

    EXPECT_THAT(mesh_1_refined.vertices(), Pointwise(VertexDoubleEq(), expected_vertices));
    EXPECT_THAT(mesh_1_refined.elements(), Pointwise(Eq(), expected_elements_1_refined));
    EXPECT_THAT(mesh_1_refined.boundary_vertices(), ElementsAreArray({0, 1, 2, 3}));
    EXPECT_THAT(mesh_1_refined.neighbors_for(0), ElementsAre(3, 2, NO_NEIGHBOR));
    EXPECT_THAT(mesh_1_refined.neighbors_for(1), ElementsAre(2, 3, NO_NEIGHBOR));
    EXPECT_THAT(mesh_1_refined.neighbors_for(2), ElementsAre(0, 1, NO_NEIGHBOR));
    EXPECT_THAT(mesh_1_refined.neighbors_for(3), ElementsAre(1, 0, NO_NEIGHBOR));
    EXPECT_THAT(mesh_1_refined.ancestor_for(0), Eq(0u));
    EXPECT_THAT(mesh_1_refined.ancestor_for(1), Eq(1u));
    EXPECT_THAT(mesh_1_refined.ancestor_for(2), Eq(1u));
    EXPECT_THAT(mesh_1_refined.ancestor_for(3), Eq(0u));

    EXPECT_THAT(mesh_both_refined.vertices(), Pointwise(VertexDoubleEq(), expected_vertices));
    EXPECT_THAT(mesh_both_refined.elements(), Pointwise(Eq(), expected_elements_both_refined));
    EXPECT_THAT(mesh_both_refined.boundary_vertices(), ElementsAreArray({0, 1, 2, 3}));
    EXPECT_THAT(mesh_both_refined.neighbors_for(0), ElementsAre(2, 3, NO_NEIGHBOR));
    EXPECT_THAT(mesh_both_refined.neighbors_for(1), ElementsAre(3, 2, NO_NEIGHBOR));
    EXPECT_THAT(mesh_both_refined.neighbors_for(2), ElementsAre(1, 0, NO_NEIGHBOR));
    EXPECT_THAT(mesh_both_refined.neighbors_for(3), ElementsAre(0, 1, NO_NEIGHBOR));
    EXPECT_THAT(mesh_both_refined.ancestor_for(0), Eq(0u));
    EXPECT_THAT(mesh_both_refined.ancestor_for(1), Eq(1u));
    EXPECT_THAT(mesh_both_refined.ancestor_for(2), Eq(0u));
    EXPECT_THAT(mesh_both_refined.ancestor_for(3), Eq(1u));
}

TEST_F(indexed_mesh_refine_marked_test, two_triangles_without_shared_refinement_edge)
{
    // A unit square with *no* shared refinement edge between the two triangles
    std::vector<Vertex> vertices {
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(1.0, 1.0),
            Vertex(0.0, 1.0)
    };

    std::vector<Element> elements {
            Element({3, 0, 1}),
            Element({2, 3, 1})
    };

    auto expected_vertices_0_refined = vertices;
    expected_vertices_0_refined.push_back(Vertex(0.5, 0.5));
    expected_vertices_0_refined.push_back(Vertex(1.0, 0.5));

    auto expected_vertices_1_refined = vertices;
    expected_vertices_1_refined.push_back(Vertex(1.0, 0.5));

    auto expected_vertices_both_refined = expected_vertices_0_refined;

    auto expected_elements_0_refined = std::vector<Element> {
            Element({0, 4, 3}),
            Element({3, 5, 2}),
            Element({1, 4, 0}),
            Element({5, 4, 1}),
            Element({3, 4, 5})
    };

    auto expected_elements_1_refined = std::vector<Element> {
            Element({3, 0, 1}),
            Element({3, 4, 2}),
            Element({1, 4, 3})
    };

    auto expected_elements_both_refined = expected_elements_0_refined;

    auto mesh_0_refined = IndexedMesh<>(vertices, elements);
    mesh_0_refined.bisect_marked({0});

    auto mesh_1_refined = IndexedMesh<>(vertices, elements);
    mesh_1_refined.bisect_marked({1});

    auto mesh_both_refined = IndexedMesh<>(vertices, elements);
    mesh_both_refined.bisect_marked({0, 1});

    const auto NO_NEIGHBOR = IndexedMesh<>::sentinel();
    EXPECT_THAT(mesh_0_refined.vertices(), Pointwise(VertexDoubleEq(), expected_vertices_0_refined));
    EXPECT_THAT(mesh_0_refined.elements(), Pointwise(Eq(), expected_elements_0_refined));
    EXPECT_THAT(mesh_0_refined.boundary_vertices(), ElementsAreArray({0, 1, 2, 3, 5}));
    EXPECT_THAT(mesh_0_refined.neighbors_for(0), ElementsAre(2, 4, NO_NEIGHBOR));
    EXPECT_THAT(mesh_0_refined.neighbors_for(1), ElementsAre(4, NO_NEIGHBOR, NO_NEIGHBOR));
    EXPECT_THAT(mesh_0_refined.neighbors_for(2), ElementsAre(3, 0, NO_NEIGHBOR));
    EXPECT_THAT(mesh_0_refined.neighbors_for(3), ElementsAre(4, 2, NO_NEIGHBOR));
    EXPECT_THAT(mesh_0_refined.neighbors_for(4), ElementsAre(0, 3, 1));
    EXPECT_THAT(mesh_0_refined.ancestor_for(0), Eq(0u));
    EXPECT_THAT(mesh_0_refined.ancestor_for(1), Eq(1u));
    EXPECT_THAT(mesh_0_refined.ancestor_for(2), Eq(0u));
    EXPECT_THAT(mesh_0_refined.ancestor_for(3), Eq(1u));
    EXPECT_THAT(mesh_0_refined.ancestor_for(4), Eq(1u));

    EXPECT_THAT(mesh_1_refined.vertices(), Pointwise(VertexDoubleEq(), expected_vertices_1_refined));
    EXPECT_THAT(mesh_1_refined.elements(), Pointwise(Eq(), expected_elements_1_refined));
    EXPECT_THAT(mesh_1_refined.boundary_vertices(), ElementsAreArray({0, 1, 2, 3, 4}));
    EXPECT_THAT(mesh_1_refined.neighbors_for(0), ElementsAre(NO_NEIGHBOR, NO_NEIGHBOR, 2));
    EXPECT_THAT(mesh_1_refined.neighbors_for(1), ElementsAre(2, NO_NEIGHBOR, NO_NEIGHBOR));
    EXPECT_THAT(mesh_1_refined.neighbors_for(2), ElementsAre(NO_NEIGHBOR, 1, 0));
    EXPECT_THAT(mesh_1_refined.ancestor_for(0), Eq(0u));
    EXPECT_THAT(mesh_1_refined.ancestor_for(1), Eq(1u));
    EXPECT_THAT(mesh_1_refined.ancestor_for(2), Eq(1u));

    EXPECT_THAT(mesh_both_refined.vertices(), Pointwise(VertexDoubleEq(), expected_vertices_both_refined));
    EXPECT_THAT(mesh_both_refined.elements(), Pointwise(Eq(), expected_elements_both_refined));
    EXPECT_THAT(mesh_both_refined.boundary_vertices(), ElementsAreArray({0, 1, 2, 3, 5}));
    EXPECT_THAT(mesh_both_refined.neighbors_for(0), ElementsAre(2, 4, NO_NEIGHBOR));
    EXPECT_THAT(mesh_both_refined.neighbors_for(1), ElementsAre(4, NO_NEIGHBOR, NO_NEIGHBOR));
    EXPECT_THAT(mesh_both_refined.neighbors_for(2), ElementsAre(3, 0, NO_NEIGHBOR));
    EXPECT_THAT(mesh_both_refined.neighbors_for(3), ElementsAre(4, 2, NO_NEIGHBOR));
    EXPECT_THAT(mesh_both_refined.neighbors_for(4), ElementsAre(0, 3, 1));
    EXPECT_THAT(mesh_both_refined.ancestor_for(0), Eq(0u));
    EXPECT_THAT(mesh_both_refined.ancestor_for(1), Eq(1u));
    EXPECT_THAT(mesh_both_refined.ancestor_for(2), Eq(0u));
    EXPECT_THAT(mesh_both_refined.ancestor_for(3), Eq(1u));
    EXPECT_THAT(mesh_both_refined.ancestor_for(4), Eq(1u));
}

TEST_F(indexed_mesh_refine_marked_test, unit_square_regression)
{
    // It was discovered that this particular mesh configuration
    // made bisect_marked produce the wrong mesh.

    const std::vector<Vertex> vertices {
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(1.0, 1.0),
            Vertex(0.0, 1.0),
            Vertex(0.5, 0.5)
    };

    const std::vector<Element> elements {
            Element({0, 4, 3}),
            Element({2, 4, 1}),
            Element({1, 4, 0}),
            Element({3, 4, 2})
    };

    auto mesh = IndexedMesh<>(vertices, elements);
    mesh.bisect_marked({0, 1, 2, 3});

    const std::vector<Vertex> expected_vertices {
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(1.0, 1.0),
            Vertex(0.0, 1.0),
            Vertex(0.5, 0.5),
            Vertex(0.0, 0.5),
            Vertex(1.0, 0.5),
            Vertex(0.5, 0.0),
            Vertex(0.5, 1.0)
    };

    const std::vector<Element> expected_elements {
            Element({4, 5, 0}),
            Element({4, 6, 2}),
            Element({4, 7, 1}),
            Element({4, 8, 3}),
            Element({3, 5, 4}),
            Element({1, 6, 4}),
            Element({0, 7, 4}),
            Element({2, 8, 4})
    };

    EXPECT_THAT(mesh.vertices(), Pointwise(VertexDoubleEq(), expected_vertices));
    EXPECT_THAT(mesh.elements(), ElementsAreArray(expected_elements));

    // TODO: Neighbors
}

TEST_F(indexed_mesh_refine_marked_test, throws_when_marked_contains_nonexisting_element_indices)
{
    auto mesh = IndexedMesh<>(vertices_unit_square_with_interior, elements_unit_square_with_interior);
    // The mesh in question has 16 elements, so index 16 is out of bounds, and so we expect it to throw!
    ASSERT_THROW(mesh.bisect_marked({0, 2, 6, 16}), std::invalid_argument);
}


TEST_F(indexed_mesh_refine_marked_test, double_bisection_of_element)
{
    // This particular configuration examines a special case when the same element index needs to be
    // bisected twice in order to assure conformity.
    const std::vector<Vertex> vertices{
            Vertex(-1.0, 0.5),
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(2.0, 1.0),
            Vertex(0.0, 1.0)
    };

    const std::vector<Element> elements{
            Element({1, 0, 4}),
            Element({4, 1, 2}),
            Element({4, 2, 3}),
    };

    auto mesh = IndexedMesh<>(vertices, elements);
    mesh.bisect_marked({0, 1});

    // By creating a new mesh from the vertices and elements,
    // certain checks will be run by the constructor to check for conformity.
    const auto reconstructed_mesh = IndexedMesh<>(mesh.vertices(), mesh.elements());

    const std::vector<Vertex> expected_vertices{
            Vertex(-1.0, 0.5),
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(2.0, 1.0),
            Vertex(0.0, 1.0),
            Vertex(0.0, 0.5),
            Vertex(0.5, 0.5),
            Vertex(1.0, 1.0)
    };

    const std::vector<Element> expected_elements{
            Element({0, 5, 1}),
            Element({6, 5, 1}),
            Element({7, 6, 2}),
            Element({4, 5, 0}),
            Element({2, 6, 1}),
            Element({4, 5, 6}),
            Element({3, 7, 2}),
            Element({4, 6, 7})
    };

    const auto NO_NEIGHBOR = reconstructed_mesh.sentinel();

    EXPECT_THAT(reconstructed_mesh.vertices(), Pointwise(VertexDoubleEq(), expected_vertices));
    EXPECT_THAT(reconstructed_mesh.elements(), ElementsAreArray(expected_elements));

    EXPECT_THAT(reconstructed_mesh.neighbors_for(0), ElementsAre(3, 1, NO_NEIGHBOR));
    EXPECT_THAT(reconstructed_mesh.neighbors_for(1), ElementsAre(5, 0, 4));
    EXPECT_THAT(reconstructed_mesh.neighbors_for(2), ElementsAre(7, 4, 6));
    EXPECT_THAT(reconstructed_mesh.neighbors_for(3), ElementsAre(5, 0, NO_NEIGHBOR));
    EXPECT_THAT(reconstructed_mesh.neighbors_for(4), ElementsAre(2, 1, NO_NEIGHBOR));
    EXPECT_THAT(reconstructed_mesh.neighbors_for(5), ElementsAre(3, 1, 7));
    EXPECT_THAT(reconstructed_mesh.neighbors_for(6), ElementsAre(NO_NEIGHBOR, 2, NO_NEIGHBOR));
    EXPECT_THAT(reconstructed_mesh.neighbors_for(7), ElementsAre(5, 2, NO_NEIGHBOR));
}

RC_GTEST_FIXTURE_PROP(indexed_mesh_refine_marked_test, elements_have_nonzero_area, ())
{
    const auto mesh = *crest::gen::arbitrary_unit_square_mesh(14);

    const auto extra_mesh = IndexedMesh<double, int>(mesh.vertices(), mesh.elements());

    int num_nonzero_area = 0;
    for (int k = 0; k < mesh.num_elements(); ++k)
    {
        const auto triangle = mesh.triangle_for(k);
        if (crest::area(triangle) == 0)
        {
            ++num_nonzero_area;
        }
    }
    RC_ASSERT(num_nonzero_area == 0);
}
