#include <crest/geometry/refinement.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <util/vertex_matchers.hpp>

using Vertex = crest::Vertex<double>;
using Element = crest::Element<unsigned int>;

using ::testing::Pointwise;
using ::testing::ElementsAreArray;
using ::testing::Eq;

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
    const auto mesh = crest::refine_to_tolerance(unit_square_2_elements, 2.0);

    EXPECT_THAT(mesh.elements(), Pointwise(Eq(), unit_square_2_elements.elements()));
    EXPECT_THAT(mesh.vertices(), Pointwise(VertexDoubleEq(), unit_square_2_elements.vertices()));
    EXPECT_THAT(mesh.neighbors_for(0), Pointwise(Eq(), unit_square_2_elements.neighbors_for(0)));
    EXPECT_THAT(mesh.neighbors_for(1), Pointwise(Eq(), unit_square_2_elements.neighbors_for(1)));
}

TEST_F(refinement_test, expected_single_bisection)
{
    const auto mesh = crest::refine_to_tolerance(unit_square_2_elements, 1.1);

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
