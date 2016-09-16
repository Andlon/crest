#include <util/vertex_matchers.hpp>
#include <crest/geometry/vertex.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using crest::Vertex;

using ::testing::Not;
using ::testing::Pointwise;

TEST(vertex_matchers, vertex_double_eq)
{
    auto v1 = Vertex<double>(0.0, 1.0);
    auto v2 = Vertex<double>(1.0, 0.0);
    auto v3 = v1;
    ASSERT_THAT(v1, Not(VertexDoubleEq(v2)));
    ASSERT_THAT(v1, VertexDoubleEq(v3));
}

TEST(vertex_matchers, vertex_float_eq)
{
    auto v1 = Vertex<float>(0.0, 1.0);
    auto v2 = Vertex<float>(1.0, 0.0);
    auto v3 = v1;
    ASSERT_THAT(v1, Not(VertexFloatEq(v2)));
    ASSERT_THAT(v1, VertexFloatEq(v3));
}

TEST(vertex_matchers, vertex_double_eq_is_compatible_with_pointwise)
{
    auto v1 = std::vector<Vertex<double>> {
            Vertex<double>(0.0, 0.0),
            Vertex<double>(0.0, 1.0)
    };
    auto v2 = std::vector<Vertex<double>> {
            Vertex<double>(0.0, 0.0),
            Vertex<double>(1.0, 1.0)
    };
    auto v3 = v1;
    ASSERT_THAT(v1, Not(Pointwise(VertexDoubleEq(), v2)));
    ASSERT_THAT(v1, Pointwise(VertexDoubleEq(), v3));
}

TEST(vertex_matchers, vertex_float_eq_is_compatible_with_pointwise)
{
    auto v1 = std::vector<Vertex<float>> {
            Vertex<float>(0.0, 0.0),
            Vertex<float>(0.0, 1.0)
    };
    auto v2 = std::vector<Vertex<float>> {
            Vertex<float>(0.0, 0.0),
            Vertex<float>(1.0, 1.0)
    };
    auto v3 = v1;
    ASSERT_THAT(v1, Not(Pointwise(VertexFloatEq(), v2)));
    ASSERT_THAT(v1, Pointwise(VertexFloatEq(), v3));
}
