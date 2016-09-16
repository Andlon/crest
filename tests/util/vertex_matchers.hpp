#pragma once

#include <crest/geometry/vertex.hpp>
#include <gmock/gmock.h>

MATCHER_P(VertexDoubleEq, other, "")
{
    return ::testing::Value(arg.x, ::testing::DoubleEq(other.x))
        && ::testing::Value(arg.y, ::testing::DoubleEq(other.y));
}

MATCHER_P(VertexFloatEq, other, "")
{
    return ::testing::Value(arg.x, ::testing::FloatEq(other.x))
        && ::testing::Value(arg.y, ::testing::FloatEq(other.y));
}

MATCHER(VertexDoubleEq, "are close to each other, in a double precision sense")
{
    return ::testing::Value(::testing::get<0>(arg), VertexDoubleEq(::testing::get<1>(arg)));
}

MATCHER(VertexFloatEq, "are close to each other, in a single precision sense")
{
    return ::testing::Value(::testing::get<0>(arg), VertexFloatEq(::testing::get<1>(arg)));
}
