#pragma once

#include <crest/geometry/vertex.hpp>
#include <cmath>
#include <algorithm>

namespace crest
{
    template <typename T>
    struct Triangle {
        Vertex<T> a;
        Vertex<T> b;
        Vertex<T> c;

        explicit Triangle<T>(Vertex<T> a, Vertex<T> b, Vertex<T> c) : a(a), b(b), c(c) {}
    };

    template <typename T>
    inline T diameter_squared(const Triangle<T> & triangle)
    {
        const auto ac = norm_squared(triangle.c - triangle.a);
        const auto ab = norm_squared(triangle.b - triangle.a);
        const auto bc = norm_squared(triangle.c - triangle.b);
        return std::max<T>({ac, ab, bc});
    }

    template <typename T>
    inline T diameter(const Triangle<T> & triangle)
    {
        return sqrt(diameter_squared(triangle));
    }

    template <typename T>
    inline T distance_squared(const Triangle<T> & triangle, const Vertex<T> & point)
    {
        const auto a_dist = norm_squared(triangle.a - point);
        const auto b_dist = norm_squared(triangle.b - point);
        const auto c_dist = norm_squared(triangle.c - point);
        return std::max<T>({a_dist, b_dist, c_dist});
    }

    template <typename T>
    inline T distance(const Triangle<T> & triangle, const Vertex<T> & point)
    {
        return sqrt(distance_squared(triangle, point));
    }

    template <typename T>
    inline T area(const Triangle<T> & triangle)
    {
        const auto ac = triangle.c - triangle.a;
        const auto ab = triangle.b - triangle.a;
        return abs(cross(ac, ab)) / 2.0;
    }

}