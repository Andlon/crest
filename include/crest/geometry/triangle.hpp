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
    inline T diameter_squared(const Triangle<T> & triangle) {
        const auto ac = norm_squared(triangle.c - triangle.a);
        const auto ab = norm_squared(triangle.b - triangle.a);
        const auto bc = norm_squared(triangle.c - triangle.b);
        return std::max<T>({ac, ab, bc});
    }

    template <typename T>
    inline T diameter(const Triangle<T> & triangle) {
        return sqrt(diameter_squared(triangle));
    }

}