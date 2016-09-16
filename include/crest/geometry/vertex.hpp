//
// Created by andreas on 24.08.16.
//

#pragma once

#include <ostream>

namespace crest {

    template <typename T>
    struct Vertex {
        T x;
        T y;

        Vertex(T x, T y) : x(x), y(y) {}
    };

    /*
     * OPERATORS
     */

    template <typename T>
    Vertex<T> operator+(const Vertex<T> &a, const Vertex<T> &b)
    {
        return Vertex<T>(a.x + b.x, a.y + b.y);
    }

    template <typename T>
    Vertex<T> operator-(const Vertex<T> &a, const Vertex<T> &b)
    {
        return Vertex<T>(a.x - b.x, a.y - b.y);
    }

    template <typename T>
    Vertex<T> operator/(const Vertex<T> &v, T divisor)
    {
        return Vertex<T>(v.x / divisor, v.y / divisor);
    }

    template <typename T>
    Vertex<T> operator*(const Vertex<T> &v, T factor)
    {
        return Vertex<T>(factor * v.x, factor * v.y);
    }

    template <typename T>
    Vertex<T> operator*(T factor, const Vertex<T> &v)
    {
        return v * factor;
    }

    template <typename T>
    std::ostream & operator<<(std::ostream & o, const Vertex<T> &v)
    {
        o << "[ " << v.x << ", " << v.y << " ]";
        return o;
    }

    /*
     * UTILITY FUNCTIONS
     */

    template <typename T>
    Vertex<T> midpoint(const Vertex<T> & a, const Vertex<T> &b)
    {
        return (a + b) / static_cast<T>(2);
    }

}
