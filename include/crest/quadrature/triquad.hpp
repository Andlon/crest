#pragma once

#include <array>
#include <cstdint>
#include <cmath>
#include <crest/geometry/vertex.hpp>
#include <crest/quadrature/rules.hpp>

namespace crest
{
    template <typename Scalar>
    class ReferenceTriangleTransform
    {
    public:
        ReferenceTriangleTransform(Vertex<Scalar> v0, Vertex<Scalar> v1, Vertex<Scalar> v2);

        Vertex<Scalar> transform_from_reference(const Scalar & x, const Scalar & y) const;
        Scalar absolute_determinant() const { return _absdet; }

    private:
        Vertex<Scalar> _v0;
        Vertex<Scalar> _v1;
        Vertex<Scalar> _v2;
        Scalar _absdet;
    };

    template <typename S>
    ReferenceTriangleTransform<S>::ReferenceTriangleTransform(Vertex<S> v0,
                                                              Vertex<S> v1,
                                                              Vertex<S> v2)
            : _v0(v0), _v1(v1), _v2(v2)
    {
        _absdet = (1.0 / 4.0) * std::abs(_v1.x * _v2.y - _v1.y * _v2.x);
    }

    /**
     *The reference triangle the quadrature rules are defined on is given by the vertices
     *(-1, 1), (-1, -1), (1, -1),
     *so for a triangle with corners A, B, C, we have that each coordinate x in R^2 is given by the relation
     *
     * x = 1/2 (A + B) + 1/2 * (A - C) z_1 + 1/2 * (B - C) z_2
     *
     * for a reference coordinate z = (z_1, z_2) in R^2. This implies the following mapping:
     *  A <-> ( 1, -1)
     *  B <-> (-1,  1)
     *  C <-> (-1, -1)
     */
    template <typename S>
    Vertex<S> ReferenceTriangleTransform<S>::transform_from_reference(const S & x, const S & y) const
    {

        return 0.5 * (_v0 + x * _v1 + y * _v2);
    }

    /**
     *
     * Returns an instance of a ReferenceTriangleTransform<Scalar>, which transforms points from
     * the reference triangle used by triquad to points in the triangle defined by the vertices (a, b, c).
     *
     * @param a
     * @param b
     * @param c
     * @return An instance of ReferenceTriangleTransform<Scalar>.
     */
    template <typename Scalar>
    ReferenceTriangleTransform<Scalar> triquad_transform(const Vertex<Scalar> & a,
                                                         const Vertex<Scalar> & b,
                                                         const Vertex<Scalar> & c)
    {
        const auto v0 = a + b;
        const auto v1 = a - c;
        const auto v2 = b - c;

        return ReferenceTriangleTransform<Scalar>(v0, v1, v2);
    };

    /**
     * Computes the integral of the function f(x, y) -> R on the reference triangled defined by the vertices
     * (-1, 1), (-1, -1), (1, -1).
     * @param f
     * @return
     */
    template <unsigned int Strength, typename Scalar, typename Function2D>
    constexpr inline Scalar triquad_ref(const Function2D & f)
    {
        using quadrature = ::crest::quadrature_rules::quadrature<Scalar, Strength>;
        constexpr auto quad = quadrature();

        Scalar result = static_cast<Scalar>(0.0);
        for (unsigned int i = 0; i < quadrature::num_points; ++i)
        {
            result += quad.w[i] * f(quad.x[i], quad.y[i]);
        }

        return result;
    };


    /**
     * Computes the integral of the function f over the triangle determined by the vertices a, b and c,
     * using Gauss quadrature rules of the given Strength.
     *
     * The Strength of the quadrature rule determines the highest degree polynomial that the quadrature rule
     * is able to integrate *exactly*. For example, a Strength of 4 indicates that triquad computes the exact
     * integral of any polynomial function of *total degree* 4.
     * @param f A callable function f(x, y) that operates on real numbers.
     * @param a
     * @param b
     * @param c
     * @return
     */
    template <unsigned int Strength, typename Scalar, typename Function2D>
    constexpr inline Scalar triquad(
            const Function2D & f,
            const Vertex<Scalar> & a,
            const Vertex<Scalar> & b,
            const Vertex<Scalar> & c)
    {
        const auto transform = triquad_transform(a, b, c);
        const auto f_transformed = [&f, &transform] (auto x, auto y)
        {
            const auto coords = transform.transform_from_reference(x, y);
            return f(coords.x, coords.y);
        };
        return transform.absolute_determinant() * triquad_ref<Strength, Scalar>(f_transformed);
    };
}