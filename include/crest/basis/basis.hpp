#pragma once

#include <Eigen/Sparse>

#include <crest/util/eigen_extensions.hpp>

namespace crest
{

    enum class Norm
    {
        L2,
        H1Semi,
        H1
    };

    template <typename Scalar>
    struct Assembly
    {
        Eigen::SparseMatrix<Scalar> stiffness;
        Eigen::SparseMatrix<Scalar> mass;
    };

    template <typename Scalar, typename Impl>
    class Basis
    {
    public:
        virtual ~Basis() {}

        virtual Assembly<Scalar> assemble() const = 0;

        /**
         * The number of degrees of freedom associated with this basis.
         * @return
         */
        virtual int num_dof() const = 0;

        /**
         * Given a continuous function f, interpolate it in the space
         * represented by this basis, and return a vector of weights, such that
         * element i in the vector corresponds to the weight factor of basis function i.
         *
         * Naturally, there are multiple ways to interpolate a function in a finite
         * element space, so to have a single function for this is a simplifaction,
         * and for now we'll leave it up to the concrete implementation which
         * interpolation this corresponds to.
         *
         * @param f
         * @return
         */
        template <typename Function2d>
        VectorX<Scalar> interpolate(const Function2d &f) const
        {
            return static_cast<Impl *>(this)->interpolate<Function2d>(f);
        }

        /**
         * Computes the L2 inner product of a continuous function f
         * and every basis function b_i for every degree of freedom i.
         * More precisely, returns a vector whose ith element is
         * determined by the L2 inner product (f, b_i), where b_i
         * is the basis function associated with degree of freedom i.
         *
         * This is frequently used to compute the load vector in FEM
         * applications.
         * @param weights
         * @param f
         * @return
         */
        template <int QuadStrength, typename Function2d>
        VectorX<Scalar> load(const Function2d &f) const
        {
            return static_cast<Impl *>(this)->load<QuadStrength, Function2d>(f);
        };

        /**
         * Computes an approximation of the error between a continuous function f
         * and a function g in the space spanned by the basis given by
         * its basis weights in the specified norm.
         *
         * More precisely, if g = sum w_i b_i for all degrees of freedom i,
         * where w_i is given by weights(i) and b_i denotes the basis function
         * associated with the degree of freedom i, then this function computes
         *
         * ||f - g||
         *
         * in the specified norm.
         *
         * Note that for implementers, only Norm::L2 and Norm::H1Semi needs
         * to be implemented, as the base class is able to compute Norm::H1 from these.
         * @param f
         * @param weights
         * @param norm
         * @return
         */
        template <int QuadStrength, typename Function2d>
        Scalar error(const Function2d &f, const VectorX<Scalar> & weights, Norm norm) const;
    };
}
