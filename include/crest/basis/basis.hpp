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

        virtual std::vector<int> interior_nodes() const = 0;
        virtual std::vector<int> boundary_nodes() const = 0;

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
            return static_cast<const Impl *>(this)->template interpolate<Function2d>(f);
        }

        /*
         * Given a continuous function g, evaluate the function at each boundary node
         * and return a vector of weights.
         */
        template <typename Function2d>
        VectorX<Scalar> interpolate_boundary(const Function2d &f) const
        {
            return static_cast<const Impl *>(this)->template interpolate_boundary<Function2d>(f);
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
            return static_cast<const Impl *>(this)->template load<QuadStrength, Function2d>(f);
        };

        /**
         * Computes an approximation of the error between a continuous function f
         * and a function g in the space spanned by the basis given by
         * its basis weights in the L2 norm.
         *
         * More precisely, if g = sum w_i b_i for all degrees of freedom i,
         * where w_i is given by weights(i) and b_i denotes the basis function
         * associated with the degree of freedom i, then this function computes
         *
         * ||f - g||
         *
         * in the L2 norm.
         *
         * @param f
         * @param weights
         * @param norm
         * @return
         */
        template <int QuadStrength, typename Function2d>
        Scalar error_l2(const Function2d &f, const VectorX<Scalar> & weights) const;

        /**
         * Computes an approximation of the error between a continuous function f
         * and a function g in the space spanned by the basis given by
         * its basis weights in the H1 semi-norm.
         *
         * More precisely, if g = sum w_i b_i for all degrees of freedom i,
         * where w_i is given by weights(i) and b_i denotes the basis function
         * associated with the degree of freedom i, then this function computes
         *
         * ||f - g||
         *
         * in the H1 semi-norm, which is equivalent to
         *
         * || grad(f) - grad(g) ||
         *
         * in the L2 norm. Note that one specifies the derivatives f_x and f_y
         * for the computation of the H1 semi-norm.
         *
         * @param f
         * @param weights
         * @param norm
         * @return
         */
        template <int QuadStrength, typename Function2d_x, typename Function2d_y>
        Scalar error_h1_semi(const Function2d_x & f_x,
                             const Function2d_y & f_y,
                             const VectorX<Scalar> & weights) const;

        /**
         * Computes an approximation of the error between a continuous function f
         * and a function g in the space spanned by the basis given by
         * its basis weights in the H1 norm.
         *
         * More precisely, if g = sum w_i b_i for all degrees of freedom i,
         * where w_i is given by weights(i) and b_i denotes the basis function
         * associated with the degree of freedom i, then this function computes
         *
         * ||f - g||
         *
         * in the H1 norm. Note that the computation requires f, as well as its
         * spatial derivatives f_x and f_y.
         *
         * Also note that implementers of subclasses do not need to reimplement this
         * function, as it is implemented in terms of error_l2 and error_h1_semi.
         *
         * @param f
         * @param weights
         * @param norm
         * @return
         */
        template <int QuadStrength, typename Function2d, typename Function2d_x, typename Function2d_y>
        Scalar error_h1(const Function2d & f,
                        const Function2d_x & f_x,
                        const Function2d_y & f_y,
                        const VectorX<Scalar> & weights) const;
    };

    template <typename Scalar, typename Impl>
    template <int QuadStrength, typename Function2d>
    Scalar Basis<Scalar, Impl>::error_l2(const Function2d & f, const VectorX<Scalar> & weights) const
    {
        return static_cast<const Impl *>(this)->template error_l2<QuadStrength>(f, weights);
    };

    template <typename Scalar, typename Impl>
    template <int QuadStrength, typename Function2d_x, typename Function2d_y>
    Scalar Basis<Scalar, Impl>::error_h1_semi(const Function2d_x & f_x,
                                              const Function2d_y & f_y,
                                              const VectorX<Scalar> & weights) const
    {
        return static_cast<const Impl *>(this)->template error_h1_semi<QuadStrength>(f_x, f_y, weights);
    };

    template <typename Scalar, typename Impl>
    template <int QuadStrength, typename Function2d, typename Function2d_x, typename Function2d_y>
    Scalar Basis<Scalar, Impl>::error_h1(const Function2d & f,
                                         const Function2d_x & f_x,
                                         const Function2d_y & f_y,
                                         const VectorX<Scalar> & weights) const
    {
        const auto l2 = error_l2<QuadStrength>(f, weights);
        const auto h1_semi = error_h1_semi<QuadStrength>(f_x, f_y, weights);
        return std::sqrt(l2 * l2 + h1_semi * h1_semi);
    };
}
