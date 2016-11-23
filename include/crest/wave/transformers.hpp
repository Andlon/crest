#pragma once

#include <crest/basis/basis.hpp>
#include <crest/util/eigen_extensions.hpp>

#include <cstdint>


namespace crest
{
    namespace wave
    {
        /**
         * A ResultTransformer is responsible for taking the solution u(t) at any time t and transforming it into
         * something else. For example, it is ideal for computing errors of solutions without storing the full
         * solution for every time step t (which would be very memory-intensive). Another example is "compressing"
         * the solution by interpolating it in a low resolution mesh for visualization.
         */
        template < typename Scalar, typename TransformedResult>
        class ResultTransformer
        {
        public:
            /*
             * Transforms the solution at time t and given sample index `x` into
             * a TransformedResult.
             */
            virtual TransformedResult transform(uint64_t index, Scalar t, const VectorX<Scalar> & x) const = 0;
        };

        template <typename Scalar>
        struct ErrorSample
        {
            Scalar l2;
            Scalar h1;
            Scalar h1_semi;
        };

        template <int QuadStrength, typename Scalar, typename BasisImpl,
                typename Function2dt, typename Function2dt_x, typename Function2dt_y>
        class ErrorTransformer : public ResultTransformer<Scalar, ErrorSample<Scalar>>
        {
        public:
            explicit ErrorTransformer(const crest::Basis<Scalar, BasisImpl> & basis,
                                      const Function2dt & u,
                                      const Function2dt_x & u_x,
                                      const Function2dt_y & u_y)
                    : _basis(basis), _u(u), _u_x(u_x), _u_y(u_y) {}

            virtual ErrorSample<Scalar> transform(uint64_t i, Scalar t, const VectorX<Scalar> & weights) const override
            {
                (void) i;

                // u, u_x and u_y are time-dependent functions, but here we fix the time t.
                const auto u_i = [this, t] (auto x, auto y) { return _u(t, x, y); };
                const auto u_x_i = [this, t] (auto x, auto y) { return _u_x(t, x, y); };
                const auto u_y_i = [this, t] (auto x, auto y) { return _u_y(t, x, y); };

                ErrorSample<Scalar> errors;
                errors.l2 = _basis.template error_l2<QuadStrength>(u_i, weights);
                errors.h1_semi = _basis.template error_h1_semi<QuadStrength>(u_x_i, u_y_i, weights);
                errors.h1 = std::sqrt(errors.l2 * errors.l2 + errors.h1_semi * errors.h1_semi);
                return errors;
            }

        private:
            const crest::Basis<Scalar, BasisImpl> & _basis;
            const Function2dt & _u;
            const Function2dt_x & _u_x;
            const Function2dt_y & _u_y;
        };

        template <int QuadStrength, typename Scalar, typename BasisImpl,
                typename Function2dt, typename Function2dt_x, typename Function2dt_y>
        ErrorTransformer<QuadStrength, Scalar, BasisImpl, Function2dt, Function2dt_x, Function2dt_y>
        make_error_transformer(const crest::Basis<Scalar, BasisImpl> & basis,
                               const Function2dt & u,
                               const Function2dt_x & u_x,
                               const Function2dt_y & u_y)
        {
            return ErrorTransformer<QuadStrength, Scalar, BasisImpl, Function2dt, Function2dt_x, Function2dt_y>
                    (basis, u, u_x, u_y);
        };
    }
}
