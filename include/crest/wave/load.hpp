#pragma once

#include <crest/basis/basis.hpp>

namespace crest
{
    namespace wave
    {
        template <typename Scalar>
        class LoadProvider
        {
        public:
            virtual ~LoadProvider() {}

            /**
             * Compute the load vector at time t.
             * @param t
             * @return
             */
            virtual VectorX<Scalar> compute(Scalar t) const = 0;
        };

        template <int QuadStrength, typename Scalar, typename TemporalFunction2d, typename BasisImpl>
        class BasisLoadProvider : public LoadProvider<Scalar>
        {
        public:
            explicit BasisLoadProvider(const TemporalFunction2d & f,
                                       const Basis<Scalar, BasisImpl> & basis)
                    :   _basis(basis), _f(f)
            {}

            virtual VectorX<Scalar> compute(Scalar t) const override
            {
                return _basis.template load<QuadStrength>([this, t] (auto x, auto y) { return _f(t, x, y); });
            }

        private:
            const Basis<Scalar, BasisImpl> & _basis;
            const TemporalFunction2d & _f;
        };

        template <int QuadStrength, typename Scalar, typename TemporalFunction2d, typename BasisImpl>
        BasisLoadProvider<QuadStrength, Scalar, TemporalFunction2d, BasisImpl>
        make_basis_load_function(const TemporalFunction2d & f,
                                 const Basis<Scalar, BasisImpl> & basis)
        {
            return BasisLoadProvider<QuadStrength, Scalar, TemporalFunction2d, BasisImpl>(f, basis);
        };
    }
}


