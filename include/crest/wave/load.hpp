#pragma once

#include <crest/basis/basis.hpp>

#include <memory>

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
        make_basis_load_provider(const TemporalFunction2d & f,
                                 const Basis<Scalar, BasisImpl> & basis)
        {
            return BasisLoadProvider<QuadStrength, Scalar, TemporalFunction2d, BasisImpl>(f, basis);
        };

        /**
         * Produces a pointer to a load provider which delegates to
         * @param f
         * @param basis
         * @param quadrature_strength
         * @return
         */
        template <typename Scalar, typename TemporalFunction2d, typename BasisImpl>
        std::unique_ptr<LoadProvider<Scalar>>
        make_dynamic_basis_load_provider(const TemporalFunction2d & f,
                                         const Basis<Scalar, BasisImpl> & basis,
                                         int quadrature_strength)
        {
            switch (quadrature_strength)
            {
                case 1:
                    return std::make_unique<BasisLoadProvider<1, Scalar, TemporalFunction2d, BasisImpl>>(f, basis);
                case 2:
                    return std::make_unique<BasisLoadProvider<2, Scalar, TemporalFunction2d, BasisImpl>>(f, basis);
                case 3:
                    // No quadrature points available for strength 3, so we use strength 4 instead
                    return std::make_unique<BasisLoadProvider<4, Scalar, TemporalFunction2d, BasisImpl>>(f, basis);
                case 4:
                    return std::make_unique<BasisLoadProvider<4, Scalar, TemporalFunction2d, BasisImpl>>(f, basis);
                case 5:
                    return std::make_unique<BasisLoadProvider<5, Scalar, TemporalFunction2d, BasisImpl>>(f, basis);
                case 6:
                    return std::make_unique<BasisLoadProvider<6, Scalar, TemporalFunction2d, BasisImpl>>(f, basis);

                // TODO: Higher quadrature strengths
                default:
                    throw std::runtime_error("Unable to construct load provider with given quadrature strength.");
            }
        };
    }
}


