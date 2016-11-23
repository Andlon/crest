#pragma once

#include <crest/util/eigen_extensions.hpp>
#include <crest/basis/basis.hpp>
#include <crest/wave/constraints.hpp>
#include <crest/wave/integrators.hpp>
#include <crest/wave/transformers.hpp>

#include <utility>

#include <Eigen/Sparse>

namespace crest
{
    namespace wave
    {
        namespace detail
        {
            template <typename Scalar>
            InitialConditions<Scalar> constrain_initial_conditions(
                    const ConstraintHandler<Scalar> & constraint_handler,
                    const InitialConditions<Scalar> & full)
            {
                InitialConditions<Scalar> constrained;
                constrained.u0_h = constraint_handler.constrain_solution(full.u0_h);
                constrained.v0_h = constraint_handler.constrain_velocity(full.v0_h);
                constrained.u0_tt_h = constraint_handler.constrain_acceleration(full.u0_tt_h);
                return constrained;
            }
        }

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

        template <typename Scalar, typename TransformedResult>
        struct SolveResult
        {
            std::vector<TransformedResult> result;
        };

        template <typename Scalar>
        struct Parameters {
            Scalar dt;
            uint64_t num_samples;
        };

        template <typename Scalar, typename BasisImpl, typename TransformedResult>
        SolveResult<Scalar, TransformedResult>
        solve(const Basis<Scalar, BasisImpl> & basis,
              const InitialConditions<Scalar> & initial_conditions,
              const LoadProvider<Scalar> & load_provider,
              const ConstraintHandler<Scalar> & constraint_handler,
              Integrator<Scalar> & integrator,
              const Initializer<Scalar> & initializer,
              const Parameters<Scalar> & parameters,
              const ResultTransformer<Scalar, TransformedResult> & transformer)
        {
            const auto assembly = basis.assemble();
            const auto constrained_stiffness = constraint_handler.constrain_system_matrix(assembly.stiffness);
            const auto constrained_mass = constraint_handler.constrain_system_matrix(assembly.mass);
            const auto constrained_ic = detail::constrain_initial_conditions(constraint_handler, initial_conditions);

            const auto dt = parameters.dt;
            integrator.setup(dt, std::move(constrained_stiffness), std::move(constrained_mass));

            VectorX<Scalar> x_prev = constrained_ic.u0_h;
            VectorX<Scalar> x_current = initializer.initialize(constrained_ic, dt);
            VectorX<Scalar> x_next = VectorX<Scalar>(constraint_handler.num_free_nodes());

            VectorX<Scalar> load_prev = constraint_handler.constrain_load(load_provider.compute(Scalar(0)));
            VectorX<Scalar> load_current = constraint_handler.constrain_load(load_provider.compute(dt));
            VectorX<Scalar> load_next = constraint_handler.constrain_load(load_provider.compute(Scalar(2) * dt));

            SolveResult<Scalar, TransformedResult> sol;

            sol.result.push_back(transformer.transform(0u, 0.0, constraint_handler.expand_solution(x_prev)));
            sol.result.push_back(transformer.transform(1u, dt,  constraint_handler.expand_solution(x_current)));

            // TODO: Handle num_samples == 0 or 1?
            for (uint64_t i = 2; i < parameters.num_samples; ++i)
            {
                load_next = constraint_handler.constrain_load(load_provider.compute(Scalar(i) * dt));
                x_next = integrator.next(i, dt, x_current, x_prev, load_next, load_current, load_prev);

                sol.result.push_back(transformer.transform(i,
                                                             Scalar(i) * dt,
                                                             constraint_handler.expand_solution(x_next)));

                // TODO: Replace this with Eigen's swap for efficiency? For now keep std::swap until we
                // have confirmed working code

                // Shuffle results of previous computations, such that x_next and load_next immediately
                // get overwritten upon the next iteration
                std::swap(x_prev, x_current);
                std::swap(x_current, x_next);
                std::swap(load_prev, load_current);
                std::swap(load_current, load_next);
            }

            return sol;
        }
    }
}
