#pragma once

#include <crest/util/eigen_extensions.hpp>
#include <crest/basis/basis.hpp>
#include <crest/wave/constraints.hpp>
#include <crest/wave/integrators.hpp>
#include <crest/wave/transformers.hpp>
#include <crest/wave/load.hpp>

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
                    const ConstrainedSystem<Scalar> & system,
                    const InitialConditions<Scalar> & full)
            {
                InitialConditions<Scalar> constrained;
                constrained.u0_h = system.constrain_solution(full.u0_h);
                constrained.v0_h = system.constrain_velocity(full.v0_h);
                constrained.u0_tt_h = system.constrain_acceleration(full.u0_tt_h);
                return constrained;
            }
        }



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

        template <typename Scalar, typename TransformedResult>
        SolveResult<Scalar, TransformedResult>
        solve(const ConstrainedSystem<Scalar> & system,
              const InitialConditions<Scalar> & initial_conditions,
              Integrator<Scalar> & integrator,
              const Initializer<Scalar> & initializer,
              const Parameters<Scalar> & parameters,
              const ResultTransformer<Scalar, TransformedResult> & transformer)
        {
            if (parameters.num_samples == 0) {
                throw std::invalid_argument("Number of samples must be greater or equal to 1.");
            }

            const auto constrained_ic = detail::constrain_initial_conditions(system, initial_conditions);

            const auto dt = parameters.dt;
            integrator.setup(dt, system.stiffness(), system.mass());

            VectorX<Scalar> x_prev = constrained_ic.u0_h;
            VectorX<Scalar> x_current = initializer.initialize(constrained_ic, dt);
            VectorX<Scalar> x_next = VectorX<Scalar>(system.num_free_nodes());

            VectorX<Scalar> load_prev = system.load(Scalar(0));
            VectorX<Scalar> load_current = system.load(dt);
            VectorX<Scalar> load_next = system.load(Scalar(2) * dt);

            SolveResult<Scalar, TransformedResult> sol;

            sol.result.push_back(transformer.transform(0u, Scalar(0), system.expand_solution(Scalar(0), x_prev)));

            if (parameters.num_samples > 1) {
                sol.result.push_back(transformer.transform(1u, dt,  system.expand_solution(Scalar(dt), x_current)));
            }

            for (uint64_t i = 2; i < parameters.num_samples; ++i)
            {
                const auto t = Scalar(i) * dt;
                load_next = system.load(Scalar(i) * dt);
                x_next = integrator.next(i, dt, x_current, x_prev, load_next, load_current, load_prev);

                sol.result.push_back(transformer.transform(i,
                                                           t,
                                                           system.expand_solution(t, x_next)));

                // TODO: Replace this with Eigen's swap for efficiency? For now keep std::swap until we
                // have confirmed working code

                // Shuffle results of previous computations, such that x_next and load_next immediately
                // get overwritten upon the next iteration
                std::swap(x_prev, x_current);
                std::swap(x_current, x_next);
                std::swap(load_prev, load_current);
                std::swap(load_current, load_next);
            }

            if (sol.result.size() != parameters.num_samples) {
                throw std::logic_error("Internal error: result size is not equal to number of samples.");
            }

            return sol;
        }
    }
}
