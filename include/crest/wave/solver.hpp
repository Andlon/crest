#pragma once

#include <crest/util/timer.hpp>
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

        struct SolveTiming {
            double assembly_time;
            double load_time;
            double initializer_time;
            double integrator_setup_time;
            double integrator_solve_time;
            double transform_time;
            double total_time;

            SolveTiming() : assembly_time(NAN),
                            load_time(0.0),
                            initializer_time(0.0),
                            integrator_setup_time(0.0),
                            integrator_solve_time(0.0),
                            transform_time(0.0),
                            total_time(0.0) {}
        };

        template <typename Scalar, typename TransformedResult>
        struct SolveResult
        {
            std::vector<TransformedResult> result;
            SolveTiming timing;
            bool converged;
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
            SolveResult<Scalar, TransformedResult> sol;
            SolveTiming timing;
            Timer total_timer;

            try {
                if (parameters.num_samples == 0) {
                    throw std::invalid_argument("Number of samples must be greater or equal to 1.");
                }

                // TODO: Rewrite constrained system so that it does not take ownership of system matrices?
                const auto constrained_ic = detail::constrain_initial_conditions(system, initial_conditions);

                const auto dt = parameters.dt;
                inspect_timing(timing.integrator_setup_time, [&] {
                    integrator.setup(dt, system.assembly().stiffness, system.assembly().mass);
                });

                VectorX<Scalar> x_prev = constrained_ic.u0_h;
                VectorX<Scalar> x_current = inspect_timing(timing.initializer_time, [&] {
                    return initializer.initialize(constrained_ic, dt);
                });
                VectorX<Scalar> x_next = VectorX<Scalar>(system.num_free_nodes());

                VectorX<Scalar> load_prev = inspect_timing(timing.load_time, [&] {
                    return system.load(Scalar(0));
                });
                VectorX<Scalar> load_current = inspect_timing(timing.load_time, [&] {
                    return system.load(dt);
                });
                VectorX<Scalar> load_next = inspect_timing(timing.load_time, [&] {
                    return system.load(Scalar(2) * dt);
                });

                {
                    const auto expanded = system.expand_solution(Scalar(0), x_prev);
                    const auto transformed = inspect_timing(timing.transform_time, [&] {
                        return transformer.transform(0u, Scalar(0), expanded);
                    });
                    sol.result.emplace_back(std::move(transformed));
                }


                if (parameters.num_samples > 1) {
                    const auto expanded = system.expand_solution(Scalar(dt), x_current);
                    const auto transformed = inspect_timing(timing.transform_time, [&] {
                        return transformer.transform(1u, dt, expanded);
                    });
                    sol.result.emplace_back(std::move(transformed));
                }

                for (uint64_t i = 2; i < parameters.num_samples; ++i)
                {
                    const auto t = Scalar(i) * dt;
                    load_next = inspect_timing(timing.load_time, [&] {
                        return system.load(Scalar(i) * dt);
                    });
                    x_next = inspect_timing(timing.integrator_solve_time, [&] {
                        return integrator.next(i, dt, x_current, x_prev, load_next, load_current, load_prev);
                    });

                    {
                        const auto expanded = system.expand_solution(t, x_next);
                        const auto transformed = inspect_timing(timing.transform_time, [&] {
                            return transformer.transform(i, t, expanded);
                        });
                        sol.result.emplace_back(std::move(transformed));
                    }


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

                sol.converged = true;
            } catch (const ConvergenceError & e) {
                sol.converged = false;
            }

            timing.total_time = total_timer.elapsed();
            sol.timing = timing;

            return sol;
        }
    }
}
