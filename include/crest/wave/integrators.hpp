#pragma once

#include <crest/util/eigen_extensions.hpp>

#include <Eigen/Sparse>

#include <stdexcept>

namespace crest
{
    namespace wave
    {
        template <typename Scalar>
        struct InitialConditions
        {
            VectorX<Scalar> u0_h;
            VectorX<Scalar> v0_h;
            VectorX<Scalar> u0_tt_h;
        };

        template <typename Scalar>
        class Initializer
        {
        public:
            virtual ~Initializer() {}

            virtual VectorX<Scalar> initialize(const InitialConditions<Scalar> & initial_conditions) const = 0;
        };

        template <typename Scalar>
        class Integrator
        {
        public:
            virtual ~Integrator() {}

            virtual void setup(Eigen::SparseMatrix<Scalar> stiffness, Eigen::SparseMatrix<Scalar> mass) = 0;

            virtual VectorX<Scalar> next(int step_index,
                                         const VectorX<Scalar> & current_solution,
                                         const VectorX<Scalar> & previous_solution,
                                         const VectorX<Scalar> & load_next,
                                         const VectorX<Scalar> & load_current,
                                         const VectorX<Scalar> & load_previous) = 0;

            virtual Scalar timestep() const = 0;
        };

        /**
         * Computes u_h^1 = u_h^0 + dt * u_h_t^0 + 0.5 * dt^2 u_h_tt^0
         */
        template <typename Scalar>
        class SeriesExpansionInitializer : public Initializer<Scalar>
        {
        public:
            explicit SeriesExpansionInitializer(Scalar timestep) : _dt(timestep) {}

            virtual VectorX<Scalar> initialize(const InitialConditions<Scalar> & initial_conditions) const
            {
                const auto & ic = initial_conditions;
                return ic.u0_h + _dt * ic.v0_h + 0.5 * _dt * _dt * ic.u0_tt_h;
            }

        private:
            Scalar _dt;
        };

        template <typename Scalar>
        class ConstantTimestepIntegrator : public Integrator<Scalar>
        {
        public:
            explicit ConstantTimestepIntegrator(Scalar time_step) : _dt(time_step) {
                if (time_step <= Scalar(0)) {
                    throw std::invalid_argument("Time step must be a positive number.");
                }
            }

            virtual Scalar timestep() const { return _dt; }

        private:
            Scalar _dt;
        };

        template <typename Scalar>
        class CrankNicolson : public ConstantTimestepIntegrator<Scalar>
        {
        public:
            explicit CrankNicolson(Scalar time_step)
                    : ConstantTimestepIntegrator<Scalar>(time_step)
            {}

            virtual void setup(Eigen::SparseMatrix<Scalar> stiffness, Eigen::SparseMatrix<Scalar> mass)
            {
                const auto dt = this->timestep();
                _mass = std::move(mass);
                _stiffness = std::move(stiffness);

                const auto C = mass + Scalar(0.25) * dt * dt * stiffness;
                _c_factor.compute(C);
            }

            virtual VectorX<Scalar> next(int step_index,
                                         const VectorX<Scalar> & x_curr,
                                         const VectorX<Scalar> & x_prev,
                                         const VectorX<Scalar> & b_prev,
                                         const VectorX<Scalar> & b_curr,
                                         const VectorX<Scalar> & b_next)
            {
                (void) step_index;

                const auto & M = _mass;
                const auto & A = _stiffness;
                const auto dt = this->timestep();
                const auto dt2 = dt * dt;

                const auto rhs = M * (Scalar(2.0) * x_curr - x_prev)
                                 - 0.25 * dt2 * A * (Scalar(2.0) * x_curr + x_prev)
                                 + 0.25 * dt2 * (b_next + Scalar(2.0) * b_curr + b_prev);

                const VectorX<Scalar> x_next = _c_factor.solve(rhs);

                if (_c_factor.info() != Eigen::Success)
                {
                    // TODO Throw appropriate exception?
                    throw new std::logic_error("Did not converge.");
                }
                return x_next;
            }

        private:
            Eigen::SparseMatrix<Scalar> _stiffness;
            Eigen::SparseMatrix<Scalar> _mass;

            typedef Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> Factorization;
            Factorization _c_factor;
        };


    }
}
