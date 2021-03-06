#pragma once

#include <crest/util/eigen_extensions.hpp>

#include <Eigen/Sparse>

#include <stdexcept>

namespace crest
{
    namespace wave
    {
        class ConvergenceError : std::runtime_error {
        public:
            explicit ConvergenceError(const std::string & what) : std::runtime_error(what) {}
        };

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

            virtual VectorX<Scalar> initialize(const InitialConditions<Scalar> & initial_conditions,
                                               Scalar dt) const = 0;
        };

        template <typename Scalar>
        class Integrator
        {
        public:
            virtual ~Integrator() {}

            /**
             * Tolerance for iterative solver if the integrator uses an iterative solver, otherwise ignored.
             */
            void set_tolerance(Scalar tol) { _tol = tol; }
            Scalar tolerance() const { return _tol; }

            virtual void setup(Scalar dt, Eigen::SparseMatrix<Scalar> stiffness, Eigen::SparseMatrix<Scalar> mass) = 0;

            virtual VectorX<Scalar> next(int step_index,
                                         Scalar dt,
                                         const VectorX<Scalar> & current_solution,
                                         const VectorX<Scalar> & previous_solution,
                                         const VectorX<Scalar> & load_next,
                                         const VectorX<Scalar> & load_current,
                                         const VectorX<Scalar> & load_previous) = 0;

        protected:
            Integrator() : _tol(Scalar(10) * std::numeric_limits<Scalar>::epsilon()) {}

        private:
            Scalar _tol;
        };

        /**
         * Computes u_h^1 = u_h^0 + dt * u_h_t^0 + 0.5 * dt^2 u_h_tt^0
         */
        template <typename Scalar>
        class SeriesExpansionInitializer : public Initializer<Scalar>
        {
        public:
            explicit SeriesExpansionInitializer() {}

            virtual VectorX<Scalar> initialize(const InitialConditions<Scalar> & initial_conditions,
                                               Scalar dt) const override
            {
                const auto & ic = initial_conditions;
                return ic.u0_h + dt * ic.v0_h + 0.5 * dt * dt * ic.u0_tt_h;
            }
        };

        template <typename Scalar>
        class DirectCrankNicolson : public Integrator<Scalar>
        {
        public:
            explicit DirectCrankNicolson() {}

            virtual void setup(Scalar dt, Eigen::SparseMatrix<Scalar> stiffness, Eigen::SparseMatrix<Scalar> mass)
            {
                _mass = std::move(mass);
                _stiffness = std::move(stiffness);

                const auto C = mass + Scalar(0.25) * dt * dt * stiffness;
                _c_factor.compute(C);
            }

            virtual VectorX<Scalar> next(int step_index,
                                         Scalar dt,
                                         const VectorX<Scalar> & x_curr,
                                         const VectorX<Scalar> & x_prev,
                                         const VectorX<Scalar> & b_prev,
                                         const VectorX<Scalar> & b_curr,
                                         const VectorX<Scalar> & b_next)
            {
                (void) step_index;

                const auto & M = _mass;
                const auto & A = _stiffness;
                const auto dt2 = dt * dt;

                const auto rhs = M * (Scalar(2.0) * x_curr - x_prev)
                                 - 0.25 * dt2 * A * (Scalar(2.0) * x_curr + x_prev)
                                 + 0.25 * dt2 * (b_next + Scalar(2.0) * b_curr + b_prev);

                const VectorX<Scalar> x_next = _c_factor.solve(rhs);

                if (_c_factor.info() != Eigen::Success)
                {
                    throw ConvergenceError("Integration did not converge.");
                }
                return x_next;
            }

        private:
            Eigen::SparseMatrix<Scalar> _stiffness;
            Eigen::SparseMatrix<Scalar> _mass;

            typedef Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> Factorization;
            Factorization _c_factor;
        };

        template <typename Scalar>
        class IterativeCrankNicolson : public Integrator<Scalar>
        {
        public:
            explicit IterativeCrankNicolson() {}

            virtual void setup(Scalar dt, Eigen::SparseMatrix<Scalar> stiffness, Eigen::SparseMatrix<Scalar> mass)
            {
                _mass = std::move(mass);
                _stiffness = std::move(stiffness);

                const auto C = mass + Scalar(0.25) * dt * dt * stiffness;

                // Eigen's Conjugate gradient applies diagonal preconditioning by default
                _cg.compute(C);
                _cg.setTolerance(Integrator<Scalar>::tolerance());
            }

            virtual VectorX<Scalar> next(int step_index,
                                         Scalar dt,
                                         const VectorX<Scalar> & x_curr,
                                         const VectorX<Scalar> & x_prev,
                                         const VectorX<Scalar> & b_prev,
                                         const VectorX<Scalar> & b_curr,
                                         const VectorX<Scalar> & b_next)
            {
                (void) step_index;

                const auto & M = _mass;
                const auto & A = _stiffness;
                const auto dt2 = dt * dt;

                const auto rhs = M * (Scalar(2.0) * x_curr - x_prev)
                                 - 0.25 * dt2 * A * (Scalar(2.0) * x_curr + x_prev)
                                 + 0.25 * dt2 * (b_next + Scalar(2.0) * b_curr + b_prev);

                // Use x_curr as starting guess for CG
                const VectorX<Scalar> x_next = _cg.solveWithGuess(rhs, x_curr);

                if (_cg.info() != Eigen::Success)
                {
                    throw ConvergenceError("Integration did not converge.");
                }
                return x_next;
            }

        private:
            Eigen::SparseMatrix<Scalar> _stiffness;
            Eigen::SparseMatrix<Scalar> _mass;

            Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar>> _cg;
        };

        template <typename Scalar>
        class IterativeLeapfrog : public Integrator<Scalar>
        {
        public:
            explicit IterativeLeapfrog() {}

            virtual void setup(Scalar dt, Eigen::SparseMatrix<Scalar> stiffness, Eigen::SparseMatrix<Scalar> mass)
            {
                (void) dt;

                _mass = std::move(mass);
                _stiffness = std::move(stiffness);

                // Eigen's Conjugate gradient applies diagonal preconditioning by default
                _cg.compute(_mass);
                _cg.setTolerance(Integrator<Scalar>::tolerance());
            }

            virtual VectorX<Scalar> next(int step_index,
                                         Scalar dt,
                                         const VectorX<Scalar> & x_curr,
                                         const VectorX<Scalar> & x_prev,
                                         const VectorX<Scalar> & b_prev,
                                         const VectorX<Scalar> & b_curr,
                                         const VectorX<Scalar> & b_next)
            {
                (void) step_index;
                (void) b_prev;
                (void) b_next;

                const auto & M = _mass;
                const auto & A = _stiffness;
                const auto dt2 = dt * dt;

                const auto rhs = M * (Scalar(2.0) * x_curr - x_prev)
                                 + dt2 * (b_curr - A * x_curr);

                // Use x_curr as starting guess for CG
                const VectorX<Scalar> x_next = _cg.solveWithGuess(rhs, x_curr);

                if (_cg.info() != Eigen::Success)
                {
                    throw ConvergenceError("Integration did not converge.");
                }
                return x_next;
            }

        private:
            Eigen::SparseMatrix<Scalar> _stiffness;
            Eigen::SparseMatrix<Scalar> _mass;

            Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar>> _cg;
        };

        namespace detail
        {
            template <typename Scalar>
            Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> lump_matrix(const Eigen::SparseMatrix<Scalar> & mat)
            {
                VectorX<Scalar> row_wise_sum = mat * VectorX<Scalar>::Ones(mat.cols());
                return Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic>(std::move(row_wise_sum));
            };
        }

        template <typename Scalar>
        class LumpedLeapfrog : public Integrator<Scalar>
        {
        public:
            explicit LumpedLeapfrog() {}

            virtual void setup(Scalar dt,
                               Eigen::SparseMatrix<Scalar> stiffness,
                               Eigen::SparseMatrix<Scalar> mass) override
            {
                (void) dt;

                _diagonal_mass = detail::lump_matrix(mass);
                _stiffness = std::move(stiffness);
            }

            virtual VectorX<Scalar> next(int step_index,
                                         Scalar dt,
                                         const VectorX<Scalar> & x_curr,
                                         const VectorX<Scalar> & x_prev,
                                         const VectorX<Scalar> & b_prev,
                                         const VectorX<Scalar> & b_curr,
                                         const VectorX<Scalar> & b_next) override
            {
                (void) step_index;
                (void) b_prev;
                (void) b_next;

                const auto & M = _diagonal_mass;
                const auto & A = _stiffness;
                const auto dt2 = dt * dt;

                const auto rhs = M * (Scalar(2.0) * x_curr - x_prev)
                                 + dt2 * (b_curr - A * x_curr);

                // Inverting M is just a diagonal matrix inversion
                const VectorX<Scalar> x_next = M.inverse() * rhs;

                return x_next;
            }

        private:
            Eigen::SparseMatrix<Scalar> _stiffness;
            Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> _diagonal_mass;
        };


    }
}
