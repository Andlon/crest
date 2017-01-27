#pragma once

#include <crest/basis/homogenized_basis.hpp>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <memory>

namespace crest
{
    namespace detail
    {

        template <
                typename _Scalar,
                typename StiffnessSolver,
                typename StiffnessMatrixType = Eigen::SparseMatrix<_Scalar>,
                typename InterpolatorMatrixType = Eigen::SparseMatrix<_Scalar> >
        class SchurComplement
                : public Eigen::EigenBase<SchurComplement<_Scalar, StiffnessSolver, InterpolatorMatrixType>>
        {
        public:
            typedef _Scalar Scalar;
            typedef Scalar RealScalar;
            typedef int StorageIndex;

            enum
            {
                ColsAtCompileTime = Eigen::Dynamic,
                MaxColsAtCompileTime = Eigen::Dynamic,
                IsRowMajor = false
            };

            SchurComplement() : _stiffness_solver(nullptr), _A(nullptr), _I_H(nullptr) {}

            Eigen::Index rows() const { return _I_H->rows(); }
            Eigen::Index cols() const { return _I_H->rows(); }

            template <typename Rhs>
            Eigen::Product<SchurComplement<Scalar, StiffnessSolver>, Rhs, Eigen::AliasFreeProduct>
            operator*(const Eigen::MatrixBase<Rhs> & x) const
            {
                return Eigen::Product<
                        SchurComplement<Scalar, StiffnessSolver>,
                        Rhs,
                        Eigen::AliasFreeProduct>(*this, x.derived());
            }

            const InterpolatorMatrixType &  quasiInterpolator() const { return *_I_H; }
            const StiffnessMatrixType    &  stiffnessMatrix() const { return *_A; }
            const StiffnessSolver        &  stiffnessSolver() const { return *_stiffness_solver; }

            SchurComplement & withQuasiInterpolator(const InterpolatorMatrixType & I_H)
            {
                _I_H = &I_H;
                return *this;
            }

            SchurComplement & withStiffness(const StiffnessMatrixType & stiffness_matrix,
                                            const StiffnessSolver & stiffness_solver)
            {
                _A = &stiffness_matrix;
                _stiffness_solver = &stiffness_solver;
                return *this;
            }

            bool isInitialized() const
            {
                return _stiffness_solver != nullptr && _I_H != nullptr && _A != nullptr;
            }

        private:
            const StiffnessSolver           * _stiffness_solver;
            const StiffnessMatrixType       * _A;
            const InterpolatorMatrixType    * _I_H;
        };

        /**
         * Preconditioner for the Schur complement problem S y = b,
         * where S = I_H A^-1 I_H^T,
         * where I_H is a quasi-interpolation operator and A is an SPD stiffness matrix.
         *
         * The preconditioner is defined by the approximation
         * S_diag = I_H diag(A)^{-1} I_H^T.
         */
        template <typename Scalar>
        class DiagonalizedSchurComplementPreconditioner
        {
        public:
            DiagonalizedSchurComplementPreconditioner() {}

            template<typename MatrixType>
            DiagonalizedSchurComplementPreconditioner& analyzePattern(const MatrixType& ) { return *this; }

            template <
                    typename StiffnessSolver,
                    typename StiffnessMatrixType,
                    typename InterpolatorMatrixType>
            DiagonalizedSchurComplementPreconditioner& compute(
                    const SchurComplement<
                            Scalar,
                            StiffnessSolver,
                            StiffnessMatrixType,
                            InterpolatorMatrixType> & schur_complement)
            {
                _approx_stiffness_solver.compute(schur_complement.stiffnessMatrix());
                _diagonalized_schur.withStiffness(schur_complement.stiffnessMatrix(), _approx_stiffness_solver);
                _diagonalized_schur.withQuasiInterpolator(schur_complement.quasiInterpolator());
                _outer_solver.compute(_diagonalized_schur);
                return *this;
            }

            template<typename MatrixType>
            DiagonalizedSchurComplementPreconditioner& factorize(const MatrixType & mat) { return *this; }

            // Note: this copies the output, so it's not currently optimal. Can maybe use the Solve<> struct
            // instead? TODO
            template<typename Rhs>
            inline const Rhs solve(const Rhs& b) const
            {
                return _outer_solver.solve(b);
            }

            Eigen::ComputationInfo info() { return Eigen::Success; }

        private:
            Eigen::DiagonalPreconditioner<Scalar> _approx_stiffness_solver;
            SchurComplement<Scalar, Eigen::DiagonalPreconditioner<Scalar>> _diagonalized_schur;
            Eigen::ConjugateGradient<
                    SchurComplement<Scalar, Eigen::DiagonalPreconditioner<Scalar>>,
                    Eigen::Lower|Eigen::Upper,
                    Eigen::IdentityPreconditioner> _outer_solver;
        };

        template <typename Scalar>
        class SchurComplementCoarseStiffnessPreconditioner
        {
        public:
            SchurComplementCoarseStiffnessPreconditioner() {}

            template<typename MatrixType>
            SchurComplementCoarseStiffnessPreconditioner& analyzePattern(const MatrixType& ) { return *this; }

            template<typename MatrixType>
            SchurComplementCoarseStiffnessPreconditioner& factorize(const MatrixType&) { return *this; }

            template<typename MatrixType>
            SchurComplementCoarseStiffnessPreconditioner& compute(const MatrixType&) { return *this; }

            // Note: this copies the output, so it's not currently optimal. Can maybe use the Solve<> struct
            // instead? TODO
            template<typename Rhs>
            inline const Rhs solve(const Rhs& b) const
            {
                return _A_H * b;
            }

            Eigen::ComputationInfo info() { return Eigen::Success; }

            template <typename MatrixType>
            void setCoarseStiffnessMatrix(const MatrixType & A)
            {
                _A_H = A;
            }

        private:
            Eigen::SparseMatrix<Scalar> _A_H;
        };
    }

    template <typename Scalar>
    class SchurCorrectorSolver : public CorrectorSolver<Scalar>
    {
    public:
        virtual std::vector<Eigen::Triplet<Scalar>> resolve_element_correctors_for_patch(
                const BiscaleMesh<Scalar, int> & mesh,
                const std::vector<int> & coarse_patch_interior,
                const std::vector<int> & fine_patch_interior,
                const Eigen::SparseMatrix<Scalar> & global_coarse_stiffness,
                const Eigen::SparseMatrix<Scalar> & global_fine_stiffness,
                const Eigen::SparseMatrix<Scalar> & global_quasi_interpolator,
                int coarse_element) const override
        {
            const auto I_H = sparse_submatrix(global_quasi_interpolator,
                                              coarse_patch_interior,
                                              fine_patch_interior);
            const auto A_h = sparse_submatrix(global_fine_stiffness,
                                              fine_patch_interior,
                                              fine_patch_interior);
            const auto A_H = sparse_submatrix(global_coarse_stiffness,
                                              coarse_patch_interior,
                                              coarse_patch_interior);

            assert(A_h.rows() > 0);

            using Eigen::SparseMatrix;
            using Eigen::ConjugateGradient;
            using Eigen::SimplicialLDLT;
            using detail::SchurComplementCoarseStiffnessPreconditioner;

            // Define the type of the solver used to solve the "stiffness problem" Ax = b.
            typedef SimplicialLDLT<SparseMatrix<Scalar>> StiffnessSolver;

            // The SchurComplement uses the StiffnessSolver internally
            typedef detail::SchurComplement<Scalar, StiffnessSolver> SchurComplement;

            // Finally we define the type of solver used for the Schur complement
            typedef ConjugateGradient<
                    SchurComplement,
                    Eigen::Lower|Eigen::Upper,
                    SchurComplementCoarseStiffnessPreconditioner<Scalar>> SchurComplementSolver;

            std::vector<Eigen::Triplet<Scalar>> triplets;

            StiffnessSolver stiffness_solver(A_h);
            const auto schur_operator = SchurComplement()
                    .withQuasiInterpolator(I_H)
                    .withStiffness(A_h, stiffness_solver);
            SchurComplementSolver schur_solver;
            schur_solver.setTolerance(CorrectorSolver<Scalar>::iterative_tolerance());
            schur_solver.preconditioner().setCoarseStiffnessMatrix(A_H);
            schur_solver.compute(schur_operator);

            for (int i = 0; i < 3; ++i)
            {
                const auto b = this->local_rhs(mesh, fine_patch_interior, coarse_element, i);
                VectorX<Scalar> corrector(A_h.rows());

                if (I_H.rows() == 0)
                {
                    corrector = stiffness_solver.solve(b);
                } else
                {
                    const VectorX<Scalar> y = stiffness_solver.solve(b);
                    const VectorX<Scalar> kappa = schur_solver.solve(I_H * y);
                    corrector = stiffness_solver.solve(b - I_H.transpose() * kappa);
                }

                assert(static_cast<size_t>(corrector.rows()) == fine_patch_interior.size());
                const auto global_index = mesh.coarse_mesh().elements()[coarse_element].vertex_indices[i];
                for (size_t k = 0; k < fine_patch_interior.size(); ++k)
                {
                    const auto component = corrector(k);
                    // Due to rounding issues, some components that should perhaps be zero in exact arithmetic
                    // may be non-zero, and as such we might end up with a denser matrix than we should actually
                    // have. To prevent this, we introduce a threshold which determines whether to keep the entry.
                    // TODO: Make this threshold configurable?
                    if (std::abs(component) > 1e-12)
                    {
                        triplets.emplace_back(Eigen::Triplet<Scalar>(global_index, fine_patch_interior[k], component));
                    }
                }
            }

            return triplets;
        }
    };


}

/**
 * The code below is necessary for matrix-free iterative solvers in Eigen.
 * See https://eigen.tuxfamily.org/dox/group__MatrixfreeSolverExample.html
 */
namespace Eigen {
    namespace internal {
        // MatrixReplacement looks- a SparseMatrix, so let's inherits its traits:
        template <typename _Scalar, typename _StiffnessSolver>
        struct traits<crest::detail::SchurComplement<_Scalar, _StiffnessSolver>>
        {
            typedef _Scalar Scalar;
            typedef int StorageIndex;
            typedef Eigen::Sparse StorageKind;
            typedef Eigen::MatrixXpr XprKind;
            enum {
                RowsAtCompileTime = Dynamic,
                ColsAtCompileTime = Dynamic,
                MaxRowsAtCompileTime = Dynamic,
                MaxColsAtCompileTime = Dynamic,
                Flags = 0
            };
        };

        template<typename _Scalar, typename _StiffnessSolver, typename Rhs>
        struct generic_product_impl<crest::detail::SchurComplement<_Scalar, _StiffnessSolver>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
                : generic_product_impl_base<crest::detail::SchurComplement<_Scalar, _StiffnessSolver>,
                        Rhs,
                        generic_product_impl<crest::detail::SchurComplement<_Scalar, _StiffnessSolver>, Rhs> >
        {
            typedef typename Product<crest::detail::SchurComplement<_Scalar, _StiffnessSolver>, Rhs>::Scalar Scalar;
            template<typename Dest>
            static void scaleAndAddTo(Dest& dst,
                                      const crest::detail::SchurComplement<_Scalar, _StiffnessSolver>& lhs,
                                      const Rhs& rhs,
                                      const Scalar& alpha)
            {
                // Apply the Schur complement S = I_H A^-1 I_H^T
                // to the vector x.

                // First solve
                // Ay = I_H^T x
                // such that y = A^1 I_H^T x,
                // then multiply by I_H
                assert(lhs.isInitialized() && "Must initialize SchurComplement stiffness solver "
                        "and quasi interpolator before using.");

                const auto & A = lhs.stiffnessSolver();
                const auto & I_H = lhs.quasiInterpolator();
                dst = alpha * I_H * A.solve(I_H.transpose() * rhs);
            }
        };
    }
}
