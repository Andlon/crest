#pragma once

#include <crest/basis/homogenized_basis.hpp>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

namespace crest
{
    namespace detail
    {

        template <typename _Scalar, typename _StiffnessSolver>
        class SchurComplement : public Eigen::EigenBase<SchurComplement<_Scalar, _StiffnessSolver>>
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

            Eigen::Index rows() const { return _I_H.rows(); }

            Eigen::Index cols() const { return _I_H.rows(); }

            template <typename Rhs>
            Eigen::Product<SchurComplement<Scalar, _StiffnessSolver>, Rhs, Eigen::AliasFreeProduct>
            operator*(const Eigen::MatrixBase<Rhs> & x) const
            {
                return Eigen::Product<
                        SchurComplement<Scalar, _StiffnessSolver>,
                        Rhs,
                        Eigen::AliasFreeProduct>(*this, x.derived());
            }

            explicit SchurComplement(_StiffnessSolver & stiffness_solver, const Eigen::SparseMatrix<Scalar> & I_H)
                    : _stiffness_solver(stiffness_solver), _I_H(I_H)
            { }

            // Temporarily public, until we can find a better way to expose them to the product implementation
        public:
            const _StiffnessSolver & _stiffness_solver;
            const Eigen::SparseMatrix<Scalar> & _I_H;
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

        /**
         * WARNING: this class does not work properly, and so will be woefully detrimental to convergence at the moment!
         * TODO: Fix it!
         */
        template <typename Scalar>
        class SchurComplementDiagonalPreconditioner
        {
        public:
            SchurComplementDiagonalPreconditioner() {}

            template<typename MatrixType>
            SchurComplementDiagonalPreconditioner& analyzePattern(const MatrixType& ) { return *this; }

            template<typename MatrixType>
            SchurComplementDiagonalPreconditioner& factorize(const MatrixType&) { return *this; }

            template<typename MatrixType>
            SchurComplementDiagonalPreconditioner& compute(const MatrixType&) { return *this; }

            // Note: this copies the output, so it's not currently optimal. Can maybe use the Solve<> struct
            // instead?
            template<typename Rhs>
            inline const Rhs solve(const Rhs& b) const
            {
                // This is completely wrong
                return _I_H * _A_inv_diag * _I_H.transpose() * b;
            }

            Eigen::ComputationInfo info() { return Eigen::Success; }

            template <typename MatrixType>
            void setStiffnessMatrix(const MatrixType & A)
            {
                _A_inv_diag.resize(A.cols());
                for(int j = 0; j < A.outerSize(); ++j)
                {
                    typename MatrixType::InnerIterator it(A, j);
                    while(it && it.index() != j) ++it;

                    if(it && it.index() == j && it.value() != Scalar(0))
                        _A_inv_diag(j) = Scalar(1) / it.value();
                    else
                        _A_inv_diag(j) = Scalar(1);
                }
            }

            template<typename MatrixType>
            void setQuasiInterpolator(const MatrixType& I_H) { _I_H = I_H; }

        private:
            VectorX<Scalar> _A_inv_diag;
            Eigen::SparseMatrix<Scalar>   _I_H;
        };
    }

    template <typename Scalar>
    class SchurCorrectorSolver : public CorrectorSolver<Scalar>
    {
    public:
        virtual std::vector<Eigen::Triplet<Scalar>> compute_element_correctors_for_patch(
                const BiscaleMesh<Scalar, int> & mesh,
                const std::vector<int> & fine_patch_interior,
                const Eigen::SparseMatrix<Scalar> & local_coarse_stiffness,
                const Eigen::SparseMatrix<Scalar> & local_fine_stiffness,
                const Eigen::SparseMatrix<Scalar> & local_quasi_interpolator,
                int coarse_element) const override
        {
            assert(local_fine_stiffness.rows() > 0);

            using Eigen::SparseMatrix;
            using Eigen::ConjugateGradient;
            using Eigen::SimplicialLDLT;
            using detail::SchurComplementCoarseStiffnessPreconditioner;

            // Define the type of the solver used to solve the "stiffness problem" Ax = b.
            typedef ConjugateGradient<
                    SparseMatrix<Scalar>,
                    Eigen::Lower|Eigen::Upper,
                    SimplicialLDLT<SparseMatrix<Scalar>>> StiffnessSolver;

            // The SchurComplement uses the StiffnessSolver internally
            typedef detail::SchurComplement<Scalar, StiffnessSolver> SchurComplement;

            // Finally we define the type of solver used for the Schur complement
            typedef ConjugateGradient<
                    SchurComplement,
                    Eigen::Lower|Eigen::Upper,
                    SchurComplementCoarseStiffnessPreconditioner<Scalar>> SchurComplementSolver;

            std::vector<Eigen::Triplet<Scalar>> triplets;
            const auto & I_H = local_quasi_interpolator;
            const auto & A_h = local_fine_stiffness;
            const auto & A_H = local_coarse_stiffness;

            StiffnessSolver stiffness_solver(A_h);
            SchurComplement S(stiffness_solver, I_H);
            SchurComplementSolver schur_solver;
            schur_solver.preconditioner().setCoarseStiffnessMatrix(A_H);
            schur_solver.compute(S);

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

    private:
        SparseLuCorrectorSolver<Scalar> _lu;
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
                const auto & A = lhs._stiffness_solver;
                const auto & I_H = lhs._I_H;
                dst = alpha * I_H * A.solve(I_H.transpose() * rhs);
            }
        };
    }
}
