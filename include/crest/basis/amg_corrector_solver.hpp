#pragma once

#include <crest/basis/homogenized_basis.hpp>

// Must include detail/qr here because it does not seem to be included by LGMRES,
// which it should be
#include <amgcl/detail/qr.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/backend/eigen.hpp>
#include <amgcl/solver/lgmres.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/solver/fgmres.hpp>
#include <amgcl/preconditioner/dummy.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

namespace crest
{
    namespace detail
    {
        template <typename Scalar>
        class CorrectorBlockMatrix
        {
        public:
            typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> Matrix;

            explicit CorrectorBlockMatrix(Matrix stiffness, Matrix quasi_interpolator)
                    : _stiffness(std::move(stiffness)), _quasi_interpolator(std::move(quasi_interpolator)) {}

            const Matrix & stiffness() const { return _stiffness; }

            const Matrix & quasi_interpolator() const { return _quasi_interpolator; }

            Eigen::Index rows() const { return stiffness().rows() + quasi_interpolator().rows(); }

            Eigen::Index cols() const { return stiffness().cols() + quasi_interpolator().cols(); }

        private:
            Eigen::SparseMatrix<Scalar, Eigen::RowMajor> _stiffness;
            Eigen::SparseMatrix<Scalar, Eigen::RowMajor> _quasi_interpolator;
        };

        template <class Scalar>
        class CorrectorBlockMatrixIdentityPreconditioner
        {
        public:
            typedef amgcl::backend::eigen<Scalar> Backend;
            typedef Backend backend_type;
            typedef CorrectorBlockMatrix<Scalar> matrix;
            typedef typename Backend::vector vector;
            typedef typename Backend::value_type value_type;

            typedef amgcl::detail::empty_params params;
            typedef typename Backend::params backend_params;

            template <class Matrix>
            CorrectorBlockMatrixIdentityPreconditioner(
                    const Matrix & M,
                    const params & prm = params(),
                    const backend_params & bprm = backend_params())
                    : _matrix(&M)
            {
                (void) prm;
                (void) bprm;
            }

            template <class Vec1, class Vec2>
            void apply(const Vec1 & rhs, Vec2 && x) const
            {
                amgcl::backend::copy(rhs, x);
            }

            const matrix & system_matrix() const
            {
                return *_matrix;
            }

        private:
            const matrix * _matrix;
        };

//        template <typename StiffnessPreconditioner>
//        class CorrectorBlockPreconditioner {
//        public:
//
//        };
    }
}


namespace crest
{



    template <typename Scalar>
    class AmgCorrectorSolver : public CorrectorSolver<Scalar>
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
            const auto m = I_H.rows();

            typedef amgcl::backend::eigen<Scalar> Backend;
            typedef detail::CorrectorBlockMatrixIdentityPreconditioner<Scalar> StiffnessPreconditioner;
            typedef amgcl::solver::gmres<Backend> IterativeSolver;
            typedef amgcl::make_solver<StiffnessPreconditioner, IterativeSolver> Solver;

            std::vector<Eigen::Triplet<Scalar>> triplets;

            detail::CorrectorBlockMatrix<Scalar> C(std::move(A_h), std::move(I_H));

            typename Solver::params params;
            params.solver.tol = std::numeric_limits<Scalar>::epsilon();
            Solver solver(C, params);

            for (int i = 0; i < 3; ++i)
            {
                const auto b = this->local_rhs(mesh, fine_patch_interior, coarse_element, i);

                VectorX<Scalar> sol(C.rows());
                sol.setZero();

                VectorX<Scalar> rhs(C.rows());
                rhs << b, VectorX<Scalar>::Zero(m);

                solver(rhs, sol);

                const auto global_index = mesh.coarse_mesh().elements()[coarse_element].vertex_indices[i];
                for (size_t k = 0; k < fine_patch_interior.size(); ++k)
                {
                    const auto component = sol(k);
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

namespace amgcl
{
    namespace backend
    {

        template <typename Scalar>
        struct rows_impl<::crest::detail::CorrectorBlockMatrix<Scalar>>
        {
            static size_t get(const ::crest::detail::CorrectorBlockMatrix<Scalar> &matrix) {
                return matrix.rows();
            }
        };

        template < class Alpha, class V1, class Beta, class V2, typename Scalar>
        struct spmv_impl<
                Alpha, ::crest::detail::CorrectorBlockMatrix<Scalar>, V1, Beta, V2,
                typename boost::enable_if<
                        typename boost::mpl::and_<
                                typename is_eigen_type<V1>::type,
                                typename is_eigen_type<V2>::type
                        >::type
                >::type
        >
        {
            static void apply(Alpha alpha, const ::crest::detail::CorrectorBlockMatrix<Scalar> &mat, const V1 &x, Beta beta, V2 &y)
            {
                const auto & A = mat.stiffness();
                const auto & I_H = mat.quasi_interpolator();
                const auto n = A.rows();
                const auto m = I_H.rows();

                if (!math::is_zero(beta))
                {
                    y = beta * y;
                    y.topRows(n) += alpha * A * x.topRows(n) + alpha * I_H.transpose() * x.bottomRows(m);
                    y.bottomRows(m) += alpha * I_H * x.bottomRows(m);
                } else
                {
                    // mat * x
                    y.topRows(n) = alpha * A * x.topRows(n) + alpha * I_H.transpose() * x.bottomRows(m);
                    y.bottomRows(m) = alpha * I_H * x.topRows(n);
                }
            }
        };

        template < class V1, class V2, class V3, typename Scalar >
        struct residual_impl<
                ::crest::detail::CorrectorBlockMatrix<Scalar>, V1, V2, V3,
                typename boost::enable_if<
                        typename boost::mpl::and_<
                                typename is_eigen_type<V1>::type,
                                typename is_eigen_type<V2>::type,
                                typename is_eigen_type<V3>::type
                        >::type
                >::type
        >
        {
            static void apply(const V1 &rhs, const ::crest::detail::CorrectorBlockMatrix<Scalar> &A, const V2 &x, V3 &r)
            {
                // Let r = Ax
                backend::spmv(1.0, A, x, 0, r);
                // Adjust r so that r = rhs - Ax
                r = rhs - r;
            }
        };

        template <typename Scalar>
        struct value_type<::crest::detail::CorrectorBlockMatrix<Scalar>, void>
        {
            typedef Scalar type;
        };
    }
}
