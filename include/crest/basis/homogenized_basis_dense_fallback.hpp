#pragma once

namespace crest
{

    /**
     * Wraps an existing CorrectorSolver implementation and replaces
     * corrector computations for small patches with dense linear algebra
     * algorithms.
     *
     * This is useful primarily because sparse linear algebra is relatively slow
     * for very small linear systems, and in corrector problems one often has to solve
     * a vast amount of small problems in addition to large problems. This wrapper
     * provides a mechanism to fallback to dense linear algebra for the corrector
     * problems which are deemed to be sufficiently small.
     */
    template <typename Scalar, typename SparseCorrectorSolver>
    class DenseFallbackCorrectorSolverWrapper : public CorrectorSolver<Scalar>
    {
    private:
        unsigned int _threshold;
        SparseCorrectorSolver _wrapped;
    public:
        SparseCorrectorSolver & solver() { return _wrapped; }
        const SparseCorrectorSolver & solver() const { return _wrapped; }

        unsigned int threshold() const { return _threshold; }
        void set_threshold(unsigned int threshold) { _threshold = threshold; }

        template <typename ... Args>
        DenseFallbackCorrectorSolverWrapper(Args && ... args)
                : _threshold(300),
                  _wrapped(std::forward<Args>(args) ...) {}

        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DenseMatrix;

        virtual std::vector<Eigen::Triplet<Scalar>> resolve_element_correctors_for_patch(
                const BiscaleMesh<Scalar, int> & mesh,
                const std::vector<int> & coarse_patch_interior,
                const std::vector<int> & fine_patch_interior,
                const Eigen::SparseMatrix<Scalar> & global_coarse_stiffness,
                const Eigen::SparseMatrix<Scalar> & global_fine_stiffness,
                const Eigen::SparseMatrix<Scalar> & global_quasi_interpolator,
                int coarse_element) const override
        {
            if (fine_patch_interior.size() < threshold())
            {
                // For small patches, use dense linear algebra for faster computations
                // (sparse computations are slow for small systems)
                const auto I_H_local = sparse_submatrix_as_dense(global_quasi_interpolator,
                                                                 coarse_patch_interior,
                                                                 fine_patch_interior);
                const auto A_fine_local = sparse_submatrix_as_dense(global_fine_stiffness,
                                                                    fine_patch_interior,
                                                                    fine_patch_interior);

                return compute_element_correctors_for_patch_dense(
                        mesh,
                        fine_patch_interior,
                        std::move(A_fine_local),
                        std::move(I_H_local),
                        coarse_element);
            } else
            {
                return _wrapped.resolve_element_correctors_for_patch(
                        mesh,
                        coarse_patch_interior,
                        fine_patch_interior,
                        global_coarse_stiffness,
                        global_fine_stiffness,
                        global_quasi_interpolator,
                        coarse_element);
            }
        }

        std::vector<Eigen::Triplet<Scalar>>
        compute_element_correctors_for_patch_dense(
                const BiscaleMesh<Scalar, int> & mesh,
                const std::vector<int> & fine_patch_interior,
                DenseMatrix local_fine_stiffness,
                DenseMatrix local_quasi_interpolator,
                int coarse_element) const
        {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            const auto & I_H = local_quasi_interpolator;
            const auto & A = local_fine_stiffness;

            const auto C = detail::construct_dense_saddle_point_problem(A, I_H);
            const auto lu = C.partialPivLu();

            for (int i = 0; i < 3; ++i)
            {
                const auto b_local = this->local_rhs(mesh, fine_patch_interior, coarse_element, i);

                VectorX<Scalar> c(C.rows());
                c << b_local, VectorX<Scalar>::Zero(I_H.rows());

                // Recall that the solution is of the form [x, kappa], where kappa is merely a Lagrange multiplier, so
                // we extract x as the corrector.
                const VectorX<Scalar> corrector = lu.solve(c).topRows(A.rows());

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
