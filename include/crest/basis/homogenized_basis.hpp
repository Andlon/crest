#pragma once

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/geometry/mesh_algorithms.hpp>
#include <crest/basis/quasi_interpolation.hpp>
#include <crest/basis/lagrange_basis2d.hpp>
#include <crest/util/eigen_extensions.hpp>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <set>
#include <cassert>

namespace crest
{
    namespace detail
    {
        template <typename Scalar, typename Index>
        std::vector<Index> fine_patch_from_coarse(const IndexedMesh<Scalar, Index> & fine,
                                                  const std::vector<Index> & coarse_patch)
        {
            std::vector<Index> fine_patch;
            for (const auto t : coarse_patch)
            {
                for (int k = 0; k < fine.num_elements(); ++k)
                {
                    // TODO: Replace this with something more efficient than a full linear search, i.e. a reverse lookup
                    if (fine.ancestor_for(k) == t)
                    {
                        fine_patch.push_back(k);
                    }
                }
            }
            std::sort(fine_patch.begin(), fine_patch.end());
            fine_patch.erase(std::unique(fine_patch.begin(), fine_patch.end()), fine_patch.end());
            return fine_patch;
        }

        template <typename Scalar>
        VectorX<Scalar> local_rhs(const IndexedMesh<Scalar, int> & coarse,
                                  const IndexedMesh<Scalar, int> & fine,
                                  int coarse_element,
                                  int local_index,
                                  const std::vector<int> & fine_patch,
                                  const std::vector<int> & fine_patch_interior)
        {
            // TODO: Simplify this function
            assert(std::is_sorted(fine_patch.cbegin(), fine_patch.cend()));
            assert(std::is_sorted(fine_patch_interior.cbegin(), fine_patch_interior.cend()));

            const auto coarse_triangle = coarse.triangle_for(coarse_element);
            const Eigen::Matrix<Scalar, 3, 3> coarse_coeff = basis_coefficients_for_triangle(coarse_triangle);

            const auto coarse_grad_x = coarse_coeff(0, local_index);
            const auto coarse_grad_y = coarse_coeff(1, local_index);

            VectorX<Scalar> rhs(fine_patch_interior.size());
            rhs.setZero();
            for (auto k : fine_patch)
            {
                // TODO: Replace this linear loop with a reverse ancestry lookup
                if (fine.ancestor_for(k) == coarse_element)
                {
                    const auto vertex_indices = fine.elements()[k].vertex_indices;
                    const auto fine_triangle = fine.triangle_for(k);
                    const Eigen::Matrix<Scalar, 3, 3> fine_coeff = basis_coefficients_for_triangle(fine_triangle);


                    for (size_t j = 0; j < 3; ++j)
                    {
                        const auto fine_grad_x = fine_coeff(0, j);
                        const auto fine_grad_y = fine_coeff(1, j);

                        const auto product = [&] (auto  , auto  )
                        {
                            return coarse_grad_x * fine_grad_x + coarse_grad_y * fine_grad_y;
                        };

                        // At this point, we only know the index of the vertex in the global mesh,
                        // but we need the index of the vertex with respect to the patch interior. For now,
                        // we just perform a binary search to recover it, though there may be much more efficient ways.
                        // For example, we can construct a hashmap in the beginning of this function (which may
                        // or may not be more efficient).
                        const auto vertex_index = vertex_indices[j];
                        const auto range = std::equal_range(fine_patch_interior.cbegin(),
                                                            fine_patch_interior.cend(),
                                                            vertex_index);

                        if (range.first != range.second)
                        {
                            const auto inner_product = triquad<2>(product,
                                                                  fine_triangle.a,
                                                                  fine_triangle.b,
                                                                  fine_triangle.c);

                            const auto local_index = range.first - fine_patch_interior.cbegin();
                            rhs(local_index) += inner_product;
                        }

                    }
                }
            }
            return rhs;
        }

        template <typename Scalar>
        std::pair<Eigen::SparseMatrix<Scalar>, VectorX<Scalar>> construct_saddle_point_problem(
                const Eigen::SparseMatrix<Scalar> & A,
                const Eigen::SparseMatrix<Scalar> & I_H,
                const VectorX<Scalar> & b)
        {
            // Define the matrix
            //
            // C = [ A      I_H^T ]
            //     [ I_H      0   ]
            //
            // and the right-hand side
            //
            // c = [ b ]
            //     [ 0 ]
            //
            // Solving the system C [ x, kappa ] = c then gives the corrector weights x.
            // kappa merely corresponds to a Lagrange multiplier and can be discarded.

            // Constructing the C matrix with triplets is not the most efficient way, but it is
            // fairly simple and probably still relatively efficient.
            std::vector<Eigen::Triplet<Scalar>> triplets;
            for (int i = 0; i < A.outerSize(); ++i)
            {
                for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(A, i); it; ++it)
                {
                    triplets.push_back(Eigen::Triplet<Scalar>(it.row(), it.col(), it.value()));
                }
            }

            for (int i = 0; i < I_H.outerSize(); ++i)
            {
                for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(I_H, i); it; ++it)
                {
                    const auto row = it.row() + A.rows();
                    const auto col = it.col();
                    triplets.push_back(Eigen::Triplet<Scalar>(row, col, it.value()));
                    // Also add the transposed element, to bring I_H^T to the top-right corner
                    triplets.push_back(Eigen::Triplet<Scalar>(col, row, it.value()));
                }
            }

            Eigen::SparseMatrix<Scalar> C(A.rows() + I_H.rows(), A.cols() + I_H.rows());
            C.setFromTriplets(triplets.cbegin(), triplets.cend());

            VectorX<Scalar> c(A.rows() + I_H.rows());
            c.setZero();
            c.topRows(A.rows()) = b;

            return std::make_pair(C, c);
        };

        template <typename Scalar>
        VectorX<Scalar> solve_localized_corrector_problem(const Eigen::SparseMatrix<Scalar> & A,
                                                          const Eigen::SparseMatrix<Scalar> & I_H,
                                                          const VectorX<Scalar> & b)
        {
            const auto constrained_problem = construct_saddle_point_problem(A, I_H, b);
            const auto C = constrained_problem.first;
            const auto c = constrained_problem.second;
            VectorX<Scalar> solution(C.rows());

            // Note: This _may_ run out of memory if I_H is dense (which is the case when using the exact SVD approach)
            Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
            solver.analyzePattern(C);
            solver.factorize(C);
            assert(solver.info() == Eigen::Success);
            solution = solver.solve(c);

            // Recall that the solution is of the form [x, kappa], where kappa is merely a Lagrange multiplier, so
            // we extract x as the corrector.
            return solution.topRows(A.rows());
        }

        template <typename Scalar>
        Eigen::SparseMatrix<Scalar>
        localized_quasi_interpolator(const Eigen::SparseMatrix<Scalar> & global_interpolator,
                                     const IndexedMesh<Scalar, int> & coarse_mesh,
                                     const std::vector<int> & coarse_patch,
                                     const std::vector<int> & fine_patch_interior)
        {
            assert(std::is_sorted(coarse_patch.cbegin(), coarse_patch.cend()));
            assert(std::is_sorted(fine_patch_interior.cbegin(), fine_patch_interior.cend()));

            // Recall that the global interpolator maps only to the coarse interior, so we must
            // with exclude coarse boundary nodes from the assembled system
            const auto coarse_patch_vertices = patch_vertices(coarse_mesh, coarse_patch);
            std::vector<int> coarse_dof;
            std::set_difference(coarse_patch_vertices.begin(), coarse_patch_vertices.end(),
                                coarse_mesh.boundary_vertices().begin(), coarse_mesh.boundary_vertices().end(),
                                std::back_inserter(coarse_dof));

            if (coarse_dof.empty())
            {
                return Eigen::SparseMatrix<Scalar>(0, 0);
            }
            else
            {
                const auto I_H_local = sparse_submatrix(global_interpolator,
                                                        coarse_dof,
                                                        fine_patch_interior);

                // In its current state, the local interpolator matrix may be rank-deficient, which makes
                // the solution process problematic. In this case we want to fulfill the interpolation constraint
                // exactly, so we will resort to a very expensive dense SVD computation.
                // Given the SVD given by I_H_local = U S V^T,
                // and denoting r = rank(I_H_local), recall that the last n - r columns of V span the null space of
                // I_H_local. Since we are only interested in the null space, we can replace I_H_local by
                // an r x n matrix I_H_reduced such that ker(I_H_reduced) = ker(I_H_local).
                // To do this, we note simply that we only require the SVD of I_H_reduced to share the same right
                // singular vectors, and so we simply let the U and S matrices be the r x r identity matrix and a
                // r x n rectangular identity matrix, respectively. Hence, writing
                //
                // I_H_reduced = I_r * [ I_r   0 ] V^T
                //             = [ I_r   0 ] [ V_11^T   V_21^T ]
                //             =             [ V_12^T   V_22^T ]
                //             = [ V_11^T    V_21^T ],
                //
                // and hence we can define I_H_reduced as V_r^T, where V_r corresponds to the first r columns of
                // the original V.
                const Eigen::JacobiSVD<MatrixX<Scalar>> svd(MatrixX<Scalar>(I_H_local), Eigen::ComputeFullV);
                const auto r = svd.rank();
                const auto V_r = svd.matrixV().leftCols(r);

                const MatrixX<Scalar> I_H_reduced = V_r.transpose();
                assert(I_H_reduced.rows() == r);
                assert(I_H_reduced.cols() == static_cast<int>(fine_patch_interior.size()));
                return Eigen::SparseMatrix<Scalar>(I_H_reduced.sparseView());
            }
        }

        template <typename Scalar>
        std::vector<Eigen::Triplet<Scalar>> compute_element_corrector_for_node(
                const IndexedMesh<Scalar, int> & coarse,
                const IndexedMesh<Scalar, int> & fine,
                const Eigen::SparseMatrix<Scalar> & fine_stiffness_matrix,
                const Eigen::SparseMatrix<Scalar> & quasi_interpolator,
                int coarse_element,
                int local_index,
                unsigned int oversampling)
        {
            assert(local_index >= 0 && local_index < 3);
            assert(coarse_element >= 0 && coarse_element < coarse.num_elements());

            std::vector<Eigen::Triplet<Scalar>> triplets;

            const auto global_index = coarse.elements()[coarse_element].vertex_indices[local_index];
            const auto coarse_patch = patch_for_element(coarse, coarse_element, oversampling);
            const auto fine_patch = fine_patch_from_coarse(fine, coarse_patch);
            const auto fine_patch_interior = patch_interior(fine, fine_patch);

            if (fine_patch_interior.empty())
            {
                return triplets;
            } else
            {
                const auto I_H_local = localized_quasi_interpolator(quasi_interpolator,
                                                                    coarse,
                                                                    coarse_patch,
                                                                    fine_patch_interior);
                const auto A_local = sparse_submatrix(fine_stiffness_matrix, fine_patch_interior, fine_patch_interior);
                const auto b_local = local_rhs(coarse, fine, coarse_element, local_index, fine_patch,
                                               fine_patch_interior);

                const auto corrector = solve_localized_corrector_problem(A_local, I_H_local, b_local);

                assert(static_cast<size_t>(corrector.rows()) == fine_patch_interior.size());
                for (size_t k = 0; k < fine_patch_interior.size(); ++k)
                {
                    triplets.push_back(Eigen::Triplet<Scalar>(global_index, fine_patch_interior[k], corrector(k)));
                }
                return triplets;
            }
        }

        template <typename Scalar>
        Eigen::SparseMatrix<Scalar> homogenized_basis_correctors(
                const IndexedMesh<Scalar, int> & coarse,
                const IndexedMesh<Scalar, int> & fine,
                unsigned int oversampling)
        {
            const auto I_H = quasi_interpolator(coarse, fine);

            // TODO: Rewrite LagrangeBasis2d so that we can reuse its functionality instead of this workaround
            const auto assembly = assemble_linear_lagrangian_system_triplets(fine);
            Eigen::SparseMatrix<Scalar> A(fine.num_vertices(), fine.num_vertices());
            A.setFromTriplets(assembly.stiffness_triplets.cbegin(), assembly.stiffness_triplets.cend());

            std::vector<Eigen::Triplet<Scalar>> basis_triplets;
            for (int t = 0; t < coarse.num_elements(); ++t)
            {
                for (size_t i = 0; i < 3; ++i)
                {
                    const auto corrector_contributions = compute_element_corrector_for_node(coarse,
                                                                                            fine,
                                                                                            A,
                                                                                            I_H,
                                                                                            t,
                                                                                            i,
                                                                                            oversampling);
                    std::copy(corrector_contributions.cbegin(),
                              corrector_contributions.cend(),
                              std::back_inserter(basis_triplets));
                }
            }

            Eigen::SparseMatrix<Scalar> basis(coarse.num_vertices(), fine.num_vertices());
            basis.setFromTriplets(basis_triplets.cbegin(), basis_triplets.cend());
            return basis;
        }

        /**
         * Computes the weights of the standard coarse Lagrangian basis functions in the fine space
         * @param coarse
         * @param fine
         * @return
         */
        template <typename Scalar>
        Eigen::SparseMatrix<Scalar> standard_coarse_basis_in_fine_space(const IndexedMesh<Scalar, int> & coarse,
                                                                        const IndexedMesh<Scalar, int> & fine)
        {
            // A complicating matter here is that a vertex in the fine mesh can reside in multiple
            // coarse elements, and so it's easy to count them twice.

            // The approach demonstrated here is horribly inefficient and inelegant.
            // It is meant as a simple stop-gap solution.
            // TODO: Improve this

            std::set<std::pair<int, int>> visited;
            std::vector<Eigen::Triplet<Scalar>> triplets;

            for (int k = 0; k < coarse.num_elements(); ++k)
            {
                const auto coarse_element = coarse.elements()[k];
                const auto triangle = coarse.triangle_for(k);
                const auto coeff = detail::basis_coefficients_for_triangle(triangle);

                for (int t = 0; t < fine.num_elements(); ++t)
                {
                    // TODO: Implement efficient lookup to prevent quadratic complexity in iteration of fine nodes
                    if (fine.ancestor_for(t) == k)
                    {
                        const auto fine_element = fine.elements()[t];
                        for (int i = 0; i < 3; ++i)
                        {
                            const auto v_index = fine_element.vertex_indices[i];
                            const auto v = fine.vertices()[v_index];

                            for (int j = 0; j < 3; ++j)
                            {
                                const auto coarse_index = coarse_element.vertex_indices[j];
                                const auto key = std::make_pair(coarse_index, v_index);
                                if (visited.count(key) == 0)
                                {
                                    const auto a = coeff(0, j);
                                    const auto b = coeff(1, j);
                                    const auto c = coeff(2, j);

                                    const auto v_value = a * v.x + b * v.y + c;

                                    triplets.push_back(Eigen::Triplet<Scalar>(coarse_index, v_index, v_value));
                                    visited.insert(key);
                                }

                            }
                        }
                    }
                }
            }

            Eigen::SparseMatrix<Scalar> basis(coarse.num_vertices(), fine.num_vertices());
            basis.setFromTriplets(triplets.begin(), triplets.end());
            return basis;
        }

        template <typename Scalar>
        Eigen::SparseMatrix<Scalar> corrected_basis_coefficients(const IndexedMesh<Scalar, int> & coarse,
                                                                 const IndexedMesh<Scalar, int> & fine,
                                                                 unsigned int oversampling)
        {
            const auto corrector_weights = homogenized_basis_correctors(coarse, fine, oversampling);
            const auto lagrange_basis_weights = standard_coarse_basis_in_fine_space(coarse, fine);
            return lagrange_basis_weights - corrector_weights;
        }
    }

    template <typename Scalar>
    class HomogenizedBasis : public Basis<Scalar, HomogenizedBasis<Scalar>>
    {
    public:
        explicit HomogenizedBasis(const IndexedMesh<Scalar, int> & coarse,
                                  const IndexedMesh<Scalar, int> & fine,
                                  unsigned int oversampling)
                : _coarse(coarse), _fine(fine) {
            // TODO: Probably want to change the design of Basis to accommodate the fact that HomogenizedBasis
            // needs to do a lot of computation at construction in order to compute load vectors etc.
            _basis_weights = detail::corrected_basis_coefficients(coarse, fine, oversampling);
        }

        virtual std::vector<int> boundary_nodes() const override { return _coarse.boundary_vertices(); }

        virtual std::vector<int> interior_nodes() const override { return _coarse.compute_interior_vertices(); }

        virtual Assembly<Scalar> assemble() const override;

        virtual int num_dof() const override { return _coarse.num_vertices(); }

        template <typename Function2d>
        VectorX<Scalar> interpolate(const Function2d & f) const;

        template <int QuadStrength, typename Function2d>
        VectorX<Scalar> load(const Function2d & f) const;

        template <int QuadStrength, typename Function2d>
        Scalar error_l2(const Function2d & f, const VectorX<Scalar> & weights) const;

        template <int QuadStrength, typename Function2d_x, typename Function2d_y>
        Scalar error_h1_semi(const Function2d_x & f_x,
                             const Function2d_y & f_y,
                             const VectorX<Scalar> & weights) const;

    private:
        Eigen::SparseMatrix<Scalar> _basis_weights;
        const IndexedMesh<Scalar, int> & _coarse;
        const IndexedMesh<Scalar, int> & _fine;
    };

    template <typename Scalar>
    Assembly<Scalar> HomogenizedBasis<Scalar>::assemble() const
    {
        LagrangeBasis2d<Scalar> fine_basis(_fine);
        const auto fine_assembly = fine_basis.assemble();

        const auto & M = fine_assembly.mass;
        const auto & A = fine_assembly.stiffness;
        const auto & W = _basis_weights;

        Assembly<Scalar> assembly;
        assembly.mass = W * M * W.transpose();
        assembly.stiffness = W * A * W.transpose();
        return assembly;
    }

    template <typename Scalar>
    template <typename Function2d>
    VectorX<Scalar> HomogenizedBasis<Scalar>::interpolate(const Function2d & f) const
    {
        const LagrangeBasis2d<Scalar> fine_basis(_fine);
        const auto I_H = quasi_interpolator(_coarse, _fine);
        return I_H * fine_basis.interpolate(f);
    }

    template <typename Scalar>
    template <int QuadStrength, typename Function2d>
    VectorX<Scalar> HomogenizedBasis<Scalar>::load(const Function2d & f) const
    {
        const LagrangeBasis2d<Scalar> fine_basis(_fine);
        const auto fine_load = fine_basis.load<QuadStrength>(f);
        const auto & W = _basis_weights;
        return W * fine_load;
    };

    template <typename Scalar>
    template <int QuadStrength, typename Function2d>
    Scalar HomogenizedBasis<Scalar>::error_l2(const Function2d & f, const VectorX<Scalar> & weights) const
    {
        const LagrangeBasis2d<Scalar> fine_basis(_fine);
        const auto & W = _basis_weights;
        const VectorX<Scalar> fine_weights = W.transpose() * weights;
        return fine_basis.error_l2<QuadStrength>(f, fine_weights);
    };

    template <typename Scalar>
    template <int QuadStrength, typename Function2d_x, typename Function2d_y>
    Scalar HomogenizedBasis<Scalar>::error_h1_semi(const Function2d_x & f_x,
                                                   const Function2d_y & f_y,
                                                   const VectorX<Scalar> & weights) const
    {
        const LagrangeBasis2d<Scalar> fine_basis(_fine);
        const auto & W = _basis_weights;
        const VectorX<Scalar> fine_weights = W.transpose() * weights;
        return fine_basis.error_h1_semi<QuadStrength>(f_x, f_y, fine_weights);
    };
}
