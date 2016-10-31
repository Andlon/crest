#pragma once

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/geometry/mesh_algorithms.hpp>
#include <crest/basis/quasi_interpolation.hpp>
#include <crest/basis/linear_lagrangian_basis.hpp>
#include <crest/util/eigen_extensions.hpp>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

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
            assert(std::is_sorted(fine_patch.cbegin(), fine_patch.cend()));
            assert(std::is_sorted(fine_patch_interior.cbegin(), fine_patch_interior.cend()));

            const auto coarse_triangle = coarse.triangle_for(coarse_element);
            const Eigen::Matrix<Scalar, 3, 3> coarse_coeff = basis_coefficients_for_triangle(coarse_triangle);

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
                        const auto product = [&] (auto x, auto y)
                        {
                            const auto coarse_basis_value =
                                    coarse_coeff(0, local_index) * x
                                    + coarse_coeff(1, local_index) * y
                                    + coarse_coeff(2, local_index);
                            const auto fine_basis_value =
                                    fine_coeff(0, j) * x + fine_coeff(1, j) * y + fine_coeff(2, j);
                            return coarse_basis_value * fine_basis_value;
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
        std::pair<Eigen::SparseMatrix<Scalar>, VectorX<Scalar>> construct_constrained_problem(
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
            for (int r = 0; r < A.rows(); ++r)
            {
                for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(A, r); it; ++it)
                {
                    triplets.push_back(Eigen::Triplet<Scalar>(it.row(), it.col(), it.value()));
                }
            }

            for (int r = 0; r < I_H.rows(); ++r)
            {
                for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(I_H, r); it; ++it)
                {
                    const auto row = r + A.rows();
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
            const auto constrained_problem = construct_constrained_problem(A, I_H, b);
            const auto C = constrained_problem.first;
            const auto c = constrained_problem.second;

            Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
            solver.analyzePattern(C);
            solver.factorize(C);
            VectorX<Scalar> solution(C.rows());

            // Recall that the solution is of the form [x, kappa], where kappa is merely a Lagrange multiplier, so
            // we extract x as the corrector.
            return solution.topRows(A.rows());
        }

        template <typename Scalar>
        Eigen::SparseMatrix<Scalar> localized_quasi_interpolator(const Eigen::SparseMatrix<Scalar> & global_interpolator,
                                                                 const IndexedMesh<Scalar, int> & coarse_mesh,
                                                                 const std::vector<int> & coarse_patch,
                                                                 const std::vector<int> & fine_patch_interior)
        {
            assert(std::is_sorted(coarse_patch.cbegin(), coarse_patch.cend()));
            assert(std::is_sorted(fine_patch_interior.cbegin(), fine_patch_interior.cend()));

            // Since I_H x = 0 on the boundary of the global mesh, we need to make sure to exclude
            // the boundary vertices of the global mesh
            const auto coarse_patch_vertices = patch_vertices(coarse_mesh, coarse_patch);
            std::vector<int> coarse_patch_non_boundary_vertices;
            std::set_difference(coarse_patch_vertices.cbegin(), coarse_patch_vertices.cend(),
                                coarse_mesh.boundary_vertices().cbegin(), coarse_mesh.boundary_vertices().cend(),
                                std::back_inserter(coarse_patch_non_boundary_vertices));
            const auto I_H_local = sparse_submatrix(global_interpolator,
                                                    coarse_patch_non_boundary_vertices,
                                                    fine_patch_interior);

            // In its current state, the local interpolator matrix may be rank-deficient, which makes
            // the solution process problematic. In this case we want to fulfill the interpolation constraint
            // exactly, so we will resort to a very expensive dense SVD computation.
            // Given the SVD given by I_H_local = U S V^T,
            // and with r = rank(I_H_local), we can construct the matrix
            // I_H_reduced = U_r S V^T,
            // where U_r corresponds to the r first rows of U.
            // As it turns out, ker(I_H_reduced) = ker(I_H_local), and I_H_reduced has full row rank.
            // which means we can replace I_H_local with I_H_reduced in the saddle point formulation.
            const Eigen::JacobiSVD<MatrixX<Scalar>> svd(I_H_local, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const auto r = svd.rank();

            // In addition to the above, we'll use a compact SVD when reconstructing to reduce the work required.
            const auto U_r = svd.matrixU().topLeftCorner(r, r);
            const auto S_r = svd.singularValues().topRows(r).asDiagonal();
            const auto V_r = svd.matrixV().leftCols(r);

            const MatrixX<Scalar> I_H_reduced = U_r * S_r * V_r.transpose();
            return Eigen::SparseMatrix<Scalar>(I_H_reduced.sparseView());
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

            const auto coarse_patch = patch_for_element(coarse, coarse_element, oversampling);
            const auto fine_patch = fine_patch_from_coarse(fine, coarse_patch);
            const auto fine_patch_interior = patch_interior(fine, fine_patch);

            const auto I_H_local = localized_quasi_interpolator(quasi_interpolator,
                                                                coarse,
                                                                coarse_patch,
                                                                fine_patch_interior);
            const auto A_local = sparse_submatrix(fine_stiffness_matrix, fine_patch_interior, fine_patch_interior);
            const auto b_local = local_rhs(coarse, fine, coarse_element, local_index, fine_patch, fine_patch_interior);

            const auto corrector = solve_localized_corrector_problem(A_local, I_H_local, b_local);

            std::vector<Eigen::Triplet<Scalar>> triplets;
            const auto global_index = coarse.elements()[local_index].vertex_indices[local_index];
            for (size_t k = 0; k < fine_patch_interior.size(); ++k)
            {
                triplets.push_back(Eigen::Triplet<Scalar>(global_index, fine_patch_interior[k], corrector(k)));
            }
            return triplets;
        }

        template <typename Scalar>
        Eigen::SparseMatrix<Scalar> homogenized_basis(
                const IndexedMesh<Scalar, int> & coarse,
                const IndexedMesh<Scalar, int> & fine,
                unsigned int oversampling)
        {
            const auto I_H = quasi_interpolator(coarse, fine);

            // TODO: Rewrite LagrangeBasis2d so that we can reuse its functionality instead of this workaround
            const auto assembly = assemble_linear_lagrangian_stiffness_triplets(fine);
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
    }


}