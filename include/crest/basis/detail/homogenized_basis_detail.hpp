#pragma once

#include <crest/geometry/biscale_mesh.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace crest
{
    namespace detail
    {
        template <typename Scalar>
        Eigen::SparseMatrix<Scalar> construct_saddle_point_problem(
                const Eigen::SparseMatrix<Scalar> & A,
                const Eigen::SparseMatrix<Scalar> & I_H)
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
                    triplets.push_back(Eigen::Triplet<Scalar>(row, col, - it.value()));
                    // Also add the transposed element, to bring I_H^T to the top-right corner
                    triplets.push_back(Eigen::Triplet<Scalar>(col, row, it.value()));
                }
            }

            Eigen::SparseMatrix<Scalar> C(A.rows() + I_H.rows(), A.cols() + I_H.rows());
            C.setFromTriplets(triplets.cbegin(), triplets.cend());

            return C;
        };

        template <typename Scalar>
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> construct_dense_saddle_point_problem(
                const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> & A,
                const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> & I_H)
        {
            const auto n = A.rows();
            const auto m = I_H.rows();

            assert(n == I_H.cols());
            assert(n == A.cols());

            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> C(m + n, m + n);
            C.topLeftCorner(n, n) = A;
            C.bottomLeftCorner(m, n) = I_H;
            C.topRightCorner(n, m) = I_H.transpose();
            C.bottomRightCorner(m, m) = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(m, m);
            return C;
        }

        /**
         * Computes the weights of the standard coarse Lagrangian basis functions in the fine space
         * @param coarse
         * @param fine
         * @return
         */
        template <typename Scalar>
        Eigen::SparseMatrix<Scalar> standard_coarse_basis_in_fine_space(const BiscaleMesh<Scalar, int> & mesh)
        {
            // A complicating matter here is that a vertex in the fine mesh can reside in multiple
            // coarse elements, and so it's easy to count them twice.

            // The approach demonstrated here is horribly inefficient and inelegant.
            // It is meant as a simple stop-gap solution.
            // TODO: Improve this

            std::set<std::pair<int, int>> visited;
            std::vector<Eigen::Triplet<Scalar>> triplets;

            for (int coarse_index = 0; coarse_index < mesh.coarse_mesh().num_elements(); ++coarse_index)
            {
                const auto coarse_element = mesh.coarse_mesh().elements()[coarse_index];
                const auto triangle = mesh.coarse_mesh().triangle_for(coarse_index);
                const auto coeff = detail::basis_coefficients_for_triangle(triangle);

                for (const auto fine_index : mesh.descendants_for(coarse_index))
                    for (int i = 0; i < 3; ++i)
                    {
                        const auto fine_element = mesh.fine_mesh().elements()[fine_index];
                        const auto v_index = fine_element.vertex_indices[i];
                        const auto v = mesh.fine_mesh().vertices()[v_index];

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

            Eigen::SparseMatrix<Scalar> basis(mesh.coarse_mesh().num_vertices(), mesh.fine_mesh().num_vertices());
            basis.setFromTriplets(triplets.begin(), triplets.end());
            return basis;
        }
    }
}
