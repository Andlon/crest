#pragma once

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/geometry/triangle.hpp>
#include <crest/quadrature/triquad.hpp>
#include <crest/util/eigen_extensions.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace crest
{
    namespace detail
    {

        template <typename Scalar>
        Eigen::Matrix<Scalar, 3, 3> basis_coefficients_for_triangle(const Triangle<Scalar> & triangle)
        {
            constexpr Scalar one = static_cast<Scalar>(1.0);
            Eigen::Matrix<Scalar, 3, 3> X;
            X <<
              triangle.a.x, triangle.a.y, one,
                    triangle.b.x, triangle.b.y, one,
                    triangle.c.x, triangle.c.y, one;

            return X.partialPivLu().solve(Eigen::Matrix<Scalar, 3, 3>::Identity());
        };
        /**
         * The interpolation of a fine-scale finite element space in the (possibly discontinuous) space of
         * affine functions on a coarse finite element space can be computed by the relation
         * Py = Bx, where y represent the weights in the affine coarse space and x represents the weights
         * in the fine-scale finite element space. This function computes B.
         * @param coarse
         * @param fine
         * @return
         */
        template <typename Scalar>
        Eigen::SparseMatrix<Scalar, Eigen::RowMajor> build_affine_interpolator_rhs(
                const IndexedMesh<Scalar, int> & coarse,
                const IndexedMesh<Scalar, int> & fine)
        {
            const auto num_dof_affine_space = 3 * coarse.num_elements();
            Eigen::SparseMatrix<Scalar, Eigen::RowMajor> B(num_dof_affine_space, fine.num_vertices());

            for (int t = 0; t < coarse.num_elements(); ++t)
            {
                // The basis functions p_h in the affine space can be expressed as
                // p_h(x, y) = a x + b y + c
                // for some coefficients a, b, c.
                const auto coarse_triangle = coarse.triangle_for(t);
                const Eigen::Matrix<Scalar, 3, 3> coarse_coeff = basis_coefficients_for_triangle(coarse_triangle);
                for (size_t i = 0; i < 3; ++i)
                {
                    // TODO: Implement reverse ancestry lookup for determining fine-scale elements contained
                    // in the current coarse scale element
                    for (int k = 0; k < fine.num_elements(); ++k)
                    {
                        if (fine.ancestor_for(k) == t)
                        {
                            const auto vertex_indices = fine.elements()[k].vertex_indices;
                            const auto fine_triangle = fine.triangle_for(k);
                            const Eigen::Matrix<Scalar, 3, 3> fine_coeff = basis_coefficients_for_triangle(fine_triangle);
                            for (size_t j = 0; j < 3; ++j)
                            {
                                const auto vertex_index = vertex_indices[j];
                                // TODO: Construct matrix by triplets instead?
                                const auto product = [&] (auto x, auto y)
                                {
                                    const auto coarse_basis_value =
                                            coarse_coeff(0, i) * x + coarse_coeff(1, i) * y + coarse_coeff(2, i);
                                    const auto fine_basis_value =
                                            fine_coeff(0, j) * x + fine_coeff(1, j) * y + fine_coeff(2, j);
                                    return coarse_basis_value * fine_basis_value;
                                };
                                B.coeffRef(3 * t + i, vertex_index) += triquad<2>(product,
                                                                                  fine_triangle.a,
                                                                                  fine_triangle.b,
                                                                                  fine_triangle.c);
                            }
                        }
                    }
                }
            }

            return B;
        };

        template <typename Scalar>
        Eigen::SparseMatrix<Scalar, Eigen::RowMajor> build_affine_interpolator_lhs_inverse(
                const IndexedMesh<Scalar, int> & coarse)
        {
            const auto num_dof_affine_space = 3 * coarse.num_elements();
            Eigen::SparseMatrix<Scalar, Eigen::RowMajor> P(num_dof_affine_space, num_dof_affine_space);

            // P will be block diagonal with 3x3 blocks, so we can reserve space in advance.
            P.reserve(VectorX<Scalar>::Constant(num_dof_affine_space, 3));

            Eigen::Matrix<Scalar, 3, 3> P_ref;
            P_ref << 2.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0,
                    1.0 / 24.0, 2.0 / 24.0, 1.0 / 24.0,
                    1.0 / 24.0, 1.0 / 24.0, 2.0 / 24.0;

            for (int t = 0; t < coarse.num_elements(); ++t)
            {
                const auto triangle = coarse.triangle_for(t);
                const auto determinant = static_cast<Scalar>(2.0) * area(triangle);
                Eigen::Matrix<Scalar, 3, 3> P_local = determinant * P_ref.template cast<Scalar>();
                P_local = P_local.inverse().eval();

                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        const auto row = 3 * t + i;
                        const auto col = 3 * t + j;
                        P.insert(row, col) = P_local(i, j);
                    }
                }
            }

            return P;
        };

        /**
         * Given a coarse and fine mesh where the coarse mesh is a subset of the fine mesh (when interpreted as a forest),
         * build a matrix that interpolates functions in the fine finite element space in the
         * (possibly discontinuous) space of affine functions on the coarse space.
         * @param coarse
         * @param fine
         * @return
         */
        template <typename Scalar>
        Eigen::SparseMatrix<Scalar, Eigen::RowMajor> affine_interpolator(
                const IndexedMesh<Scalar, int> & coarse,
                const IndexedMesh<Scalar, int> & fine)
        {
            const Eigen::SparseMatrix<Scalar, Eigen::RowMajor> B = build_affine_interpolator_rhs(coarse, fine);
            const Eigen::SparseMatrix<Scalar, Eigen::RowMajor> P = build_affine_interpolator_lhs_inverse(coarse);
            return P * B;
        };
    }

}