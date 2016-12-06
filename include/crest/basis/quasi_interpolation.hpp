#pragma once

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/geometry/biscale_mesh.hpp>
#include <crest/geometry/triangle.hpp>
#include <crest/quadrature/triquad.hpp>
#include <crest/util/eigen_extensions.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace crest
{
    namespace detail
    {
        /**
         * Returns the coefficients of the three linear Lagrange basis functions
         * associated with the triangle.
         *
         * Given basis function l_i for i = 0, 1, 2, then
         * l_i = a_i x + b_i y + c_i
         * where
         *      [ a_0   a_1   a_2 ]
         * B =  [ b_0   b_1   b_2 ]
         *      [ c_0   c_1   c_2 ]
         * @param triangle
         * @return
         */
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
        Eigen::SparseMatrix<Scalar> build_affine_interpolator_rhs(
                const BiscaleMesh<Scalar, int> & mesh)
        {
            std::vector<Eigen::Triplet<Scalar>> triplets;

            for (int coarse_index = 0; coarse_index < mesh.coarse_mesh().num_elements(); ++coarse_index)
            {
                // The basis functions p_h in the affine space can be expressed as
                // p_h(x, y) = a x + b y + c
                // for some coefficients a, b, c.
                const auto coarse_triangle = mesh.coarse_mesh().triangle_for(coarse_index);
                const Eigen::Matrix<Scalar, 3, 3> coarse_coeff = basis_coefficients_for_triangle(coarse_triangle);
                for (const auto fine_index : mesh.descendants_for(coarse_index))
                {
                    const auto vertex_indices = mesh.fine_mesh().elements()[fine_index].vertex_indices;
                    const auto fine_triangle = mesh.fine_mesh().triangle_for(fine_index);
                    const Eigen::Matrix<Scalar, 3, 3> fine_coeff = basis_coefficients_for_triangle(fine_triangle);

                    // Local coarse basis function i
                    for (size_t i = 0; i < 3; ++i)
                    {
                        // Local fine basis function j
                        for (size_t j = 0; j < 3; ++j)
                        {
                            const auto vertex_index = vertex_indices[j];
                            const auto product = [&](auto x, auto y)
                            {
                                const auto coarse_basis_value =
                                        coarse_coeff(0, i) * x + coarse_coeff(1, i) * y + coarse_coeff(2, i);
                                const auto fine_basis_value =
                                        fine_coeff(0, j) * x + fine_coeff(1, j) * y + fine_coeff(2, j);
                                return coarse_basis_value * fine_basis_value;
                            };
                            const auto inner_product = triquad<2>(product,
                                                                  fine_triangle.a,
                                                                  fine_triangle.b,
                                                                  fine_triangle.c);
                            const auto row = 3 * coarse_index + i;
                            triplets.push_back(Eigen::Triplet<Scalar>(row, vertex_index, inner_product));
                        }
                    }
                }
            }

            const auto num_dof_affine_space = 3 * mesh.coarse_mesh().num_elements();
            Eigen::SparseMatrix<Scalar> B(num_dof_affine_space, mesh.fine_mesh().num_vertices());
            B.setFromTriplets(triplets.cbegin(), triplets.cend());
            return B;
        };

        template <typename Scalar>
        Eigen::SparseMatrix<Scalar> build_affine_interpolator_lhs_inverse(
                const IndexedMesh<Scalar, int> & coarse)
        {
            const auto num_dof_affine_space = 3 * coarse.num_elements();
            Eigen::SparseMatrix<Scalar> P(num_dof_affine_space, num_dof_affine_space);

            // P will be block diagonal with 3x3 blocks, so we can reserve space in advance.
            P.reserve(Eigen::VectorXi::Constant(num_dof_affine_space, 3));

            Eigen::Matrix<Scalar, 3, 3> P_ref;
            P_ref << 2.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0,
                    1.0 / 24.0, 2.0 / 24.0, 1.0 / 24.0,
                    1.0 / 24.0, 1.0 / 24.0, 2.0 / 24.0;

            for (int t = 0; t < coarse.num_elements(); ++t)
            {
                const auto triangle = coarse.triangle_for(t);
                const auto determinant = static_cast<Scalar>(2.0) * crest::area(triangle);
                assert(determinant != static_cast<Scalar>(0));
                Eigen::Matrix<Scalar, 3, 3> P_local = determinant * P_ref.template cast<Scalar>();
                P_local = P_local.inverse().eval();

                for (int j = 0; j < 3; ++j)
                {
                    for (int i = 0; i < 3; ++i)
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
        Eigen::SparseMatrix<Scalar> affine_interpolator(
                const BiscaleMesh<Scalar, int> & mesh)
        {
            const Eigen::SparseMatrix<Scalar> B = build_affine_interpolator_rhs(mesh);
            const Eigen::SparseMatrix<Scalar> P = build_affine_interpolator_lhs_inverse(mesh.coarse_mesh());
            return P * B;
        };

        template <typename Scalar>
        std::vector<unsigned int> count_vertex_occurrences(const IndexedMesh<Scalar, int> & mesh)
        {
            auto count = std::vector<unsigned int>(mesh.num_vertices(), static_cast<Scalar>(0));
            for (const auto & element : mesh.elements())
            {
                for (const auto & v : element.vertex_indices)
                {
                    ++count[v];
                }
            }
            return count;
        }

        /**
         * Given a triangulation T, returns a matrix of dimensions N(T) x [3 * card(T)] which maps functions in the
         * (possibly discontinuous) affine space P_1 to functions in the standard linear finite element space
         * on the mesh.
         * Here, N(T) denotes the number of vertices in T, and card(T) denotes the number of elements in T.
         * @param mesh
         * @return
         */
        template <typename Scalar>
        Eigen::SparseMatrix<Scalar> nodal_average_interpolator(const IndexedMesh<Scalar, int> & mesh)
        {
            const auto occurrences = count_vertex_occurrences(mesh);

            // Reserve space for 1 non-zero per column, since each basis function in the affine space
            // maps to exactly one vertex in the standard finite element space.
            const auto num_dof_affine_space = 3 * mesh.num_elements();
            Eigen::SparseMatrix<Scalar> J(mesh.num_vertices(), num_dof_affine_space);
            J.reserve(Eigen::VectorXi::Constant(num_dof_affine_space, 1));

            for (int t = 0; t < mesh.num_elements(); ++t)
            {
                const auto vertices = mesh.elements()[t].vertex_indices;

                for (size_t v = 0; v < 3; ++v)
                {
                    const auto col = 3 * t + v;
                    const auto vertex_index = vertices[v];
                    const auto cardinality = occurrences[vertex_index];
                    J.insert(vertex_index, col) = static_cast<Scalar>(1.0) / static_cast<Scalar>(cardinality);
                }
            }

            return J;
        };
    }

    template <typename Scalar>
    Eigen::SparseMatrix<Scalar> quasi_interpolator(const BiscaleMesh<Scalar, int> & mesh)
    {
        const auto P = detail::affine_interpolator(mesh);
        const auto J = detail::nodal_average_interpolator(mesh.coarse_mesh());
        return J * P;
    }

}
