#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <rapidcheck/gtest.h>

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/geometry/refinement.hpp>
#include <crest/basis/lagrange_basis2d.hpp>
#include <crest/basis/homogenized_basis.hpp>

#include <util/eigen_matchers.hpp>
#include <util/test_generators.hpp>

using ::testing::ElementsAreArray;
using ::testing::Pointwise;
using ::testing::Eq;
using ::testing::DoubleEq;
using ::testing::Lt;
using ::testing::Each;

using crest::IndexedMesh;
using crest::LagrangeBasis2d;

typedef IndexedMesh<double, int>::Vertex Vertex;
typedef IndexedMesh<double, int>::Element Element;

TEST(homogenized_basis_test, correctors_are_in_interpolator_kernel_for_threshold_fine_mesh)
{
    // Recall that by definition, I_H c = 0, for any corrector with weights c.
    const std::vector<Vertex> coarse_vertices {
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(1.0, 1.0),
            Vertex(0.0, 1.0)
    };

    const std::vector<Element> coarse_elements {
            Element({3, 0, 1}),
            Element({1, 2, 3})
    };

    const auto initial_mesh = IndexedMesh<double, int>(coarse_vertices, coarse_elements);

    const auto two_scale_meshes = crest::threshold(initial_mesh, 0.5,
                                                   { crest::ReentrantCorner<double, int>(0, 2.35)});

    const auto basis = crest::detail::homogenized_basis(two_scale_meshes.coarse,
                                                        two_scale_meshes.fine,
                                                        4);

    ASSERT_THAT(basis.rows(), Eq(two_scale_meshes.coarse.num_vertices()));
    ASSERT_THAT(basis.cols(), Eq(two_scale_meshes.fine.num_vertices()));

    const auto I_H = crest::quasi_interpolator(two_scale_meshes.coarse, two_scale_meshes.fine);

    // Recall that each row of 'basis' corresponds to weights in the fine space, so by transposing it and
    // left-multiplying by I_H, we effectively compute I_H x_i for each basis function i.
    // Since x_i is in the kernel of I_H, we expect the result to be zero.
    const Eigen::SparseMatrix<double> Z = I_H * basis.transpose();
    const Eigen::MatrixXd Z_dense = Z;

    const auto max_abs = std::max(Z_dense.maxCoeff(), std::abs(Z_dense.minCoeff()));

    EXPECT_THAT(max_abs, Lt(1e-15));
}

TEST(construct_saddle_point_problem_test, blockwise_correct)
{
    MatrixX<double> A(5, 5);
    A.setZero();
    A(0, 0) = 1.0;
    A(1, 1) = 2.0;
    A(2, 2) = 3.0;
    A(3, 3) = 4.0;
    A(4, 4) = 5.0;

    MatrixX<double> I_H(3, 5);
    I_H.setZero();
    I_H(0, 0) = 6.0;
    I_H(1, 1) = 7.0;
    I_H(2, 2) = 8.0;

    VectorX<double> b(5);
    b.setZero();
    b(0) = 1.0;
    b(1) = 2.0;
    b(2) = 3.0;
    b(3) = 4.0;
    b(4) = 5.0;

    const Eigen::SparseMatrix<double> A_sparse = A.sparseView();
    const Eigen::SparseMatrix<double> I_H_sparse = I_H.sparseView();

    const auto saddle_point = crest::detail::construct_saddle_point_problem(A_sparse, I_H_sparse, b);
    const MatrixX<double> C = saddle_point.first;
    const VectorX<double> c = saddle_point.second;

    ASSERT_THAT(C.rows(), 8);
    ASSERT_THAT(C.cols(), 8);
    ASSERT_THAT(c.rows(), 8);

    const MatrixX<double> bottomRightCorner = C.bottomRightCorner(3, 3);
    const auto bottomRightCornerElements = std::vector<double>(bottomRightCorner.data(), bottomRightCorner.data() + 9);

    // TODO: The current MatrixEq is not entirely reliable, I think. Improve!
    EXPECT_THAT(C.topLeftCorner(5, 5), MatrixEq(A));
    EXPECT_THAT(C.bottomLeftCorner(3, 5), MatrixEq(I_H));
    EXPECT_THAT(C.topRightCorner(5, 3), MatrixEq(I_H.transpose()));
    EXPECT_THAT(bottomRightCornerElements, Each(DoubleEq(0.0)));

    EXPECT_THAT(c.topRows(5), MatrixEq(b));
    EXPECT_THAT(c.bottomRows(3)(0), DoubleEq(0.0));
    EXPECT_THAT(c.bottomRows(3)(1), DoubleEq(0.0));
    EXPECT_THAT(c.bottomRows(3)(2), DoubleEq(0.0));
}

RC_GTEST_PROP(homogenized_basis_test, correctors_are_in_interpolator_kernel, ())
{
    const auto oversampling = *rc::gen::arbitrary<unsigned int>();
    const auto coarse_mesh = *crest::gen::arbitrary_unit_square_mesh(4);

    // Make sure the fine mesh does not coincide with the coarse mesh
    const auto fine_mesh = *rc::gen::suchThat(crest::gen::arbitrary_refinement(coarse_mesh, 1, 4),
                                              [coarse_mesh] (auto mesh)
                                              {
                                                  return mesh.num_elements() != coarse_mesh.num_elements();
                                              });

    const auto basis = crest::detail::homogenized_basis(coarse_mesh, fine_mesh, oversampling);

    ASSERT_THAT(basis.rows(), Eq(coarse_mesh.num_vertices()));
    ASSERT_THAT(basis.cols(), Eq(fine_mesh.num_vertices()));

    const auto I_H = crest::quasi_interpolator(coarse_mesh, fine_mesh);

    // Recall that each row of 'basis' corresponds to weights in the fine space, so by transposing it and
    // left-multiplying by I_H, we effectively compute I_H x_i for each basis function i.
    // Since x_i is in the kernel of I_H, we expect the result to be zero.
    const Eigen::SparseMatrix<double> Z = I_H * basis.transpose();
    const Eigen::MatrixXd Z_dense = Z;

    const auto max_abs = std::max(std::abs(Z_dense.maxCoeff()), std::abs(Z_dense.minCoeff()));
    RC_ASSERT(max_abs < 1e-14);
}

RC_GTEST_PROP(homogenized_basis_test, correctors_are_zero_with_no_refinement, ())
{
    const auto oversampling = static_cast<unsigned int>(*rc::gen::inRange(0, 6));
    const auto mesh = *crest::gen::arbitrary_unit_square_mesh();
    const auto basis = crest::detail::homogenized_basis(mesh, mesh, oversampling);

    ASSERT_THAT(basis.rows(), Eq(mesh.num_vertices()));
    ASSERT_THAT(basis.cols(), Eq(mesh.num_vertices()));

    double max_abs = 0.0;
    for (int o = 0; o < basis.outerSize(); ++o)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(basis, o); it; ++it)
        {
            max_abs = std::max(max_abs, std::abs(it.value()));
        }
    }

    RC_ASSERT(max_abs < 1e-14);
}
