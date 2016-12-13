#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <rapidcheck/gtest.h>

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/geometry/refinement.hpp>
#include <crest/basis/lagrange_basis2d.hpp>
#include <crest/basis/homogenized_basis.hpp>
#include <crest/basis/schur_corrector_solver.hpp>
#include <crest/basis/homogenized_basis_dense_fallback.hpp>

#include <crest/util/algorithms.hpp>

#include <util/eigen_matchers.hpp>
#include <util/test_generators.hpp>

#include <Eigen/Sparse>
#include <Eigen/Dense>

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

    const auto mesh = crest::threshold(initial_mesh, 0.5, { crest::ReentrantCorner<double, int>(0, 2.35)});
    const auto correctors = crest::SparseLuCorrectorSolver<double>().compute_correctors(mesh, 4);

    ASSERT_THAT(correctors.rows(), Eq(mesh.coarse_mesh().num_vertices()));
    ASSERT_THAT(correctors.cols(), Eq(mesh.fine_mesh().num_vertices()));

    const auto coarse_interior = mesh.coarse_mesh().compute_interior_vertices();
    const auto fine_dof = crest::algo::integer_range<int>(0, mesh.fine_mesh().num_vertices());

    const auto I_H = crest::quasi_interpolator(mesh);
    // We need to test with the coarse interior, because the matrix returned by quasi_interpolator
    // does not impose that I_H x = 0 on the boundary.
    const auto I_H_interior = sparse_submatrix(I_H, coarse_interior, fine_dof);

    // Recall that each row of 'basis' corresponds to weights in the fine space, so by transposing it and
    // left-multiplying by I_H, we effectively compute I_H x_i for each basis function i.
    // Since x_i is in the kernel of I_H, we expect the result to be zero.
    const Eigen::SparseMatrix<double> Z = I_H_interior * correctors.transpose();
    const Eigen::MatrixXd Z_dense = Z;

    EXPECT_TRUE(Z_dense.isZero(1e-15));
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

    const Eigen::SparseMatrix<double> A_sparse = A.sparseView();
    const Eigen::SparseMatrix<double> I_H_sparse = I_H.sparseView();

    const MatrixX<double> C = crest::detail::construct_saddle_point_problem(A_sparse, I_H_sparse);

    ASSERT_THAT(C.rows(), 8);
    ASSERT_THAT(C.cols(), 8);

    const MatrixX<double> bottomRightCorner = C.bottomRightCorner(3, 3);
    const auto bottomRightCornerElements = std::vector<double>(bottomRightCorner.data(), bottomRightCorner.data() + 9);

    EXPECT_TRUE(C.topLeftCorner(5, 5).isApprox(A));
    EXPECT_TRUE(C.bottomLeftCorner(3, 5).isApprox(I_H));
    EXPECT_TRUE(C.topRightCorner(5, 3).isApprox(I_H.transpose()));
    EXPECT_THAT(bottomRightCornerElements, Each(DoubleEq(0.0)));
}

template <typename CorrectorSolver>
void check_correctors_are_in_interpolator_kernel()
{
    const auto oversampling = *rc::gen::arbitrary<unsigned int>();
    const auto coarse_mesh = *crest::gen::arbitrary_unit_square_mesh(4);

    // Make sure the fine mesh does not coincide with the coarse mesh
    const auto fine_mesh = *rc::gen::suchThat(crest::gen::arbitrary_refinement(coarse_mesh, 1, 4),
                                              [coarse_mesh] (auto mesh)
                                              {
                                                  return mesh.num_elements() != coarse_mesh.num_elements();
                                              });

    const auto biscale = crest::BiscaleMesh<double, int>(coarse_mesh, fine_mesh);

    const auto correctors = CorrectorSolver().compute_correctors(biscale, oversampling);

    ASSERT_THAT(correctors.rows(), Eq(coarse_mesh.num_vertices()));
    ASSERT_THAT(correctors.cols(), Eq(fine_mesh.num_vertices()));

    const auto coarse_interior = coarse_mesh.compute_interior_vertices();
    const auto fine_dof = crest::algo::integer_range<int>(0, fine_mesh.num_vertices());

    const auto I_H = crest::quasi_interpolator(biscale);

    // We need to test with the coarse interior, because the matrix returned by quasi_interpolator
    // does not impose that I_H x = 0 on the boundary.
    const auto I_H_interior = sparse_submatrix(I_H, coarse_interior, fine_dof);

    // Recall that each row of 'basis' corresponds to weights in the fine space, so by transposing it and
    // left-multiplying by I_H, we effectively compute I_H x_i for each basis function i.
    // Since x_i is in the kernel of I_H, we expect the result to be zero.
    const Eigen::SparseMatrix<double> Z = I_H_interior * correctors.transpose();
    const Eigen::MatrixXd Z_dense = Z;

    RC_ASSERT(Z_dense.isZero(1e-14));
}

RC_GTEST_PROP(homogenized_basis_test, correctors_are_in_interpolator_kernel_sparselu, ())
{
    check_correctors_are_in_interpolator_kernel<crest::SparseLuCorrectorSolver<double>>();
}

RC_GTEST_PROP(homogenized_basis_test, correctors_are_in_interpolator_kernel_schur, ())
{
    check_correctors_are_in_interpolator_kernel<crest::SchurCorrectorSolver<double>>();
}

RC_GTEST_PROP(homogenized_basis_test, correctors_are_in_interpolator_kernel_sparselu_dense_fallback, ())
{
    typedef crest::DenseFallbackCorrectorSolverWrapper<double, crest::SparseLuCorrectorSolver<double>> Wrapper;
    check_correctors_are_in_interpolator_kernel<Wrapper>();
}

template <typename CorrectorSolver>
void check_correctors_are_zero_with_no_refinement()
{
    const auto oversampling = static_cast<unsigned int>(*rc::gen::inRange(0, 6));
    const auto mesh = *crest::gen::arbitrary_unit_square_mesh();
    const auto biscale = crest::BiscaleMesh<double, int>(mesh, mesh);
    const auto correctors = crest::SparseLuCorrectorSolver<double>().compute_correctors(biscale, oversampling);
    ASSERT_THAT(correctors.rows(), Eq(mesh.num_vertices()));
    ASSERT_THAT(correctors.cols(), Eq(mesh.num_vertices()));

    double max_abs = 0.0;
    for (int o = 0; o < correctors.outerSize(); ++o)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(correctors, o); it; ++it)
        {
            max_abs = std::max(max_abs, std::abs(it.value()));
        }
    }

    RC_ASSERT(max_abs < 1e-14);
}

RC_GTEST_PROP(homogenized_basis_test, correctors_are_zero_with_no_refinement_sparselu, ())
{
    check_correctors_are_zero_with_no_refinement<crest::SparseLuCorrectorSolver<double>>();
}

RC_GTEST_PROP(homogenized_basis_test, correctors_are_zero_with_no_refinement_schur, ())
{
    check_correctors_are_zero_with_no_refinement<crest::SchurCorrectorSolver<double>>();
}

RC_GTEST_PROP(homogenized_basis_test, correctors_are_zero_with_no_refinement_sparselu_dense_fallback, ())
{
    typedef crest::DenseFallbackCorrectorSolverWrapper<double, crest::SparseLuCorrectorSolver<double>> Wrapper;
    check_correctors_are_zero_with_no_refinement<Wrapper>();
}

template <typename CorrectorSolver>
void check_corrected_basis_is_orthogonal_to_fine_space()
{
    // Let W be such that W_ij corrected basis function i is defined by
    //
    // lambda_i = sum_j W_ij phi_j
    //
    // where phi_j denotes the fine-scale basis functions. Then the a-orthogonality of the corrected space V_H and
    // the fine-scale finite element space implies that
    //
    // W A N = 0
    //
    // where A is the fine-scale stiffness matrix (interior nodes) and N is a basis for W_H = ker(I_H).
    // In real applications, obtaining a basis for W_H is normally infeasible due to sparsity requirements.
    // However, since we only do small-scale tests here, we have the luxury of just simply building a dense
    // basis through the SVD.
    //
    // Our corrected basis is constructed by a localized approximation, so the above property does not hold exactly.
    // We can, however, choose an oversampling parameter so large that each corrector problem is a global problem.
    // In this case, the above property must still hold.

    // Make sure interior is not empty, because Eigen's SVD fails on empty matrices
    const auto coarse = *rc::gen::suchThat(crest::gen::arbitrary_unit_square_mesh(),
                                           [] (const auto & mesh) {
                                               return !mesh.compute_interior_vertices().empty();
                                           }).as("coarse mesh");
    const auto fine = *crest::gen::arbitrary_refinement(coarse).as("fine mesh");
    const auto biscale = crest::BiscaleMesh<double, int>(coarse, fine);

    const auto fine_interior = fine.compute_interior_vertices();
    const auto coarse_interior = coarse.compute_interior_vertices();
    const auto fine_basis = LagrangeBasis2d<double>(fine);

    const auto oversampling = static_cast<unsigned int>(coarse.num_vertices());
    const auto basis_weights = crest::SparseLuCorrectorSolver<double>()
            .compute_basis(biscale, oversampling).basis_weights();
    const auto basis_interior_weights = sparse_submatrix(basis_weights, coarse_interior, fine_interior);

    // Construct a basis for W_H, the kernel of I_H.
    const Eigen::MatrixXd I_H = sparse_submatrix(crest::quasi_interpolator(biscale), coarse_interior, fine_interior);
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(I_H, Eigen::ComputeFullV);
    const auto r = svd.rank();
    const auto n = I_H.cols();
    const Eigen::MatrixXd W_H_kernel_basis = svd.matrixV().rightCols(n - r);

    const Eigen::MatrixXd W = basis_interior_weights;
    const Eigen::MatrixXd A = sparse_submatrix(fine_basis.assemble().stiffness, fine_interior, fine_interior);
    const auto & N = W_H_kernel_basis;

    const Eigen::MatrixXd product = W * A * N;

    RC_LOG() << "W:   " << std::endl << W << std::endl << std::endl;
    RC_LOG() << "A:   " << std::endl << A << std::endl << std::endl;
    RC_LOG() << "N:   " << std::endl << N << std::endl << std::endl;
    RC_LOG() << "WAN: " << std::endl << product << std::endl << std::endl;

    RC_ASSERT(product.isZero(1e-14));
}

RC_GTEST_PROP(homogenized_basis_test, corrected_basis_is_orthogonal_to_fine_space_sparselu, ())
{
    check_corrected_basis_is_orthogonal_to_fine_space<crest::SparseLuCorrectorSolver<double>>();
}

RC_GTEST_PROP(homogenized_basis_test, corrected_basis_is_orthogonal_to_fine_space_schur, ())
{
    check_corrected_basis_is_orthogonal_to_fine_space<crest::SchurCorrectorSolver<double>>();
}

RC_GTEST_PROP(homogenized_basis_test, corrected_basis_is_orthogonal_to_fine_space_sparselu_dense_fallback, ())
{
    typedef crest::DenseFallbackCorrectorSolverWrapper<double, crest::SparseLuCorrectorSolver<double>> Wrapper;
    check_corrected_basis_is_orthogonal_to_fine_space<Wrapper>();
}

TEST(standard_coarse_basis_in_fine_space, basic_mesh)
{
    const std::vector<Vertex> coarse_vertices{
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(1.0, 1.0),
            Vertex(0.0, 1.0)
    };

    const std::vector<Element> coarse_elements {
            Element({3, 0, 1}),
            Element({1, 2, 3})
    };

    const std::vector<Vertex> fine_vertices {
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(1.0, 1.0),
            Vertex(0.0, 1.0),
            Vertex(0.5, 0.5)
    };

    const std::vector<Element> fine_elements {
            Element({0, 4, 3}),
            Element({2, 4, 1}),
            Element({1, 4, 0}),
            Element({2, 4, 3})
    };

    const std::vector<int> fine_ancestry { 0, 1, 0, 1 };

    const auto coarse_mesh = IndexedMesh<double, int>(coarse_vertices, coarse_elements);
    const auto fine_mesh = IndexedMesh<double, int>(fine_vertices, fine_elements, fine_ancestry);
    const auto biscale = crest::BiscaleMesh<double, int>(coarse_mesh, fine_mesh);

    const Eigen::Matrix<double, 4, 5> coarse_basis_in_fine =
            crest::detail::standard_coarse_basis_in_fine_space(biscale);

    Eigen::Matrix<double, 4, 5> expected;
    expected <<
             1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.5,
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.5;

    ASSERT_TRUE(coarse_basis_in_fine.isApprox(expected));
}

RC_GTEST_PROP(standard_coarse_basis_in_fine_space, quasi_interpolation_recovers_coarse_basis, ())
{
    // Let v_H be the weights of a function in the coarse space. Then v_H is also a function in the fine space,
    // and since I_H is a quasi interpolation, we have that I_H v_h = v_H, where v_h are the fine weights of
    // the coarse function.
    // We can verify that this property holds for all basis functions as represented in the fine space.

    const auto coarse = *crest::gen::arbitrary_unit_square_mesh().as("Coarse mesh");
    const auto fine = *crest::gen::arbitrary_refinement(coarse, 0, 5).as("Fine mesh");
    const auto biscale = crest::BiscaleMesh<double, int>(coarse, fine);

    const auto basis = crest::detail::standard_coarse_basis_in_fine_space(biscale);
    const auto I_H = crest::quasi_interpolator(biscale);

    const Eigen::SparseMatrix<double> interpolated = I_H * basis.transpose();
    const Eigen::MatrixXd interpolated_dense = interpolated;

    RC_LOG() << interpolated_dense << std::endl;

    // We could have had a shaper tolerance here, but it appears Eigen does not do an elementwise comparison
    // for isIdentity (which is very annoying, by the way), as it can spuriously fail even when all non-diagonal
    // elements are smaller than 3e-16.
    RC_ASSERT(interpolated_dense.isIdentity(1e-14));
}
