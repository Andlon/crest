#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <rapidcheck/gtest.h>

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/geometry/biscale_mesh.hpp>
#include <crest/geometry/refinement.hpp>
#include <crest/basis/quasi_interpolation.hpp>

#include <util/vertex_matchers.hpp>
#include <util/test_generators.hpp>
#include <util/eigen_matchers.hpp>

#include <iostream>

using ::testing::ElementsAreArray;
using ::testing::Pointwise;
using ::testing::Eq;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Value;

typedef crest::IndexedMesh<double, int> IndexedMesh;
typedef IndexedMesh::Vertex Vertex;
typedef IndexedMesh::Element Element;

TEST(affine_interpolator_test, minimal_mesh)
{
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

    const auto coarse_mesh = IndexedMesh(coarse_vertices, coarse_elements);
    const auto fine_mesh = crest::bisect_to_tolerance(coarse_mesh, 1.1);
    const auto biscale = crest::BiscaleMesh<double, int>(coarse_mesh, fine_mesh);

    ASSERT_THAT(fine_mesh.num_vertices(), Eq(5));
    ASSERT_THAT(fine_mesh.num_elements(), Eq(4));
    ASSERT_THAT(fine_mesh.vertices()[4], VertexDoubleEq(Vertex(0.5, 0.5)));

    const Eigen::SparseMatrix<double, Eigen::RowMajor> interpolator =
            crest::detail::affine_interpolator(biscale);

    EXPECT_THAT(interpolator.rows(), Eq(6));
    EXPECT_THAT(interpolator.cols(), Eq(5));

    Eigen::Matrix<double, 5, 1> weights;
    weights << 1.0, 2.0, 3.0, 4.0, 5.0;

    const Eigen::Matrix<double, 6, 1> interpolated_weights = interpolator * weights;
    EXPECT_THAT(interpolated_weights(0), DoubleNear(5.0, 1e-14));
    EXPECT_THAT(interpolated_weights(1), DoubleNear(1.0, 1e-14));
    EXPECT_THAT(interpolated_weights(2), DoubleNear(3.0, 1e-14));
    EXPECT_THAT(interpolated_weights(3), DoubleNear(3.0, 1e-14));
    EXPECT_THAT(interpolated_weights(4), DoubleNear(3.0, 1e-14));
    EXPECT_THAT(interpolated_weights(5), DoubleNear(5.0, 1e-14));
}

TEST(nodal_average_interpolator_test, minimal_mesh)
{
    const std::vector<Vertex> vertices {
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(1.0, 1.0),
            Vertex(0.0, 1.0)
    };

    const std::vector<Element> elements {
            Element({3, 0, 1}),
            Element({1, 2, 3})
    };

    const auto mesh = IndexedMesh(vertices, elements);
    const Eigen::SparseMatrix<double> interpolator = crest::detail::nodal_average_interpolator(mesh);

    EXPECT_THAT(interpolator.rows(), Eq(4));
    EXPECT_THAT(interpolator.cols(), Eq(6));

    Eigen::VectorXd affine_weights(6);
    affine_weights << 5.0, 1.0, 3.0, 3.0, 3.0, 5.0;

    const Eigen::VectorXd standard_weights = interpolator * affine_weights;
    EXPECT_THAT(standard_weights(0), DoubleEq(1.0));
    EXPECT_THAT(standard_weights(1), DoubleEq(3.0));
    EXPECT_THAT(standard_weights(2), DoubleEq(3.0));
    EXPECT_THAT(standard_weights(3), DoubleEq(5.0));
}

TEST(quasi_interpolator, verify_successful_compilation)
{
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

    const auto coarse_mesh = IndexedMesh(coarse_vertices, coarse_elements);
    const auto fine_mesh = crest::bisect_to_tolerance(coarse_mesh, 1.1);
    const auto biscale = crest::BiscaleMesh<double, int>(coarse_mesh, fine_mesh);

    ASSERT_THAT(fine_mesh.num_vertices(), Eq(5));
    ASSERT_THAT(fine_mesh.num_elements(), Eq(4));
    ASSERT_THAT(fine_mesh.vertices()[4], VertexDoubleEq(Vertex(0.5, 0.5)));

    const auto I_H = crest::quasi_interpolator(biscale);
    EXPECT_THAT(I_H.rows(), Eq(coarse_mesh.num_vertices()));
    EXPECT_THAT(I_H.cols(), Eq(fine_mesh.num_vertices()));
}


RC_GTEST_PROP(quasi_interpolator_test, interpolator_is_identity_map_for_equal_coarse_and_fine, ())
{
    const auto mesh = *crest::gen::arbitrary_unit_square_mesh();
    const auto biscale = crest::BiscaleMesh<double, int>(mesh, mesh);
    const auto N = mesh.num_vertices();
    const MatrixX<double> identity = MatrixX<double>::Identity(N, N);
    const MatrixX<double> I_H = crest::quasi_interpolator(biscale);

    RC_ASSERT(I_H.isApprox(identity));
}
