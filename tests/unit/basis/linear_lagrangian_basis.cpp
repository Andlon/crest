#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/basis/linear_lagrangian_basis.hpp>

#include <util/eigen_matchers.hpp>

using ::testing::ElementsAreArray;
using ::testing::Pointwise;
using ::testing::Eq;
using ::testing::DoubleEq;

using crest::IndexedMesh;
using crest::LagrangeBasis2d;

typedef IndexedMesh<double, int>::Vertex Vertex;
typedef IndexedMesh<double, int>::Element Element;

class linear_lagrangian_basis_test : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        vertices_unit_square_1_interior_node = {
                Vertex(0.0, 0.0),
                Vertex(1.0, 0.0),
                Vertex(1.0, 1.0),
                Vertex(0.0, 1.0),
                Vertex(0.5, 0.5)

        };

        elements_unit_square_1_interior_node = {
                Element({0, 1, 4}),
                Element({1, 2, 4}),
                Element({2, 3, 4}),
                Element({3, 0, 4})
        };

        vertices_unit_square_5_interior_nodes = {
                Vertex(0.0, 0.0),
                Vertex(0.5, 0.0),
                Vertex(1.0, 0.0),
                Vertex(0.75, 0.25),
                Vertex(0.25, 0.25),
                Vertex(0.0, 0.5),
                Vertex(0.5, 0.5),
                Vertex(1.0, 0.5),
                Vertex(0.75, 0.75),
                Vertex(0.25, 0.75),
                Vertex(0.0, 1.0),
                Vertex(0.5, 1.0),
                Vertex(1.0, 1.0)
        };

        elements_unit_square_5_interior_nodes = {
                Element({0, 4, 5}),
                Element({0, 1, 4}),
                Element({1, 3, 4}),
                Element({1, 2, 3}),
                Element({2, 7, 3}),
                Element({7, 8, 3}),
                Element({3, 8, 6}),
                Element({6, 4, 3}),
                Element({4, 6, 9}),
                Element({4, 9, 5}),
                Element({5, 9, 10}),
                Element({10, 9, 11}),
                Element({9, 8, 11}),
                Element({8, 12, 11}),
                Element({7, 12, 8}),
                Element({6, 8, 9})
        };
    }

    std::vector<Vertex>     vertices_unit_square_1_interior_node;
    std::vector<Element>    elements_unit_square_1_interior_node;

    // A triangulation of the unit square with a regular pattern consisting of 13 vertices and 15 triangles,
    // showcasing vertices both on the boundary and in the interior.
    std::vector<Vertex>     vertices_unit_square_5_interior_nodes;
    std::vector<Element>    elements_unit_square_5_interior_nodes;


};

TEST_F(linear_lagrangian_basis_test, assemble_mass_matrix_1_interior_node)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_1_interior_node, elements_unit_square_1_interior_node);
    const auto basis = LagrangeBasis2d::assemble_from_mesh(mesh);

    EXPECT_THAT(basis.mass_matrix().rows(), Eq(1));
    EXPECT_THAT(basis.mass_matrix().cols(), Eq(1));
    EXPECT_THAT(basis.mass_matrix().coeff(0, 0), DoubleEq(1.0 / 6.0));
}

TEST_F(linear_lagrangian_basis_test, assemble_stiffness_matrix_1_interior_node)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_1_interior_node, elements_unit_square_1_interior_node);
    const auto basis = LagrangeBasis2d::assemble_from_mesh(mesh);

    EXPECT_THAT(basis.stiffness_matrix().rows(), Eq(1));
    EXPECT_THAT(basis.stiffness_matrix().cols(), Eq(1));
    EXPECT_THAT(basis.stiffness_matrix().coeff(0, 0), DoubleEq(4.0));
}

TEST_F(linear_lagrangian_basis_test, assemble_boundary_mass_matrix_1_interior_node)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_1_interior_node, elements_unit_square_1_interior_node);
    const auto basis = LagrangeBasis2d::assemble_from_mesh(mesh);

    Eigen::Matrix<double, 1, 4> expected;
    expected << (1.0 / 24.0), (1.0 / 24.0), (1.0 / 24.0), (1.0 / 24.0);

    EXPECT_THAT(basis.boundary_mass_matrix().rows(), Eq(1));
    EXPECT_THAT(basis.boundary_mass_matrix().cols(), Eq(4));
    EXPECT_THAT(basis.boundary_mass_matrix(), MatrixEq(expected));
}

TEST_F(linear_lagrangian_basis_test, assemble_boundary_stiffness_matrix_1_interior_node)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_1_interior_node, elements_unit_square_1_interior_node);
    const auto basis = LagrangeBasis2d::assemble_from_mesh(mesh);

    Eigen::Matrix<double, 1, 4> expected;
    expected << -1.0, -1.0, -1.0, -1.0;

    EXPECT_THAT(basis.boundary_stiffness_matrix().rows(), Eq(1));
    EXPECT_THAT(basis.boundary_stiffness_matrix().cols(), Eq(4));
    EXPECT_THAT(basis.boundary_stiffness_matrix(), MatrixEq(expected));
}

TEST_F(linear_lagrangian_basis_test, assemble_mass_matrix_5_interior_nodes)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_5_interior_nodes, elements_unit_square_5_interior_nodes);
    const auto basis = LagrangeBasis2d::assemble_from_mesh(mesh);

    // The expected data is obtained through laborious semi-manual calculations,
    // exploiting symmetry of the mesh to reduce work. I set up the integrals,
    // and solved them with Mathematica.
    const auto a = 1.0 / 16.0;
    const auto b = 1.0 / 96.0;
    const auto c = 1.0 / 96.0;
    const auto d = 1.0 / 24.0;

    Eigen::Matrix<double, 5, 5> expected;
    expected <<
             a,    b,  c,    b,  0.0,
            b,     a,  c,  0.0,    b,
            c,     c,  d,    c,    c,
            b,   0.0,  c,    a,    b,
            0.0,   b,  c,    b,    a;

    EXPECT_THAT(basis.mass_matrix().rows(), Eq(5));
    EXPECT_THAT(basis.mass_matrix().cols(), Eq(5));
    EXPECT_THAT(basis.mass_matrix(), MatrixEq(expected));
}

TEST_F(linear_lagrangian_basis_test, assemble_stiffness_matrix_5_interior_nodes)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_5_interior_nodes, elements_unit_square_5_interior_nodes);
    const auto basis = LagrangeBasis2d::assemble_from_mesh(mesh);

    // The expected data is obtained through laborious semi-manual calculations,
    // exploiting symmetry of the mesh to reduce work. I set up the integrals,
    // and solved them with Mathematica.
    const auto a = 4.0;
    const auto b = 0.0;
    const auto c = -1.0;
    const auto d = 4.0;

    Eigen::Matrix<double, 5, 5> expected;
    expected <<
             a,    b,  c,    b,  0.0,
            b,     a,  c,  0.0,    b,
            c,     c,  d,    c,    c,
            b,   0.0,  c,    a,    b,
            0.0,   b,  c,    b,    a;

    EXPECT_THAT(basis.stiffness_matrix().rows(), Eq(5));
    EXPECT_THAT(basis.stiffness_matrix().cols(), Eq(5));
    EXPECT_THAT(basis.stiffness_matrix(), MatrixEq(expected));
}

TEST_F(linear_lagrangian_basis_test, assemble_boundary_mass_matrix_5_interior_nodes)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_5_interior_nodes, elements_unit_square_5_interior_nodes);
    const auto basis = LagrangeBasis2d::assemble_from_mesh(mesh);

    // The expected data is obtained through laborious semi-manual calculations,
    // exploiting symmetry of the mesh to reduce work. I set up the integrals,
    // and solved them with Mathematica.
    const auto a = 1.0 / 96.0;
    const auto b = 1.0 / 96.0;

    Eigen::Matrix<double, 5, 8> expected;
    expected.setZero();
    expected(0, 1) = b;
    expected(0, 2) = a;
    expected(0, 4) = b;
    expected(1, 0) = a;
    expected(1, 1) = b;
    expected(1, 3) = b;
    expected(3, 4) = b;
    expected(3, 6) = b;
    expected(3, 7) = a;
    expected(4, 3) = b;
    expected(4, 5) = a;
    expected(4, 6) = b;

    EXPECT_THAT(basis.boundary_mass_matrix().rows(), Eq(5));
    EXPECT_THAT(basis.boundary_mass_matrix().cols(), Eq(8));
    EXPECT_THAT(basis.boundary_mass_matrix(), MatrixEq(expected));
}

TEST_F(linear_lagrangian_basis_test, assemble_boundary_stiffness_matrix_5_interior_nodes)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_5_interior_nodes, elements_unit_square_5_interior_nodes);
    const auto basis = LagrangeBasis2d::assemble_from_mesh(mesh);

    // The expected data is obtained through laborious semi-manual calculations,
    // exploiting symmetry of the mesh to reduce work. I set up the integrals,
    // and solved them with Mathematica.
    const auto a = -1.0;
    const auto b = -1.0;

    Eigen::Matrix<double, 5, 8> expected;
    expected.setZero();
    expected(0, 1) = b;
    expected(0, 2) = a;
    expected(0, 4) = b;
    expected(1, 0) = a;
    expected(1, 1) = b;
    expected(1, 3) = b;
    expected(3, 4) = b;
    expected(3, 6) = b;
    expected(3, 7) = a;
    expected(4, 3) = b;
    expected(4, 5) = a;
    expected(4, 6) = b;

    EXPECT_THAT(basis.boundary_stiffness_matrix().rows(), Eq(5));
    EXPECT_THAT(basis.boundary_stiffness_matrix().cols(), Eq(8));
    EXPECT_THAT(basis.boundary_stiffness_matrix(), MatrixEq(expected));
}

TEST_F(linear_lagrangian_basis_test, compute_load_zero_1_interior_node)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_1_interior_node, elements_unit_square_1_interior_node);
    const auto basis = LagrangeBasis2d::assemble_from_mesh(mesh);

    const auto f = [] (auto, auto) { return 0.0; };

    Eigen::VectorXd expected(1);
    expected.setZero();

    EXPECT_THAT(basis.compute_load<1>(f), MatrixEq(expected));
}

TEST_F(linear_lagrangian_basis_test, compute_load_zero_5_interior_nodes)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_5_interior_nodes, elements_unit_square_5_interior_nodes);
    const auto basis = LagrangeBasis2d::assemble_from_mesh(mesh);

    const auto f = [] (auto, auto) { return 0.0; };

    Eigen::VectorXd expected(5);
    expected.setZero();

    EXPECT_THAT(basis.compute_load<1>(f), MatrixEq(expected));
}

TEST_F(linear_lagrangian_basis_test, compute_load_one_1_interior_node)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_1_interior_node, elements_unit_square_1_interior_node);
    const auto basis = LagrangeBasis2d::assemble_from_mesh(mesh);

    const auto f = [] (auto, auto) { return 1.0; };

    Eigen::VectorXd expected(1);
    expected << (1.0 / 3.0);

    EXPECT_THAT(basis.compute_load<1>(f), MatrixEq(expected));
}

TEST_F(linear_lagrangian_basis_test, compute_quadratic_function_5_interior_node)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_5_interior_nodes, elements_unit_square_5_interior_nodes);
    const auto basis = LagrangeBasis2d::assemble_from_mesh(mesh);

    const auto f = [] (auto x, auto y) { return (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) - 2; };

    const auto a = -15.0 / 64.0;
    const auto b = -79.0 / 480.0;
    Eigen::VectorXd expected(5);
    expected << a, a, b, a, a;

    EXPECT_THAT(basis.compute_load<10>(f), MatrixEq(expected));
}
