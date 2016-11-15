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
        vertices_unit_square_basic = {
                Vertex(0.0, 0.0),
                Vertex(0.0, 1.0),
                Vertex(1.0, 1.0),
                Vertex(1.0, 0.0)
        };

        elements_unit_square_basic = {
                Element({3, 0, 1}),
                Element({1, 2, 3})
        };

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

    // The simplest possible triangulation of the unit square
    std::vector<Vertex>     vertices_unit_square_basic;
    std::vector<Element>    elements_unit_square_basic;

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
    const auto basis = LagrangeBasis2d<double>(mesh);
    const auto assembly = basis.assemble();

    // For historical reasons, there's only the values for entries of the
    // "interior" and "boundary" parts of the matrix available
    const auto interior = mesh.compute_interior_vertices();
    const auto interior_mass = sparse_submatrix(assembly.mass, interior, interior);

    EXPECT_THAT(interior_mass.rows(), Eq(1));
    EXPECT_THAT(interior_mass.cols(), Eq(1));
    EXPECT_THAT(interior_mass.coeff(0, 0), DoubleEq(1.0 / 6.0));

    const auto boundary = mesh.boundary_vertices();
    const auto boundary_mass = sparse_submatrix(assembly.mass, interior, boundary);

    Eigen::Matrix<double, 1, 4> expected_boundary_mass;
    expected_boundary_mass << (1.0 / 24.0), (1.0 / 24.0), (1.0 / 24.0), (1.0 / 24.0);

    EXPECT_THAT(boundary_mass.rows(), Eq(1));
    EXPECT_THAT(boundary_mass.cols(), Eq(4));
    EXPECT_THAT(boundary_mass, MatrixEq(expected_boundary_mass));
}

TEST_F(linear_lagrangian_basis_test, assemble_stiffness_matrix_1_interior_node)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_1_interior_node, elements_unit_square_1_interior_node);
    const auto basis = LagrangeBasis2d<double>(mesh);
    const auto assembly = basis.assemble();

    // For historical reasons, there's only the values for entries of the
    // "interior" and "boundary" parts of the matrix available
    const auto interior = mesh.compute_interior_vertices();
    const auto interior_stiffness = sparse_submatrix(assembly.stiffness, interior, interior);

    EXPECT_THAT(interior_stiffness.rows(), Eq(1));
    EXPECT_THAT(interior_stiffness.cols(), Eq(1));
    EXPECT_THAT(interior_stiffness.coeff(0, 0), DoubleEq(4.0));

    const auto boundary = mesh.boundary_vertices();
    const auto boundary_stiffness = sparse_submatrix(assembly.stiffness, interior, boundary);

    Eigen::Matrix<double, 1, 4> expected_boundary_stiffness;
    expected_boundary_stiffness << -1.0, -1.0, -1.0, -1.0;

    EXPECT_THAT(boundary_stiffness.rows(), Eq(1));
    EXPECT_THAT(boundary_stiffness.cols(), Eq(4));
    EXPECT_THAT(boundary_stiffness, MatrixEq(expected_boundary_stiffness));
}


TEST_F(linear_lagrangian_basis_test, assemble_mass_matrix_5_interior_nodes)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_5_interior_nodes, elements_unit_square_5_interior_nodes);
    const auto basis = LagrangeBasis2d<double>(mesh);
    const auto assembly = basis.assemble();

    const auto interior = mesh.compute_interior_vertices();
    const auto boundary = mesh.boundary_vertices();

    // For historical reasons, there's only the values for entries of the
    // "interior" and "boundary" parts of the matrix available

    {
        const auto interior_mass = sparse_submatrix(assembly.mass, interior, interior);

        // The expected data is obtained through laborious semi-manual calculations,
        // exploiting symmetry of the mesh to reduce work. I set up the integrals,
        // and solved them with Mathematica.
        const auto a = 1.0 / 16.0;
        const auto b = 1.0 / 96.0;
        const auto c = 1.0 / 96.0;
        const auto d = 1.0 / 24.0;

        Eigen::Matrix<double, 5, 5> expected_interior;
        expected_interior <<
                          a,    b,  c,    b,  0.0,
                b,     a,  c,  0.0,    b,
                c,     c,  d,    c,    c,
                b,   0.0,  c,    a,    b,
                0.0,   b,  c,    b,    a;

        EXPECT_THAT(interior_mass.rows(), Eq(5));
        EXPECT_THAT(interior_mass.cols(), Eq(5));
        EXPECT_THAT(interior_mass, MatrixEq(expected_interior));
    }

    {
        const auto boundary_mass = sparse_submatrix(assembly.mass, interior, boundary);

        const auto a = 1.0 / 96.0;
        const auto b = 1.0 / 96.0;

        Eigen::Matrix<double, 5, 8> expected_boundary;
        expected_boundary.setZero();
        expected_boundary(0, 1) = b;
        expected_boundary(0, 2) = a;
        expected_boundary(0, 4) = b;
        expected_boundary(1, 0) = a;
        expected_boundary(1, 1) = b;
        expected_boundary(1, 3) = b;
        expected_boundary(3, 4) = b;
        expected_boundary(3, 6) = b;
        expected_boundary(3, 7) = a;
        expected_boundary(4, 3) = b;
        expected_boundary(4, 5) = a;
        expected_boundary(4, 6) = b;

        EXPECT_THAT(boundary_mass.rows(), Eq(5));
        EXPECT_THAT(boundary_mass.cols(), Eq(8));
        EXPECT_THAT(boundary_mass, MatrixEq(expected_boundary));
    }
}

TEST_F(linear_lagrangian_basis_test, assemble_stiffness_matrix_5_interior_nodes)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_5_interior_nodes, elements_unit_square_5_interior_nodes);
    const auto basis = LagrangeBasis2d<double>(mesh);
    const auto assembly = basis.assemble();

    const auto interior = mesh.compute_interior_vertices();
    const auto boundary = mesh.boundary_vertices();

    // For historical reasons, there's only the values for entries of the
    // "interior" and "boundary" parts of the matrix available

    // The expected data is obtained through laborious semi-manual calculations,
    // exploiting symmetry of the mesh to reduce work. I set up the integrals,
    // and solved them with Mathematica.

    {
        const auto interior_stiffness = sparse_submatrix(assembly.stiffness, interior, interior);

        const auto a = 4.0;
        const auto b = 0.0;
        const auto c = -1.0;
        const auto d = 4.0;

        Eigen::Matrix<double, 5, 5> expected_interior;
        expected_interior <<
                          a,    b,  c,    b,  0.0,
                b,     a,  c,  0.0,    b,
                c,     c,  d,    c,    c,
                b,   0.0,  c,    a,    b,
                0.0,   b,  c,    b,    a;

        EXPECT_THAT(interior_stiffness.rows(), Eq(5));
        EXPECT_THAT(interior_stiffness.cols(), Eq(5));
        EXPECT_THAT(interior_stiffness, MatrixEq(expected_interior));
    }

    {
        const auto boundary_stiffness = sparse_submatrix(assembly.stiffness, interior, boundary);
        const auto a = -1.0;
        const auto b = -1.0;

        Eigen::Matrix<double, 5, 8> expected_boundary;
        expected_boundary.setZero();
        expected_boundary(0, 1) = b;
        expected_boundary(0, 2) = a;
        expected_boundary(0, 4) = b;
        expected_boundary(1, 0) = a;
        expected_boundary(1, 1) = b;
        expected_boundary(1, 3) = b;
        expected_boundary(3, 4) = b;
        expected_boundary(3, 6) = b;
        expected_boundary(3, 7) = a;
        expected_boundary(4, 3) = b;
        expected_boundary(4, 5) = a;
        expected_boundary(4, 6) = b;

        EXPECT_THAT(boundary_stiffness.rows(), Eq(5));
        EXPECT_THAT(boundary_stiffness.cols(), Eq(8));
        EXPECT_THAT(boundary_stiffness, MatrixEq(expected_boundary));
    }


}


TEST_F(linear_lagrangian_basis_test, compute_load_zero_1_interior_node)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_1_interior_node, elements_unit_square_1_interior_node);
    const auto basis = LagrangeBasis2d<double>(mesh);

    const auto f = [] (auto, auto) { return 0.0; };

    Eigen::VectorXd expected(5);
    expected.setZero();

    EXPECT_THAT(basis.load<1>(f), MatrixEq(expected));
}

TEST_F(linear_lagrangian_basis_test, compute_load_zero_5_interior_nodes)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_5_interior_nodes, elements_unit_square_5_interior_nodes);
    const auto basis = LagrangeBasis2d<double>(mesh);

    const auto f = [] (auto, auto) { return 0.0; };

    Eigen::VectorXd expected(13);
    expected.setZero();

    EXPECT_THAT(basis.load<1>(f), MatrixEq(expected));
}

TEST_F(linear_lagrangian_basis_test, compute_load_one_1_interior_node)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_1_interior_node, elements_unit_square_1_interior_node);
    const auto basis = LagrangeBasis2d<double>(mesh);
    const auto interior = mesh.compute_interior_vertices();

    const auto f = [] (auto, auto) { return 1.0; };

    // For historical reasons, we only have expected values for the interior nodes
    const auto load = basis.load<1>(f);
    const auto load_interior = submatrix(load, interior, { 0 });

    Eigen::VectorXd expected(1);
    expected << (1.0 / 3.0);

    EXPECT_THAT(load_interior, MatrixEq(expected));
}

TEST_F(linear_lagrangian_basis_test, compute_quadratic_function_5_interior_node)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_5_interior_nodes, elements_unit_square_5_interior_nodes);
    const auto basis = LagrangeBasis2d<double>(mesh);
    const auto interior = mesh.compute_interior_vertices();

    const auto f = [] (auto x, auto y) { return (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) - 2; };

    const auto load = basis.load<10>(f);
    const auto load_interior = submatrix(load, interior, { 0 });

    const auto a = -15.0 / 64.0;
    const auto b = -79.0 / 480.0;
    Eigen::VectorXd expected(5);
    expected << a, a, b, a, a;

    EXPECT_THAT(load_interior, MatrixEq(expected));
}

TEST_F(linear_lagrangian_basis_test, interpolate_1_interior_node)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_1_interior_node, elements_unit_square_1_interior_node);
    const auto basis = LagrangeBasis2d<double>(mesh);

    const auto f = [] (auto x, auto y) { return x * x + y * y + 1.0; };

    const auto weights = basis.interpolate(f);

    EXPECT_THAT(weights(0), DoubleEq(1.0));
    EXPECT_THAT(weights(1), DoubleEq(2.0));
    EXPECT_THAT(weights(2), DoubleEq(3.0));
    EXPECT_THAT(weights(3), DoubleEq(2.0));
    EXPECT_THAT(weights(4), DoubleEq(1.5));
}

TEST_F(linear_lagrangian_basis_test, error_l2_basic_unit_square)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_basic, elements_unit_square_basic);
    const auto basis = LagrangeBasis2d<double>(mesh);

    const auto u = [] (auto x, auto y) { return x * x + y * y; };

    VectorX<double> weights(4);
    weights << 0.0, 1.0, 2.0, 1.0;

    const auto expected_error = std::sqrt(11.0 / 90.0);

    EXPECT_THAT(basis.error_l2<10>(u, weights), DoubleEq(expected_error));
}

TEST_F(linear_lagrangian_basis_test, error_h1_semi_basic_unit_square)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_basic, elements_unit_square_basic);
    const auto basis = LagrangeBasis2d<double>(mesh);

    const auto u_x = [] (auto x, auto  ) { return 2.0 * x; };
    const auto u_y = [] (auto  , auto y) { return 2.0 * y; };

    VectorX<double> weights(4);
    weights << 0.0, 1.0, 2.0, 1.0;

    const auto expected_error = std::sqrt(2.0 / 3.0);

    EXPECT_THAT(basis.error_h1_semi<10>(u_x, u_y, weights), DoubleEq(expected_error));
}

TEST_F(linear_lagrangian_basis_test, error_h1_basic_unit_square)
{
    const auto mesh = IndexedMesh<double, int>(vertices_unit_square_basic, elements_unit_square_basic);
    const auto basis = LagrangeBasis2d<double>(mesh);

    const auto u = [] (auto x, auto y) { return x * x + y * y; };
    const auto u_x = [] (auto x, auto  ) { return 2.0 * x; };
    const auto u_y = [] (auto  , auto y) { return 2.0 * y; };

    VectorX<double> weights(4);
    weights << 0.0, 1.0, 2.0, 1.0;

    const auto expected_error = std::sqrt(11.0 / 90.0 + 2.0 / 3.0);

    EXPECT_THAT(basis.error_h1<10>(u, u_x, u_y, weights), DoubleEq(expected_error));
}
