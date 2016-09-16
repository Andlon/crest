#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <crest/quadrature/triquad.hpp>
#include <crest/geometry/vertex.hpp>

#include <cstdint>
#include <cmath>

using crest::triquad;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::IsEmpty;
using ::testing::ElementsAreArray;
using ::testing::UnorderedElementsAreArray;

typedef crest::Vertex<double> Vertex;

constexpr static double TOLERANCE = 1e-13;

/**
 * Generates all permutations of a and b such that a + b = degree.
 * @param degree
 * @return A pair where first corresponds to a and seconds corresponds to b.
 */
std::vector<std::pair<unsigned int, unsigned int>> generate_polynomial_basis_coefficients_for_degree(unsigned int degree)
{
    std::vector<std::pair<unsigned int, unsigned int>> coefficient_pairs;
    for (unsigned int i = 0; i <= degree; ++i)
    {
        for (unsigned int j = degree - i; i + j == degree; ++j)
        {
            coefficient_pairs.push_back(std::make_pair(i, j));
        }
    }
    return coefficient_pairs;
}

/**
 * Returns the solution of the problem of integrating f(x, y) = x^a * y^b on the reference triangle
 * defined by (0, 0), (1, 0) and (0, 1).
 * @param a
 * @param b
 * @return
 */
double expected_solution_for_polynomial(uint64_t a, uint64_t b)
{
    return exp(lgamma(a + 1) + lgamma(b + 1) - lgamma(a + b + 3));
}

TEST(generate_polynomial_basis_coefficients_for_degree_test, degree_0)
{
    auto pairs = generate_polynomial_basis_coefficients_for_degree(0);
    auto expected = std::vector<std::pair<double, double>> { std::make_pair(0, 0) };
    EXPECT_THAT(pairs, ElementsAreArray(expected));
}

TEST(generate_polynomial_basis_coefficients_for_degree_test, degree_1)
{
    auto pairs = generate_polynomial_basis_coefficients_for_degree(1);
    auto expected = std::vector<std::pair<double, double>> {
            std::make_pair(1, 0),
            std::make_pair(0, 1)
    };
    EXPECT_THAT(pairs, UnorderedElementsAreArray(expected));
}

TEST(generate_polynomial_basis_coefficients_for_degree_test, degree_2)
{
    auto pairs = generate_polynomial_basis_coefficients_for_degree(2);
    auto expected = std::vector<std::pair<double, double>> {
            std::make_pair(1, 1),
            std::make_pair(2, 0),
            std::make_pair(0, 2)
    };
    EXPECT_THAT(pairs, UnorderedElementsAreArray(expected));
}

TEST(generate_polynomial_basis_coefficients_for_degree_test, degree_3)
{
    auto pairs = generate_polynomial_basis_coefficients_for_degree(3);
    auto expected = std::vector<std::pair<double, double>> {
            std::make_pair(1, 2),
            std::make_pair(2, 1),
            std::make_pair(3, 0),
            std::make_pair(0, 3)
    };
    EXPECT_THAT(pairs, UnorderedElementsAreArray(expected));
}

class triquad_test : public ::testing::Test
{
protected:
    triquad_test() : a(0.0, 0.0), b(0.0, 0.0), c(0.0, 0.0) {}

    virtual void SetUp()
    {
        a = Vertex(0.0, 0.0);
        b = Vertex(1.0, 0.0);
        c = Vertex(0.0, 1.0);
    }

    // Reference triangle
    Vertex a;
    Vertex b;
    Vertex c;

    std::function<double (double, double)> create_polynomial(unsigned int coeff_x, unsigned int coeff_y)
    {
        return [coeff_x, coeff_y] (auto x, auto y) { return pow(x, coeff_x) * pow(y, coeff_y); };
    }

};

TEST_F(triquad_test, identicially_zero)
{
    const auto f = [] (auto, auto) { return 0.0; };
    const auto sol = 0.0;

    EXPECT_THAT(triquad<1>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<2>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<4>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<5>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<6>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<7>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<8>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<9>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<10>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<10>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<11>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<12>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
}

TEST_F(triquad_test, identically_one)
{
    const auto f = [] (auto, auto) { return 1.0; };
    const auto sol = 0.5;

    EXPECT_THAT(triquad<1>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<2>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<4>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<5>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<6>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<7>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<8>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<9>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<10>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<10>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<11>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<12>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
}

TEST_F(triquad_test, polynomial_degree_1) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(1);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<1>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<2>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<4>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<5>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<6>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<7>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<8>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<9>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<10>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<11>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<12>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_2) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(2);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<2>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<4>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<5>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<6>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<7>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<8>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<9>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<10>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<11>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<12>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_3) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(3);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<4>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<5>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<6>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<7>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<8>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<9>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<10>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<11>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<12>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_4) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(4);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<4>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<5>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<6>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<7>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<8>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<9>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<10>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<11>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<12>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_5) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(5);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<5>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<6>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<7>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<8>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<9>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<10>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<11>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<12>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_6) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(6);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<6>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<7>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<8>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<9>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<10>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<11>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<12>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_7) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(7);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<7>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<8>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<9>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<10>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<11>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<12>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_8) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(8);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<8>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<9>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<10>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<11>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<12>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_9) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(9);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<9>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<10>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<11>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<12>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_10) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(10);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<10>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<11>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<12>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_11) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(11);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<11>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<12>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_12) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(12);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<12>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_13) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(13);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<13>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_14) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(14);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<14>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_15) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(15);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<15>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_16) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(16);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<16>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_17) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(17);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<17>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_18) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(18);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<18>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_19) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(19);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<19>(f, a, b, c), DoubleNear(sol, TOLERANCE));
        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, polynomial_degree_20) {
    auto coefficients = generate_polynomial_basis_coefficients_for_degree(20);

    for (auto coeffs : coefficients)
    {
        const auto f = create_polynomial(coeffs.first, coeffs.second);
        const auto sol = expected_solution_for_polynomial(coeffs.first, coeffs.second);

        EXPECT_THAT(triquad<20>(f, a, b, c), DoubleNear(sol, TOLERANCE));
    }
}

TEST_F(triquad_test, affine_function_on_arbitrary_triangle)
{
    // The polynomial degree tests verify that the routine correctly integrates polynomial basis functions
    // exactly up to the degree given by the strength of the quadrature, and as such it should be correct
    // for all polynomials of degree <= strength. However, we used the same reference triangle for all tests.
    // By choosing a completely arbitrary triangle for this test, we check that the coordinate transformation
    // is applied correctly, which together with the other tests practically almost completely verifies the correctness
    // of the implementation.

    const auto p = Vertex(-1.0, 1.0);
    const auto q = Vertex(3.0, 0.0);
    const auto r = Vertex(2.0, 2.0);

    // Thank Mathematica for computing the exact value of the integral!
    const auto f = [] (auto x, auto y) { return 3.0 * x + 15.0 * y - 8.0; };
    const auto sol = 77.0 / 2.0;

    EXPECT_THAT(triquad<1>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<2>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<4>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<5>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<6>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<7>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<8>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<9>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<10>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<11>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<12>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<13>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<14>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<15>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<16>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<17>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<18>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<19>(f, p, q, r), DoubleNear(sol, TOLERANCE));
    EXPECT_THAT(triquad<20>(f, p, q, r), DoubleNear(sol, TOLERANCE));
}