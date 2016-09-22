#include <crest/quadrature/simpsons.hpp>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <exception>

using ::testing::DoubleEq;
using ::testing::DoubleNear;

template <typename T>
std::vector<T> linspace(T a, T b, size_t num)
{
    if (num <= 1) throw std::invalid_argument("num must be greater than 1");

    std::vector<T> result;
    result.reserve(num);

    const auto h = b - a;

    for (size_t i = 0; i < num; ++i)
    {
        const auto fraction = static_cast<double>(i) / static_cast<double>(num - 1);
        result.emplace_back(a + fraction * h);
    }
    return result;
}

class simpsons_test_odd_samples : public ::testing::TestWithParam<unsigned int> {

};

class simpsons_test_even_samples : public ::testing::TestWithParam<unsigned int> {

};

TEST(simpsons_test, zero_samples_throws_exception)
{
    const std::vector<double> samples(0, 0.0);
    EXPECT_THROW(crest::composite_simpsons(samples.cbegin(), samples.cend(), 0.1), std::invalid_argument);
}

TEST_P(simpsons_test_even_samples, even_number_throws_exception)
{
    const auto num_samples = GetParam();
    const std::vector<double> samples(num_samples, 0.0);
    EXPECT_THROW(crest::composite_simpsons(samples.cbegin(), samples.cend(), 0.1), std::invalid_argument);
}

TEST_P(simpsons_test_odd_samples, zero_function)
{
    const auto num_samples = GetParam();
    const std::vector<double> samples(num_samples, 0.0);
    const auto integral = crest::composite_simpsons(samples.cbegin(), samples.cend(), 0.1);

    EXPECT_THAT(integral, DoubleEq(0.0));
}

TEST_P(simpsons_test_odd_samples, affine_function)
{
    const auto n = GetParam();

    const double a = 3.0;
    const double b = 7.0;
    const double dx = (b - a) / (n - 1);
    const auto f = [] (auto x) { return 3.0 * x - 4.0; };

    auto samples = linspace(a, b, n);
    std::transform(samples.begin(), samples.end(), samples.begin(), f);

    // Affine functions are expected to be integrated exactly!
    const auto integral = crest::composite_simpsons(samples.cbegin(), samples.cend(), dx);
    EXPECT_THAT(integral, DoubleEq(44.0));
}

TEST_P(simpsons_test_odd_samples, third_degree_polynomial)
{
    const auto n = GetParam();

    const double a = 3.0;
    const double b = 7.0;
    const double dx = (b - a) / (n - 1);
    const auto f = [] (auto x) { return 3.0 * x * x * x - x - 4.0; };

    auto samples = linspace(a, b, n);
    std::transform(samples.begin(), samples.end(), samples.begin(), f);

    // Third-degree polynomials can be integrated exactly!
    const auto integral = crest::composite_simpsons(samples.cbegin(), samples.cend(), dx);
    EXPECT_THAT(integral, DoubleEq(1704.0));
}

TEST_P(simpsons_test_odd_samples, trigonometric)
{
    const auto n = GetParam();

    const double a = 3.0;
    const double b = 7.0;
    const double dx = (b - a) / (n - 1);
    const auto f = [] (auto x) { return cos(2.0 * x); };

    auto samples = linspace(a, b, n);
    std::transform(samples.begin(), samples.end(), samples.begin(), f);

    // Allowable error is given by
    // dx^4 / 180 * (b - a) * max |f^4|,
    // where max f^4 is the fourth derivative of f, and max refers to the maximum on the given interval
    const auto max_error = pow(dx, 4.0) / 180.0 * (b - a) * 16.0;
    const auto exact_integral = -0.5 * sin(6) + cos(7) * sin(7.0);

    const auto integral = crest::composite_simpsons(samples.cbegin(), samples.cend(), dx);
    EXPECT_THAT(integral, DoubleNear(exact_integral, max_error));
}

INSTANTIATE_TEST_CASE_P(simpsons_test_odd,
                        simpsons_test_odd_samples,
                        ::testing::Values(3, 5, 19, 99));

INSTANTIATE_TEST_CASE_P(simpsons_test_even,
                        simpsons_test_even_samples,
                        ::testing::Values(2, 4, 10, 100));

