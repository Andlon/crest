#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <crest/util/eigen_extensions.hpp>
#include <util/eigen_matchers.hpp>

#include <Eigen/Sparse>

using ::testing::Eq;

class sparse_submatrix_test : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        zero_10x10 = Eigen::SparseMatrix<int>(10, 10);

        diagonal_10x10 = Eigen::SparseMatrix<int>(10, 10);
        for (int i = 0; i < 10; ++i)
        {
            diagonal_10x10.insert(i, i) = i;
        }

        dense_5x5 = Eigen::SparseMatrix<int>(5, 5);
        for (int i = 0; i < 5; ++i)
        {
            for (int j = 0; j < 5; ++j)
            {
                dense_5x5.insert(i, j) = j + 5 * i;
            }
        }
    }

    Eigen::SparseMatrix<int> zero_10x10;
    Eigen::SparseMatrix<int> diagonal_10x10;
    Eigen::SparseMatrix<int> dense_5x5;
};

class submatrix_test : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        dense_5x5 << 0, 1, 2, 3, 4,
                5, 6, 7, 8, 9,
                10, 11, 12, 13, 14,
                15, 16, 17, 18, 19,
                20, 21, 22, 23, 24;
    }

    Eigen::Matrix<int, 5, 5> dense_5x5;
};

TEST_F(sparse_submatrix_test, zero_10x10)
{
    const auto rows = std::vector<int> { 3, 5, 7 };
    const auto cols = std::vector<int> { 0, 9 };
    const auto mat = sparse_submatrix(zero_10x10, rows, cols);

    EXPECT_THAT(mat.rows(), Eq(3));
    EXPECT_THAT(mat.cols(), Eq(2));
    EXPECT_THAT(mat.nonZeros(), Eq(0));
}

TEST_F(sparse_submatrix_test, diagonal_10x10)
{
    const auto rows = std::vector<int> { 3, 5, 7 };
    const auto cols = std::vector<int> { 3, 4, 5 };
    const auto mat = sparse_submatrix(diagonal_10x10, rows, cols);

    EXPECT_THAT(mat.rows(), Eq(3));
    EXPECT_THAT(mat.cols(), Eq(3));
    EXPECT_THAT(mat.nonZeros(), Eq(2));

    EXPECT_THAT(mat.coeff(0, 0), Eq(3));
    EXPECT_THAT(mat.coeff(1, 2), Eq(5));
}

TEST_F(sparse_submatrix_test, dense_5x5_arbitrary_indices)
{
    const auto rows = std::vector<int> { 1, 2, 4 };
    const auto cols = std::vector<int> { 2, 4 };
    const auto mat = sparse_submatrix(dense_5x5, rows, cols);

    EXPECT_THAT(mat.rows(), Eq(3));
    EXPECT_THAT(mat.cols(), Eq(2));
    EXPECT_THAT(mat.nonZeros(), Eq(6));

    Eigen::Matrix<int, 3, 2> expected;
    expected <<
             7, 9,
            12, 14,
            22, 24;

    EXPECT_THAT(Eigen::MatrixXi(mat), MatrixEq(expected));
}

TEST_F(sparse_submatrix_test, dense_5x5_all_indices)
{
    const auto rows = std::vector<int> { 0, 1, 2, 3, 4 };
    const auto cols = std::vector<int> { 0, 1, 2, 3, 4 };
    const auto mat = sparse_submatrix(dense_5x5, rows, cols);

    EXPECT_THAT(mat.rows(), Eq(5));
    EXPECT_THAT(mat.cols(), Eq(5));
    EXPECT_THAT(mat.nonZeros(), Eq(25));

    const Eigen::MatrixXi expected = Eigen::MatrixXi(dense_5x5);

    EXPECT_THAT(Eigen::MatrixXi(mat), MatrixEq(expected));
}

TEST_F(submatrix_test, dense_5x5)
{
    const auto rows = std::vector<int> { 1, 2, 4 };
    const auto cols = std::vector<int> { 2, 4 };
    const auto mat = submatrix(dense_5x5, rows, cols);

    Eigen::Matrix<int, 3, 2> expected;
    expected <<
             7, 9,
            12, 14,
            22, 24;

    EXPECT_THAT(mat, MatrixEq(expected));
}

TEST_F(submatrix_test, dense_5x5_all_indices)
{
    const auto rows = std::vector<int> { 0, 1, 2, 3, 4 };
    const auto cols = std::vector<int> { 0, 1, 2, 3, 4 };
    const auto mat = submatrix(dense_5x5, rows, cols);

    const Eigen::MatrixXi expected = Eigen::MatrixXi(dense_5x5);

    EXPECT_THAT(mat, MatrixEq(expected));
}

TEST_F(sparse_submatrix_test, sparsity_pattern_zero_10x10) {
    const auto reverse_rows = std::vector<int> { -1, -1, -1, 0, -1, 1, -1, 2, -1, -1, -1};
    const auto cols = std::vector<int> { 0, 9 };
    const auto pattern = detail::submatrix_sparsity_pattern(zero_10x10, reverse_rows, cols);

    EXPECT_TRUE(pattern.isZero());
    EXPECT_THAT(pattern.rows(), Eq(2));
}

TEST_F(sparse_submatrix_test, sparsity_pattern_diagonal_10x10) {
    const auto reverse_rows = std::vector<int> { -1, -1, -1, 0, -1, 1, -1, 2, -1, -1, -1};
    const auto cols = std::vector<int> { 3, 4, 5 };
    const auto pattern = detail::submatrix_sparsity_pattern(diagonal_10x10, reverse_rows, cols);

    EXPECT_THAT(pattern.rows(), Eq(3));
    EXPECT_THAT(pattern(0), 1);
    EXPECT_THAT(pattern(1), 0);
    EXPECT_THAT(pattern(2), 1);
}

TEST_F(sparse_submatrix_test, sparsity_pattern_dense_5x5_arbitrary_indices)
{
    const auto reverse_rows = std::vector<int> { -1, 0, 0, -1, 2 };
    const auto cols = std::vector<int> { 2, 4 };
    const auto pattern = detail::submatrix_sparsity_pattern(dense_5x5, reverse_rows, cols);

    EXPECT_THAT(pattern.rows(), Eq(2));
    EXPECT_THAT(pattern(0), 3);
    EXPECT_THAT(pattern(1), 3);
}

TEST_F(sparse_submatrix_test, sparsity_pattern_dense_5x5_all_indices)
{
    const auto reverse_rows = std::vector<int> { 0, 1, 2, 3, 4 };
    const auto cols = std::vector<int> { 0, 1, 2, 3, 4 };
    const auto pattern = detail::submatrix_sparsity_pattern(dense_5x5, reverse_rows, cols);

    EXPECT_THAT(pattern.rows(), Eq(5));
    EXPECT_THAT(pattern(0), Eq(5));
    EXPECT_THAT(pattern(1), Eq(5));
    EXPECT_THAT(pattern(2), Eq(5));
    EXPECT_THAT(pattern(3), Eq(5));
    EXPECT_THAT(pattern(4), Eq(5));
}
