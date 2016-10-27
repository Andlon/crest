#pragma once

#include <Eigen/Sparse>
#include <cassert>
#include <algorithm>
#include <ostream>


template <typename Scalar>
using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar, typename Index>
Index max_nnz_in_cols(const Eigen::SparseMatrix<Scalar, 0, Index> & matrix)
{
    Index max_nnz = 0;
    for (Index i = 0; i < matrix.cols(); ++i)
    {
        const auto nnz = matrix.col(i).nonZeros();
        if (nnz > max_nnz) max_nnz = nnz;
    }
    return max_nnz;
}

template <typename Scalar, typename Index>
Eigen::SparseMatrix<Scalar, 0, Index> sparse_submatrix(const Eigen::SparseMatrix<Scalar, 0, Index> & matrix,
                                                       const std::vector<Index> & rows,
                                                       const std::vector<Index> & cols)
{
    assert(std::is_sorted(rows.cbegin(), rows.cend()) && "Row indices must be sorted.");
    assert(std::is_sorted(cols.cbegin(), cols.cend()) && "Column indices must be sorted.");


    Eigen::SparseMatrix<Scalar, 0, Index> submat(static_cast<Index>(rows.size()), static_cast<Index>(cols.size()));


    // TODO: Determine vector of exact number of nnz in each column, so we can more accurately reserve
    // the appropriate amount of nnzs in submat
    //const auto largest_column_nnz_number = max_nnz_in_cols(matrix);

    // TODO: Reserve is crashing on an assertion. Determine why.
    //submat.reserve(Eigen::VectorXi::Constant(static_cast<Index>(cols.size()), largest_column_nnz_number));

    for (Index j = 0; j < submat.cols(); ++j)
    {
        for (Index i = 0; i < submat.rows(); ++i)
        {
            const auto row = rows[i];
            const auto col = cols[j];
            // TODO: Rewrite this function so that we loop over NNZs in matrix instead, as now
            // we have to pay for the binary search of .coeff(), and on top of that we
            // loop over every (i, j) pair in the submatrix, which is very expensive for a very sparse matrix.
            const auto coeff = matrix.coeff(row, col);
            if (coeff != static_cast<Index>(0))
            {
                submat.insert(i, j) = matrix.coeff(row, col);
            }
        }
    }

    submat.makeCompressed();
    return submat;
};

template <typename Scalar, int Rows, int Cols>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> submatrix(const Eigen::Matrix<Scalar, Rows, Cols> &matrix,
                                                                const std::vector<int> rows,
                                                                const std::vector<int> cols)
{
    assert(std::is_sorted(rows.cbegin(), rows.cend()) && "Row indices must be sorted");
    assert(std::is_sorted(cols.cbegin(), cols.cend()) && "Col indices must be sorted");

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(rows.size(), cols.size());

    for (size_t j = 0; j < cols.size(); ++j)
    {
        for (size_t i = 0; i < rows.size(); ++i)
        {
            const int row = static_cast<int>(rows[i]);
            const int col = static_cast<int>(cols[j]);
            result(i, j) = matrix(row, col);
        }
    }

    return result;
};
