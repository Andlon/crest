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
Eigen::VectorXi submatrix_sparsity_pattern(const Eigen::SparseMatrix<Scalar, 0, Index> & matrix,
                                           const std::vector<Index> & rows,
                                           const std::vector<Index> & cols)
{
    typedef typename Eigen::SparseMatrix<Scalar, 0, Index>::InnerIterator InnerIterator;
    const auto submat_rows = static_cast<Index>(rows.size());
    const auto submat_cols = static_cast<Index>(cols.size());

    // Each entry in the vector holds the number of non-zero rows in the column
    Eigen::VectorXi sparsity_pattern(submat_cols);

    for (Index col = 0; col < submat_cols; ++col)
    {
        sparsity_pattern(col) = 0;
        Index current_submat_row = 0;
        const auto original_col = cols[col];

        for (InnerIterator it(matrix, original_col); it && current_submat_row < submat_rows; ++it)
        {
            while (current_submat_row < submat_rows && it.row() > rows[current_submat_row])
            {
                ++current_submat_row;
            }

            if (current_submat_row < submat_rows && it.row() == rows[current_submat_row])
            {
                sparsity_pattern(col) += 1;
                ++current_submat_row;
            }
        }
    }

    return sparsity_pattern;
};

template <typename Scalar, typename Index>
Eigen::SparseMatrix<Scalar, 0, Index> sparse_submatrix(const Eigen::SparseMatrix<Scalar, 0, Index> & matrix,
                                                       const std::vector<Index> & rows,
                                                       const std::vector<Index> & cols)
{
    typedef typename Eigen::SparseMatrix<Scalar, 0, Index>::InnerIterator InnerIterator;
    assert(std::is_sorted(rows.cbegin(), rows.cend()) && "Row indices must be sorted.");
    assert(std::is_sorted(cols.cbegin(), cols.cend()) && "Column indices must be sorted.");

    const auto submat_rows = static_cast<Index>(rows.size());
    const auto submat_cols = static_cast<Index>(cols.size());

    Eigen::SparseMatrix<Scalar, 0, Index> submat(submat_rows, submat_cols);
    if (submat_rows == 0 || submat_cols == 0)
    {
        return submat;
    }

    submat.reserve(submatrix_sparsity_pattern(matrix, rows, cols));

    for (Index col = 0; col < submat.cols(); ++col)
    {
        Index current_submat_row = 0;
        const auto original_col = cols[col];

        // We implicitly make the assumption here that the original matrix's columns are relatively sparse,
        // which is almost always the case.
        for (InnerIterator it(matrix, original_col); it && current_submat_row < submat_rows; ++it)
        {
            while (current_submat_row < submat_rows && it.row() > rows[current_submat_row])
            {
                ++current_submat_row;
            }

            if (current_submat_row < submat_rows && it.row() == rows[current_submat_row])
            {
                const auto val = it.value();
                submat.insert(current_submat_row, col) = val;
                ++current_submat_row;
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
