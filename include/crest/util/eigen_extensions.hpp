#pragma once

#include <Eigen/Sparse>
#include <cassert>
#include <algorithm>
#include <ostream>


template <typename Scalar>
using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

namespace detail
{
    template <typename Scalar, typename Index>
    Eigen::VectorXi submatrix_sparsity_pattern(const Eigen::SparseMatrix<Scalar, 0, Index> & matrix,
                                               const std::vector<Index> & reverse_rowmap,
                                               const std::vector<Index> & cols)
    {
        typedef typename Eigen::SparseMatrix<Scalar, 0, Index>::InnerIterator InnerIterator;
        const auto submat_cols = static_cast<Index>(cols.size());

        // Each entry in the vector holds the number of non-zero rows in the column
        Eigen::VectorXi sparsity_pattern(submat_cols);

        for (Index col = 0; col < submat_cols; ++col)
        {
            sparsity_pattern(col) = 0;
            const auto original_col = cols[col];

            for (InnerIterator it(matrix, original_col); it; ++it)
            {
                const auto submatrix_row = reverse_rowmap[it.row()];
                if (submatrix_row >= 0)
                {
                    sparsity_pattern(col) += 1;
                }
            }
        }

        return sparsity_pattern;
    }
}

template <typename Scalar, typename Index>
Eigen::SparseMatrix<Scalar, 0, Index> sparse_submatrix(const Eigen::SparseMatrix<Scalar, 0, Index> & matrix,
                                                       const std::vector<Index> & rows,
                                                       const std::vector<Index> & cols)
{
    typedef typename Eigen::SparseMatrix<Scalar, 0, Index>::InnerIterator InnerIterator;
    assert(std::is_sorted(rows.cbegin(), rows.cend()) && "Row indices must be sorted.");
    assert(std::is_sorted(cols.cbegin(), cols.cend()) && "Column indices must be sorted.");

    // rows and cols map indices from the submatrix into the original matrix. We want to build a reverse map,
    // which is generally not surjective, hence we denote elements that should not be present by -1.
    std::vector<Index> reverse_rowmap(matrix.rows(), -1);

    for (size_t submatrix_row = 0; submatrix_row < rows.size(); ++submatrix_row)
    {
        const auto original_row = rows[submatrix_row];
        reverse_rowmap[original_row] = submatrix_row;
    }

    const auto submat_rows = static_cast<Index>(rows.size());
    const auto submat_cols = static_cast<Index>(cols.size());

    Eigen::SparseMatrix<Scalar, 0, Index> submat(submat_rows, submat_cols);
    if (submat_rows == 0 || submat_cols == 0)
    {
        return submat;
    }

    submat.reserve(detail::submatrix_sparsity_pattern(matrix, reverse_rowmap, cols));

    for (Index col = 0; col < submat.cols(); ++col)
    {
        const auto original_col = cols[col];

        // We implicitly make the assumption here that the original matrix's columns are relatively sparse,
        // which is almost always the case.
        for (InnerIterator it(matrix, original_col); it; ++it)
        {
            const auto submatrix_row = reverse_rowmap[it.row()];
            if (submatrix_row >= 0)
            {
                submat.insert(submatrix_row, col) = it.value();
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
