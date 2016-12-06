#pragma once

#include <crest/basis/homogenized_basis.hpp>

#include <string>
#include <sstream>
#include <iostream>

#include <H5Cpp.h>

namespace crest
{
    void export_basis(const HomogenizedBasis<double> & basis,
                      const std::string & filename);
    HomogenizedBasis<double> import_basis(const IndexedMesh<double, int> & coarse,
                                          const IndexedMesh<double, int> & fine,
                                          const std::string & filename);

    inline void export_basis(const HomogenizedBasis<double> & basis, const std::string & filename)
    {
        const auto & weights = basis.basis_weights();

        if (!weights.isCompressed())
        {
            throw std::logic_error("Internal error: Basis weights MUST be in compressed form.");
        }

        H5::H5File file(filename.c_str(), H5F_ACC_TRUNC);

        // We wish to store the matrix as the three arrays Values, InnerIndices, OuterStarts (Eigen terminology)

        // Set up data spaces for each array
        hsize_t values_dim[1];
        values_dim[0] = weights.nonZeros();
        H5::DataSpace values_dataspace(1, values_dim);

        // Recall that number of inner indices is the same as number of values
        hsize_t inner_indices_dim[1];
        inner_indices_dim[0] = weights.nonZeros();
        H5::DataSpace inner_indices_dataspace(1, inner_indices_dim);

        hsize_t outer_starts_dim[1];
        outer_starts_dim[0] = weights.outerSize() + 1;
        H5::DataSpace outer_starts_dataspace(1, outer_starts_dim);

        // Set up data spaces for matrix dimension attribute
        hsize_t matrix_dim_dim[1];
        matrix_dim_dim[0] = 2;

        H5::DataSpace matrix_dim_dataspace(1, matrix_dim_dim);

        // Note: Using NATIVE_INT and NATIVE_DOUBLE here is only guaranteed to be correctly readable on a computer
        // with the same architecture.
        // TODO: Tweak DataSet::write call
        H5::FloatType values_type(H5::PredType::NATIVE_DOUBLE);
        H5::IntType indices_type(H5::PredType::NATIVE_INT);
        indices_type.setOrder(H5T_ORDER_LE);

        H5::DataSet values_dataset = file.createDataSet("values", values_type, values_dataspace);
        H5::DataSet inner_indices_dataset = file.createDataSet("inner_indices", indices_type, inner_indices_dataspace);
        H5::DataSet outer_starts_dataset = file.createDataSet("outer_starts", indices_type, outer_starts_dataspace);
        H5::Attribute matrix_dim_attrib = file.createAttribute("dim", indices_type, matrix_dim_dataspace);

        int matrix_dim[2];
        matrix_dim[0] = weights.rows();
        matrix_dim[1] = weights.cols();

        values_dataset.write(weights.valuePtr(), H5::PredType::NATIVE_DOUBLE);
        inner_indices_dataset.write(weights.innerIndexPtr(), H5::PredType::NATIVE_INT);
        outer_starts_dataset.write(weights.outerIndexPtr(), H5::PredType::NATIVE_INT);
        matrix_dim_attrib.write(H5::PredType::NATIVE_INT, matrix_dim);
    }

    inline HomogenizedBasis<double> import_basis(const BiscaleMesh<double, int> & mesh,
                                                 const std::string &filename)
    {
        H5::H5File file(filename.c_str(), H5F_ACC_RDONLY);

        const auto matrix_dim_attrib = file.openAttribute("dim");
        const auto values_dataset = file.openDataSet("values");
        const auto inner_indices_dataset = file.openDataSet("inner_indices");
        const auto outer_starts_dataset = file.openDataSet("outer_starts");

        const auto verify_ndims = [&] (const auto & object, const auto & name)
        {
            if (object.getSpace().getSimpleExtentNdims() != 1)
            {
                std::stringstream ss;
                ss << "Corrupted dimensions for `" << name << "`.";
                throw std::runtime_error(ss.str());
            }
        };

        verify_ndims(matrix_dim_attrib, "dim");
        verify_ndims(values_dataset, "values");
        verify_ndims(inner_indices_dataset, "inner_indices");
        verify_ndims(outer_starts_dataset, "outer_starts");

        hsize_t matrix_dim_dim[1];
        matrix_dim_attrib.getSpace().getSimpleExtentDims(matrix_dim_dim);

        if (matrix_dim_dim[0] != 2)
        {
            throw std::runtime_error("Matrix dimensions corruption in data file.");
        }

        int matrix_dim[2];
        matrix_dim_attrib.read(H5::PredType::NATIVE_INT, matrix_dim);

        const auto rows = matrix_dim[0];
        const auto cols = matrix_dim[1];

        hsize_t values_dim[1];
        values_dataset.getSpace().getSimpleExtentDims(values_dim);

        hsize_t inner_indices_dim[1];
        inner_indices_dataset.getSpace().getSimpleExtentDims(inner_indices_dim);

        hsize_t outer_starts_dim[1];
        outer_starts_dataset.getSpace().getSimpleExtentDims(outer_starts_dim);

        const auto nnz = values_dim[0];
        const auto outer_starts_size = outer_starts_dim[0];

        if (nnz != inner_indices_dim[0])
        {
            throw std::runtime_error("Number of inner indices does not match number of non-zeros.");
        }

        if (outer_starts_size != static_cast<unsigned int>(cols) + 1)
        {
            throw std::runtime_error("Outer size does not match number of columns.");
        }


        // TODO: What happens below is _extremely_ hacky. Would really prefer a better way.
        Eigen::SparseMatrix<double> weights(rows, cols);
        // According to docs, matrix needs to be compressed before .reserve() actually takes in effect.
        weights.reserve(nnz);
        weights.uncompress();

        values_dataset.read(weights.valuePtr(), H5::PredType::NATIVE_DOUBLE);
        inner_indices_dataset.read(weights.innerIndexPtr(), H5::PredType::NATIVE_INT);
        outer_starts_dataset.read(weights.outerIndexPtr(), H5::PredType::NATIVE_INT);

        // At this point, Eigen still thinks it has an uncompressed matrix, so we also need to adjust its
        // inner indices, so that Eigen does not destroy our matrix when it tries to compress it.
        for (int i = 0; i < cols; ++i)
        {
            const auto nnz_in_col = *(weights.outerIndexPtr() + i + 1) - *(weights.outerIndexPtr() + i);
            *(weights.innerNonZeroPtr() + i) = nnz_in_col;
        }

        weights.makeCompressed();

        return HomogenizedBasis<double>(mesh, weights);
    }
}
