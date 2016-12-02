#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <rapidcheck/gtest.h>

#include <crest/basis/homogenized_basis.hpp>
#include <crest/io/homogenized_basis_io.hpp>

#include <util/test_generators.hpp>

#include <cstdio>

class TemporaryFilePath
{
public:
    TemporaryFilePath()
    {
        _path = tmpnam(nullptr);
    }

    ~TemporaryFilePath()
    {
        remove(_path.c_str());
    }

    std::string path() const { return _path; }

private:
    std::string _path;
};


RC_GTEST_PROP(basis_io, export_import_roundtrip, ())
{
    const auto coarse = *crest::gen::arbitrary_unit_square_mesh();
    const auto fine = *crest::gen::arbitrary_refinement(coarse, 0).as("fine mesh");

    const auto oversampling = static_cast<unsigned int>(coarse.num_vertices());
    const auto basis = crest::HomogenizedBasis<double>(coarse, fine, oversampling);

    const auto file_path = TemporaryFilePath();

    crest::export_basis(basis, file_path.path());

    const auto imported = crest::import_basis(coarse, fine, file_path.path());

    const Eigen::MatrixXd original_weights = basis.basis_weights();
    const Eigen::MatrixXd imported_weights = imported.basis_weights();

    RC_LOG() << "temporary file: " << file_path.path() << std::endl;
    RC_LOG() << "original basis weights: " << std::endl << original_weights << std::endl;
    RC_LOG() << "imported basis weights: " << std::endl << imported_weights << std::endl;

    RC_ASSERT(original_weights == imported_weights);
}
