#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <ostream>


namespace Eigen
{
    template <typename Scalar, int Row, int Col, int Options, int MaxRows, int MaxCols>
    void PrintTo(const ::Eigen::Matrix<Scalar, Row, Col, Options, MaxRows, MaxCols> &matrix,::std::ostream * os)
    {
        *os << std::endl << matrix;
    };
}

