#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <vector>
#include <algorithm>

#include <crest/util/stat.hpp>

using ::testing::Eq;
using ::testing::ElementsAreArray;
using ::testing::DoubleNear;

TEST(accumulated_binned_data_test, simple_example)
{
    std::vector<double> x {  1.0, -2.0, 3.0, 15.0, 0.0, 11.0 };
    std::vector<double> y { -1.0, -3.0, -5.0, 0.0, 0.4,  7.0 };

    crest::AccumulatedDensityHistogram data(5, x.begin(), x.end(), y.begin(), y.end());

    std::vector<crest::AccumulatedDensityBin> bins;
    std::copy(data.begin(), data.end(), std::back_inserter(bins));

    ASSERT_THAT(bins.size(), Eq(5u));
    EXPECT_THAT(bins[0].lower_bound(), Eq(-2.0));
    EXPECT_THAT(bins[0].upper_bound(), Eq(1.4));
    EXPECT_THAT(bins[0].accumulated(), Eq(-3.6));
    EXPECT_THAT(bins[1].lower_bound(), Eq(1.4));
    EXPECT_THAT(bins[1].upper_bound(), Eq(4.8));
    EXPECT_THAT(bins[1].accumulated(), Eq(-5.0));
    EXPECT_THAT(bins[2].lower_bound(), Eq(4.8));
    EXPECT_THAT(bins[2].upper_bound(), Eq(8.2));
    EXPECT_THAT(bins[2].accumulated(), Eq(0.0));
    EXPECT_THAT(bins[3].lower_bound(), Eq(8.2));
    EXPECT_THAT(bins[3].upper_bound(), Eq(11.6));
    EXPECT_THAT(bins[3].accumulated(), Eq(7.0));
    EXPECT_THAT(bins[4].lower_bound(), Eq(11.6));
    EXPECT_THAT(bins[4].upper_bound(), Eq(15.0));
    EXPECT_THAT(bins[4].accumulated(), Eq(0.0));
}



