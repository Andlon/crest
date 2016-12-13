#pragma once

#include <limits>
#include <vector>

namespace crest
{

    class AccumulatedDensityBin
    {
    public:
        double lower_bound() const { return _lower; }
        double upper_bound() const { return _upper; }
        double accumulated() const { return _accumulated; }

        explicit AccumulatedDensityBin(double lower, double upper, double accumulated)
                :   _lower(lower), _upper(upper), _accumulated(accumulated)
        {}

    private:
        double _lower;
        double _upper;
        double _accumulated;
    };

    /**
     * Given samples (X_i, Y_i), divides the range [X_min, X_max] into N bins
     * and accumulates Y_i within each bin such that
     * each bin holds the sum of Y_i for which X_i belongs to the bin.
     */
    class AccumulatedDensityHistogram
    {
    public:
        AccumulatedDensityHistogram() {}

        typedef std::vector<AccumulatedDensityBin>::const_iterator const_iterator;

        template <typename XForwardIterator, typename YForwardIterator>
        explicit AccumulatedDensityHistogram(size_t num_bins,
                                       XForwardIterator xbegin, XForwardIterator xend,
                                       YForwardIterator ybegin, YForwardIterator yend);

        const_iterator begin() const { return _bins.begin(); }
        const_iterator end() const { return _bins.end(); }

    private:
        std::vector<AccumulatedDensityBin> _bins;
    };

    class AccumulatedDensityHistogramBuilder
    {
    public:
        AccumulatedDensityHistogramBuilder() : _bin_count(20) {}

        AccumulatedDensityHistogramBuilder & add_sample(double x, double y)
        {
            _x.push_back(x);
            _y.push_back(y);
            return *this;
        }

        AccumulatedDensityHistogramBuilder & with_bin_count(size_t bin_count)
        {
            _bin_count = bin_count;
            return *this;
        };

        AccumulatedDensityHistogram build() const
        {
            return AccumulatedDensityHistogram(_bin_count, _x.begin(), _x.end(), _y.begin(), _y.end());
        };

    private:
        size_t _bin_count;
        std::vector<double> _x;
        std::vector<double> _y;
    };

    template <typename XForwardIterator, typename YForwardIterator>
    AccumulatedDensityHistogram::AccumulatedDensityHistogram(size_t num_bins,
                                                             XForwardIterator xbegin, XForwardIterator xend,
                                                             YForwardIterator ybegin, YForwardIterator yend)
    {
        if (num_bins == 0) {
            throw std::invalid_argument("Number of bins must be non-zero.");
        }
        if (std::distance(xbegin, xend) != std::distance(ybegin, yend)) {
            throw std::invalid_argument("X and Y containers must have the same size.");
        }

        double x_min, x_max;

        std::vector<double> y_accumulated(num_bins, 0.0);

        const auto x_minmax = std::minmax_element(xbegin, xend);
        if (x_minmax.first != xend)
        {
            x_min = *x_minmax.first;
            x_max = *x_minmax.second;
        } else
        {
            // The range is empty
            x_min = 0.0;
            x_max = 0.0;
        }

        const auto bin_width = (x_max - x_min) / num_bins;

        for (auto iter_pair = std::make_pair(xbegin, ybegin);
             iter_pair.first != xend;
             ++iter_pair.first, ++iter_pair.second)
        {
            const auto x = *iter_pair.first;
            const auto y = *iter_pair.second;
            const auto bin_index = std::min(num_bins - 1, static_cast<size_t>((x - x_min) / bin_width));
            y_accumulated[bin_index] += y;
        }

        _bins.reserve(y_accumulated.size());
        for (size_t i = 0; i < y_accumulated.size(); ++i)
        {
            const auto lower_bound = x_min + static_cast<double>(i) * bin_width;
            const auto upper_bound = lower_bound + bin_width;
            const auto accumulated = y_accumulated[i];
            _bins.emplace_back(AccumulatedDensityBin(lower_bound, upper_bound, accumulated));
        }
    };


}
