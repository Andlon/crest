#pragma once

#include <exception>
#include <stdexcept>
#include <algorithm>

namespace crest
{

    template <typename Scalar, typename ForwardIterator>
    Scalar composite_simpsons(ForwardIterator begin, ForwardIterator end, Scalar dx)
    {
        const auto num_samples = std::distance(begin, end);
        if (dx <= 0.0) throw std::invalid_argument("dx must be a positive quantity.");
        if (num_samples <= 1 || num_samples % 2 == 0)
        {
            throw std::invalid_argument("Number of samples must be an odd number greater than zero.");
        }

        const size_t num_intervals = num_samples - 1;
        auto & it = begin;
        Scalar result = 0.0;
        for (size_t i = 0; i < num_intervals; i += 2)
        {
            result += *(it++) + 4.0 * *(it++) + *it;
        }

        return (dx / 3.0) * result;
    };

}