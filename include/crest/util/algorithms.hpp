#pragma once

#include <algorithm>

namespace crest {

    namespace algo {

        template <typename Container, typename Element>
        bool contains(const Container & c, const Element & element)
        {
            return std::find(c.begin(), c.end(), element) != c.end();
        };

        template <typename Integer, typename ForwardIt>
        void fill_strided_integers(ForwardIt begin, ForwardIt end, Integer start = 0, Integer stride = 1)
        {
            Integer current_value = start;
            std::generate(begin, end, [&] {
                const auto val = current_value;
                current_value += stride;
                return val;
            });
        };

        template <typename OutputIt, typename Integer>
        void fill_strided_integers_n(OutputIt first, Integer count, Integer start = 0, Integer stride = 1)
        {
            Integer current_value = start;
            std::generate_n(first, count, [&] {
                const auto val = current_value;
                current_value += stride;
                return val;
            });
        };

    }

}
