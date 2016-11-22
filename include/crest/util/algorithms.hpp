#pragma once

#include <algorithm>
#include <vector>

namespace crest {

    namespace algo {

        template <typename Container, typename Element>
        bool contains(const Container & c, const Element & element)
        {
            return std::find(c.begin(), c.end(), element) != c.end();
        };

        // Returns the index of the first occurrence of element in the contiguous random-access container c.
        // If the element does not exist, returns ~0u (the maximum unsigned value).
        template <typename Container, typename Element>
        size_t index_of(const Container & c, const Element & element)
        {
            const auto it = std::find(c.begin(), c.end(), element);

            return it != c.end()
                   ? it - c.begin()
                   : ~0u;
        };

        template <typename Container>
        Container sorted_unique(Container c)
        {
            std::sort(c.begin(), c.end());
            c.erase(std::unique(c.begin(), c.end()), c.end());
            return c;
        }

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

        /**
         * Returns a vector filled with consecutive integers in the range [start, end).
         * @param start
         * @param end
         * @return
         */
        template <typename Integer>
        std::vector<Integer> integer_range(Integer start, Integer end)
        {
            // Of course, this vector is a waste of memory, and one would be better off with some
            // range iterator type, but for now this is simpler.

            if (end < start)
            {
                throw std::logic_error("end must be greater or equal to start.");
            }

            std::vector<int> v;
            v.reserve(end - start);
            for (auto i = start; i < end; ++i)
            {
                v.push_back(i);
            }
            return v;
        };

    }

}
