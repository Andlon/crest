#pragma once

#include <algorithm>

namespace crest {

    namespace algo {

        template <typename Container, typename Element>
        bool contains(const Container & c, const Element & element)
        {
            return std::find(c.begin(), c.end(), element) != c.end();
        };

    }

}
