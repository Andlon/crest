#pragma once

#include <rapidcheck/gtest.h>

#include <crest/geometry/indexed_mesh.hpp>

#include <sstream>

namespace crest
{


    namespace gen {
        inline auto minimal_unit_square_mesh()
        {
            typedef crest::IndexedMesh<double, int>::Vertex Vertex;
            typedef crest::IndexedMesh<double, int>::Element Element;

            const std::vector<Vertex> coarse_vertices {
                    Vertex(0.0, 0.0),
                    Vertex(1.0, 0.0),
                    Vertex(1.0, 1.0),
                    Vertex(0.0, 1.0)
            };

            const std::vector<Element> coarse_elements {
                    Element({3, 0, 1}),
                    Element({1, 2, 3})
            };
            auto mesh = crest::IndexedMesh<double, int>(coarse_vertices, coarse_elements);
            return rc::gen::just(mesh).as("initial unit square mesh");
        }

        /**
         * A generator for a small almost arbitrary course mesh for the unit square,
         * constructed through successive rounds of bisection on random elements,
         * starting from a two-element triangulation.
         */
        inline auto arbitrary_unit_square_mesh(const int max_bisection_rounds = 6)
        {
            auto mesh = *minimal_unit_square_mesh();

            const int num_bisections = *rc::gen::inRange(0, max_bisection_rounds + 1).as("number of bisection rounds");

            for (int i = 0; i < num_bisections; ++i)
            {
                std::ostringstream ss;
                ss << "marked elements in round " << i;
                const auto marked = *rc::gen::container<std::vector<int>>(rc::gen::inRange(0, mesh.num_elements()))
                    .as(ss.str());
                mesh.bisect_marked(marked);
            }

            mesh.reset_ancestry();

            return rc::gen::just(mesh).as("unit square mesh");
        };
    }
}