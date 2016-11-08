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

        inline auto arbitrary_refinement(crest::IndexedMesh<double, int> mesh,
                                         const int min_bisection_rounds = 1,
                                         const int max_bisection_rounds = 5)
        {
            const int num_bisections = *rc::gen::inRange(min_bisection_rounds,
                                                         max_bisection_rounds + 1).as("number of bisection rounds");

            for (int i = 0; i < num_bisections; ++i)
            {
                std::ostringstream ss;
                ss << "marked elements in round " << i;
                // Ideally we'd just use the generator std::set here, but it seems to generate some very strange
                // behavior that makes no sense to me
                const auto marked = *rc::gen::map(rc::gen::container<std::vector<int>>(rc::gen::inRange(0, mesh.num_elements())),
                                                  [] (auto vec)
                                                  {
                                                      return algo::sorted_unique(vec);
                                                  })
                        .as(ss.str());
                mesh.bisect_marked(marked);
            }

            return rc::gen::just(mesh).as("refined mesh");
        }

        /**
         * A generator for a small almost arbitrary course mesh for the unit square,
         * constructed through successive rounds of bisection on random elements,
         * starting from a two-element triangulation.
         */
        inline auto arbitrary_unit_square_mesh(const int max_bisection_rounds = 5)
        {
            auto mesh_gen = arbitrary_refinement(*minimal_unit_square_mesh(), 0, max_bisection_rounds);

            return rc::gen::map(mesh_gen, [] (auto mesh) {
                mesh.reset_ancestry();
                return mesh;
            }).as("unit square mesh");
        };
    }
}
