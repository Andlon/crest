#include <crest/geometry/refinement.hpp>

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

typedef crest::IndexedMesh<double, int>::Vertex Vertex;
typedef crest::IndexedMesh<double, int>::Element Element;

int main(int argc, char ** argv)
{
    if (argc != 2)
    {
        std::cerr << "Invalid number of arguments. Executable takes a single argument, corresponding to the "
                "scalar tolerance of the bisection procedure." << std::endl;
    }

    const auto tolerance = std::atof(argv[1]);

    if (std::isfinite(tolerance) &&  tolerance <= 0 )
    {
        std::cerr << "Tolerance must be a positive finite number." << std::endl;
    }

    const std::vector<Vertex> vertices {
            Vertex(0.0, 0.0),
            Vertex(1.0, 0.0),
            Vertex(1.0, 1.0),
            Vertex(0.0, 1.0)
    };

    const std::vector<Element> elements {
            Element({3, 0, 1}),
            Element({1, 2, 3})
    };

    const auto initial_mesh = crest::IndexedMesh<double, int>(vertices, elements);

    std::cout << "Bisecting..." << std::endl;

    const auto start_time = std::chrono::steady_clock::now();
    const auto mesh = crest::bisect_to_tolerance(initial_mesh, tolerance);
    const auto end_time = std::chrono::steady_clock::now();

    const auto duration = end_time - start_time;
    const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    std::cout << "Bisection complete." << std::endl
              << "Number of vertices: " << mesh.num_vertices() << std::endl
              << "Number of elements: " << mesh.num_elements() << std::endl
              << "Runtime: " << duration_ms.count() << " ms" << std::endl;
}