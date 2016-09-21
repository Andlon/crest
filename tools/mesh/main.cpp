#include <crest/geometry/indexed_mesh.hpp>
#include <crest/geometry/refinement.hpp>

#include <json.hpp>

#include <iostream>
#include <vector>

using nlohmann::json;

std::vector<crest::Vertex<double>> extract_vertices(const json & vertices_field)
{
    std::vector<crest::Vertex<double>> vertices;
    for (const auto & json_vertex : vertices_field)
    {
        const double x = json_vertex.at(0);
        const double y = json_vertex.at(1);
        vertices.emplace_back(crest::Vertex<double>(x, y));
    }
    return vertices;
}

std::vector<crest::Element<int>> extract_elements(const json & elements_field)
{
    std::vector<crest::Element<int>> elements;
    for (const auto & json_element :elements_field)
    {
        const int a = json_element.at(0);
        const int b = json_element.at(1);
        const int c = json_element.at(2);
        elements.emplace_back(crest::Element<int>({a, b, c}));
    }
    return elements;
}

int main(int, const char **)
{
    nlohmann::json json_input;
    std::cin >> json_input;

    const auto vertices = extract_vertices(json_input["vertices"]);
    const auto elements = extract_elements(json_input["elements"]);
    const double tolerance = json_input["tolerance"];

    auto initial_mesh = crest::IndexedMesh<double, int>(vertices, elements);
    const auto mesh = crest::refine_to_tolerance<double, int>(std::move(initial_mesh), tolerance);

    json json_output;
    {
        json json_vertices;
        for (const auto & vertex : mesh.vertices())
        {
            json json_vertex;
            json_vertex.push_back(vertex.x);
            json_vertex.push_back(vertex.y);
            json_vertices.push_back(json_vertex);
        }
        json_output["vertices"] = json_vertices;
    }
    {
        json json_elements;
        for (const auto & element : mesh.elements())
        {
            json json_element;
            json_element.push_back(element.vertex_indices[0]);
            json_element.push_back(element.vertex_indices[1]);
            json_element.push_back(element.vertex_indices[2]);
            json_elements.push_back(json_element);
        }
        json_output["elements"] = json_elements;
    }

    std::cout << json_output.dump(4) << std::endl;


    return 0;
}