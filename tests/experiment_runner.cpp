#include <crest/wave/solver.hpp>

#include <crest/wave/constraints.hpp>

#include <crest/basis/lagrange_basis2d.hpp>
#include <crest/geometry/refinement.hpp>
#include <crest/quadrature/simpsons.hpp>

#include <experiments/experiments.hpp>

#include <json.hpp>

#include <iostream>
#include <iomanip>
#include <string>

using std::cout;
using std::endl;
using std::cerr;
using std::setw;

auto experiment_result_as_json(const ExperimentResult & result)
{
    using nlohmann::json;
    json output = {
            { "experiment", result.name },
            { "mesh", {
                                    {"num_vertices", result.mesh_details.num_vertices },
                                    {"num_elements", result.mesh_details.num_elements }
                            }},
            { "error_summary", {
                                    { "h1", result.error_summary.h1 },
                                    { "h1_semi", result.error_summary.h1_semi },
                                    { "l2", result.error_summary.l2 }
                            }},
            { "parameters", {
                                    { "mesh_resolution", result.parameters.mesh_resolution },
                                    { "sample_count", result.parameters.sample_count },
                                    { "end_time", result.parameters.end_time }
                            }}
    };

    return output;
}

ExperimentResult run_experiment(const std::string & name,
                                const ExperimentParameters & params,
                                crest::wave::Integrator<double> & integrator)
{
    if (name == "homogeneous_dirichlet_unit_square") {
        return HomogeneousLoadUnitSquare().run(params, integrator);
    } else if (name == "inhomogeneous_dirichlet_unit_square") {
        return InhomogeneousLoadUnitSquare().run(params, integrator);
    } else if (name == "homogenized_l_shaped")
    {
        return HomogenizedLShaped().run(params, integrator);
    } else if (name == "standard_l_shaped")
    {
        return StandardLShaped().run(params, integrator);
    } else {
        throw std::invalid_argument("Unknown experiment requested.");
    }
}

int main(int, char **)
{
    try
    {
        nlohmann::json j;
        std::cin >> j;

        const std::string experiment_name = j["experiment"];
        const auto input_params = j["parameters"];

        ExperimentParameters params;
        params.end_time = input_params["end_time"];
        params.mesh_resolution = input_params["mesh_resolution"];
        params.sample_count = input_params["sample_count"];

        crest::wave::CrankNicolson<double> integrator;
        const auto result = run_experiment(experiment_name, params, integrator)
                .with_name(experiment_name);
        const auto json = experiment_result_as_json(result);
        cout << json.dump(4) << endl;
        return 0;
    } catch (const std::exception & e)
    {
        nlohmann::json j = {
                { "error", e.what() }
        };
        cout << j.dump(4) << endl;

        return 1;
    }
}
