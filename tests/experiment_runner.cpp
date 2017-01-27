#include <crest/wave/solver.hpp>

#include <crest/wave/constraints.hpp>

#include <crest/basis/lagrange_basis2d.hpp>
#include <crest/geometry/refinement.hpp>
#include <crest/quadrature/simpsons.hpp>
#include <crest/basis/amg_corrector_solver.hpp>

#include <experiments/experiments.hpp>

#include <json.hpp>

#include <iostream>
#include <iomanip>
#include <string>
#include <memory>
#include <sstream>

#include <boost/optional.hpp>

using std::cout;
using std::endl;
using std::cerr;
using std::setw;

auto sampled_data_summary_as_json(const crest::AccumulatedDensityHistogram & histogram)
{
    nlohmann::json j;
    auto & hist = j["distribution"];
    for (const auto & bin : histogram)
    {
        hist.push_back({
                               { "lower", bin.lower_bound() },
                               { "upper", bin.upper_bound() },
                               { "accumulated", bin.accumulated() }
                       });
    }

    return j;
}

auto stats_as_json(const std::unordered_map<std::string, crest::AccumulatedDensityHistogram> & stats)
{
    nlohmann::json j;
    for (const auto pair : stats)
    {
        j[pair.first] = sampled_data_summary_as_json(pair.second);
    }

    return j;
}

auto experiment_result_as_json(const ExperimentResult & result)
{
    using nlohmann::json;

    const auto offline_timing = result.offline_result.timing;
    const auto mesh_details = result.offline_result.mesh_details;

    json output = {
            { "experiment", result.name },
            { "offline", {
                                    { "parameters", {
                                                            { "mesh_resolution", result.offline_parameters.mesh_resolution },
                                                            { "oversampling", result.offline_parameters.oversampling },
                                                            { "basis_import_file", result.offline_parameters.basis_import_file },
                                                            { "basis_export_file", result.offline_parameters.basis_export_file },
                                                            { "dense_fallback_threshold", result.offline_parameters.dense_fallback_threshold },
                                                            { "iterative_tolerance", result.offline_parameters.iterative_tolerance }
                                                    }},
                                    { "result", {
                                                        { "mesh_details", {
                                                                                  { "num_coarse_vertices", mesh_details.num_coarse_vertices },
                                                                                  { "num_coarse_elements", mesh_details.num_coarse_elements },
                                                                                  { "num_fine_vertices", mesh_details.num_fine_vertices },
                                                                                  { "num_fine_elements", mesh_details.num_fine_elements }
                                                                          }},
                                                        { "timing", {
                                                                            { "mesh_construction", offline_timing.mesh_construction },
                                                                            { "basis_construction", offline_timing.basis_construction },
                                                                            { "total", offline_timing.total }
                                                                    }},
                                                        { "stats", stats_as_json(result.offline_result.stats) }
                                                }}
                            }}
    };

    if (result.online_parameters && result.online_result)
    {
        const auto errors = result.online_result->error_summary;
        const auto timing = result.online_result->timing;
        output["online"] = {
                { "parameters", {
                                        { "end_time", result.online_parameters->end_time },
                                        { "sample_count", result.online_parameters->sample_count },
                                        { "integrator", result.online_parameters->integrator_name },
                                        { "load_quadrature_strength", result.online_parameters->load_quadrature_strength },
                                        { "use_coarse_rhs", result.online_parameters->use_coarse_rhs },
                                        { "use_coarse_mass_matrix", result.online_parameters->use_coarse_mass_matrix },
                                        { "iterative_tolerance", result.online_parameters->iterative_tolerance }
                                }},
                { "result", {
                                        { "error_summary", {
                                                              { "h1", errors.h1 },
                                                              { "h1_semi", errors.h1_semi },
                                                              { "l2", errors.l2 }
                                                      }},
                                        { "timing", {
                                                                  { "load_time", timing.load_time },
                                                                  { "initializer_time", timing.initializer_time },
                                                                  { "integrator_setup_time", timing.integrator_setup_time },
                                                                  { "integrator_solve_time", timing.integrator_solve_time },
                                                                  { "transform_time", timing.transform_time },
                                                                  { "total_time", timing.total_time }
                                                          }}
                                }}

        };
    } else if (result.online_parameters || result.online_result)
    {
        throw new std::logic_error("Internal error: Online results and parameters should either both be absent,"
                                           " or both be present.");
    }

    return output;
}

std::unique_ptr<Experiment> make_experiment(const std::string & name)
{
    if (name == "homogeneous_dirichlet_unit_square") {
        return std::make_unique<HomogeneousDirichletUnitSquare>();
    } else if (name == "inhomogeneous_dirichlet_unit_square") {
        return std::make_unique<InhomogeneousDirichletUnitSquare>();
    } else if (name == "homogenized_l_shaped") {
        return std::make_unique<HomogenizedLShaped<>>();
    } else if (name == "homogenized_l_shaped_sparselu") {
        return std::make_unique<HomogenizedLShaped<crest::SparseLuCorrectorSolver<double>>>();
    } else if (name == "homogenized_l_shaped_schur") {
        return std::make_unique<HomogenizedLShaped<crest::SchurCorrectorSolver<double>>>();
    } else if (name == "homogenized_l_shaped_amg") {
        return std::make_unique<HomogenizedLShaped<crest::AmgCorrectorSolver<double>>>();
    } else if (name == "standard_l_shaped") {
        return std::make_unique<StandardLShaped>();
    } else {
        throw std::invalid_argument("Unknown experiment requested.");
    }
}

std::unique_ptr<crest::wave::Integrator<double>> make_integrator(const std::string & name)
{
    using namespace crest::wave;
    if (name == "direct_crank_nicolson")
    {
        return std::make_unique<DirectCrankNicolson<double>>();
    } else if (name == "iterative_crank_nicolson")
    {
        return std::make_unique<IterativeCrankNicolson<double>>();
    } else if (name == "iterative_leapfrog")
    {
        return std::make_unique<IterativeLeapfrog<double>>();
    } else if (name == "lumped_leapfrog")
    {
        return std::make_unique<LumpedLeapfrog<double>>();
    } else {
        return std::unique_ptr<crest::wave::Integrator<double>>();
    }
}


template <typename OutputType = nlohmann::json>
OutputType extract_mandatory_field(const nlohmann::json & json, const std::string & field_name)
{
    const auto it = json.find(field_name);
    if (it == json.end())
    {
        std::stringstream ss;
        ss << "Mandatory field `" << field_name << "` not found in JSON input.";
        throw std::runtime_error(ss.str());
    } else {
        return *it;
    }
}

template <typename OutputType = nlohmann::json>
boost::optional<OutputType> extract_optional_field(const nlohmann::json & json,
                                                   const std::string & field_name)
{
    const auto it = json.find(field_name);
    if (it == json.end())
    {
        return boost::optional<OutputType>();
    } else {
        return *it;
    }
}

OfflineParameters parse_offline_parameters(const nlohmann::json & offline_json) {
    OfflineParameters offline;
    offline.mesh_resolution = extract_mandatory_field(offline_json, "mesh_resolution");
    if (const auto oversampling = extract_optional_field(offline_json, "oversampling")) {
        offline.oversampling = *oversampling;
    }
    if (const auto basis_import_file = extract_optional_field(offline_json, "basis_import_file")) {
        offline.basis_import_file = *basis_import_file;
    }
    if (const auto basis_export_file = extract_optional_field(offline_json, "basis_export_file")) {
        offline.basis_export_file = *basis_export_file;
    }
    if (const auto threshold = extract_optional_field(offline_json, "dense_fallback_threshold")) {
        offline.dense_fallback_threshold = *threshold;
    }
    if (const auto iterative_tolerance = extract_optional_field(offline_json, "iterative_tolerance")) {
        offline.iterative_tolerance = *iterative_tolerance;
    }
    return offline;
}

OnlineParameters parse_online_parameters(const nlohmann::json & online_json) {
    OnlineParameters parameters;
    parameters.integrator_name = extract_mandatory_field(online_json, "integrator");
    parameters.end_time = extract_mandatory_field(online_json, "end_time");
    parameters.sample_count = extract_mandatory_field(online_json, "sample_count");

    if (const auto load_strength = extract_optional_field(online_json, "load_quadrature_strength")) {
        parameters.load_quadrature_strength = *load_strength;
    }

    if (const auto use_coarse_rhs = extract_optional_field(online_json, "use_coarse_rhs")) {
        parameters.use_coarse_rhs = *use_coarse_rhs;
    }

    if (const auto use_coarse_mass_matrix = extract_optional_field(online_json, "use_coarse_mass_matrix")) {
        parameters.use_coarse_mass_matrix = *use_coarse_mass_matrix;
    }

    if (const auto iterative_tol = extract_optional_field(online_json, "iterative_tolerance")) {
        parameters.iterative_tolerance = *iterative_tol;
    }

    return parameters;
}


int main(int, char **)
{
    H5::Exception::dontPrint();

    // Disable automatic multi-threading in Eigen, so that we have predictable behavior. In order to get
    // decent measurements, we'd rather parallelize explicitly where we actually want to parallelize.
    // Remember that the main purpose of this binary is not to solve problems as fast as possible,
    // but to facilitate numerical experimentation, including making inferences about runtime behavior.
    Eigen::setNbThreads(1);
    try
    {
        nlohmann::json j;
        std::cin >> j;

        const auto experiment_name = extract_mandatory_field<std::string>(j, "experiment");

        const auto offline_json = extract_mandatory_field(j, "offline");
        const auto optional_online_json = extract_optional_field(j, "online");

        const auto offline_param = parse_offline_parameters(offline_json);
        const auto online_param = optional_online_json
                                  ? parse_online_parameters(*optional_online_json)
                                  : boost::optional<OnlineParameters>();

        auto experiment = make_experiment(experiment_name);
        const auto offline_result = experiment->run_offline(offline_param);
        boost::optional<OnlineResult> online_result;
        if (online_param)
        {
            auto integrator = make_integrator(online_param->integrator_name);

            if (integrator) {
                integrator->set_tolerance(online_param->iterative_tolerance);
                online_result = experiment->run_online(*online_param, *integrator);
            } else {
                throw std::runtime_error("Unknown integrator requested.");
            }
        }

        const auto result = ExperimentResult()
                .with_name(experiment_name)
                .with_offline_parameters(offline_param)
                .with_online_parameters(online_param)
                .with_offline_result(offline_result)
                .with_online_result(online_result);
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
