#pragma once

#include <crest/wave/solver.hpp>
#include <crest/geometry/indexed_mesh.hpp>
#include <crest/quadrature/simpsons.hpp>
#include <crest/util/stat.hpp>

#include <string>
#include <cmath>

#include <boost/optional.hpp>

struct OfflineParameters
{
    unsigned int oversampling;
    double mesh_resolution;

    std::string basis_export_file;
    std::string basis_import_file;

    unsigned int dense_fallback_threshold;
    double iterative_tolerance;

    OfflineParameters()
            : oversampling(0),
              mesh_resolution(0.0),
              dense_fallback_threshold(300),
              iterative_tolerance(10.0 * std::numeric_limits<double>::epsilon())
    { }

    OfflineParameters & with_oversampling(unsigned int oversampling)
    {
        this->oversampling = oversampling;
        return *this;
    }

    OfflineParameters & with_mesh_resolution(double mesh_resolution)
    {
        this->mesh_resolution = mesh_resolution;
        return *this;
    }

    OfflineParameters & with_basis_export_file(std::string path)
    {
        this->basis_export_file = path;
        return *this;
    }

    OfflineParameters & with_basis_import_file(std::string path)
    {
        this->basis_import_file = path;
        return *this;
    }

    OfflineParameters & with_dense_fallback_threshold(unsigned int threshold)
    {
        this->dense_fallback_threshold = threshold;
        return *this;
    }

    OfflineParameters & with_iterative_tolerance(double iterative_tolerance)
    {
        this->iterative_tolerance = iterative_tolerance;
        return *this;
    }
};

struct OnlineParameters
{
    double end_time;
    uint64_t sample_count;
    int load_quadrature_strength;
    bool use_coarse_rhs;
    bool use_coarse_mass_matrix;
    double iterative_tolerance;
    bool measure_error;
    double max_error_before_abort;

    std::string integrator_name;

    OnlineParameters()
            : end_time(0.0),
              sample_count(0),
              load_quadrature_strength(4),
              use_coarse_rhs(false),
              use_coarse_mass_matrix(false),
              iterative_tolerance(10.0 * std::numeric_limits<double>::epsilon()),
              measure_error(true),
              max_error_before_abort(std::numeric_limits<double>::max())
    {}

    OnlineParameters & with_end_time(double end_time)
    {
        this->end_time = end_time;
        return *this;
    }

    OnlineParameters & with_sample_count(uint64_t sample_count)
    {
        this->sample_count = sample_count;
        return *this;
    }

    OnlineParameters & with_integrator_name(std::string name)
    {
        this->integrator_name = name;
        return *this;
    }

    OnlineParameters & with_coarse_rhs(bool use_coarse_rhs)
    {
        this->use_coarse_rhs = use_coarse_rhs;
        return *this;
    }

    OnlineParameters & with_coarse_mass_matrix(bool use_coarse_mass_matrix)
    {
        this->use_coarse_mass_matrix = use_coarse_mass_matrix;
        return *this;
    }

    OnlineParameters & with_iterative_tolerance(double iterative_tolerance)
    {
        this->iterative_tolerance = iterative_tolerance;
        return *this;
    }

    OnlineParameters & with_error_measurement(bool measure_error)
    {
        this->measure_error = measure_error;
        return *this;
    }

    double dt() const
    {
        return end_time / (sample_count - 1);
    }
};

struct ExperimentInput
{
    std::string                         name;
    OfflineParameters                   offline_parameters;
    boost::optional<OnlineParameters>   online_parameters;

    ExperimentInput & with_name(std::string name)
    {
        this->name = name;
        return *this;
    }

    ExperimentInput & with_offline_parameters(OfflineParameters parameters)
    {
        this->offline_parameters = parameters;
        return *this;
    }

    ExperimentInput & with_online_parameters(OnlineParameters parameters)
    {
        this->online_parameters = parameters;
        return *this;
    }
};

struct MeshDetails
{
    int num_coarse_vertices;
    int num_coarse_elements;
    int num_fine_vertices;
    int num_fine_elements;

    // h is diameter of element
    double h_max_coarse;
    double h_min_coarse;
    double h_max_fine;
    double h_min_fine;

    MeshDetails() : num_coarse_vertices(0),
                    num_coarse_elements(0),
                    num_fine_vertices(0),
                    num_fine_elements(0),
                    h_max_coarse(0.0),
                    h_min_coarse(0.0),
                    h_max_fine(0.0),
                    h_min_fine(0.0)
    { }
};

std::pair<double, double> extremal_diameters(const crest::IndexedMesh<double, int> & mesh)
{
    double hmin_squared = std::numeric_limits<double>::max();
    double hmax_squared = 0.0;

    for (int element = 0; element < mesh.num_elements(); ++element)
    {
        const auto triangle = mesh.triangle_for(element);
        const auto diameter_squared = crest::diameter_squared(triangle);
        if (diameter_squared > hmax_squared) {
            hmax_squared = diameter_squared;
        }
        if (diameter_squared < hmin_squared) {
            hmin_squared = diameter_squared;
        }
    }

    return std::make_pair(std::sqrt(hmin_squared), std::sqrt(hmax_squared));
};

MeshDetails details_from_meshes(const crest::IndexedMesh<double, int> & coarse,
                                const crest::IndexedMesh<double, int> & fine)
{
    MeshDetails details;
    details.num_coarse_elements = coarse.num_elements();
    details.num_coarse_vertices = coarse.num_vertices();
    details.num_fine_elements = fine.num_elements();
    details.num_fine_vertices = fine.num_vertices();

    std::tie(details.h_min_coarse, details.h_max_coarse) = extremal_diameters(coarse);
    std::tie(details.h_min_fine, details.h_max_fine) = extremal_diameters(fine);

    return details;
}

struct ErrorSummary
{
    double l2;
    double h1;
    double h1_semi;

    ErrorSummary & with_l2(double error)
    {
        this->l2 = error;
        return *this;
    }

    ErrorSummary & with_h1(double error)
    {
        this->h1 = error;
        return *this;
    }

    ErrorSummary & with_h1_semi(double error)
    {
        this->h1_semi = error;
        return *this;
    }
};

struct OfflineTiming
{
    double mesh_construction;
    double basis_construction;
    double total;

    OfflineTiming() : mesh_construction(0.0), basis_construction(0.0), total(0.0) {}
};

struct OfflineResult
{
    MeshDetails mesh_details;
    OfflineTiming timing;

    std::unordered_map<std::string, crest::AccumulatedDensityHistogram> stats;

    OfflineResult & with_mesh_details(MeshDetails mesh_details)
    {
        this->mesh_details = mesh_details;
        return *this;
    }

    OfflineResult & with_timing(OfflineTiming timing)
    {
        this->timing = timing;
        return *this;
    }

    OfflineResult & with_stats(std::unordered_map<std::string, crest::AccumulatedDensityHistogram> stats)
    {
        this->stats = stats;
        return *this;
    }
};

typedef crest::wave::SolveTiming OnlineTiming;

struct OnlineResult
{
    ErrorSummary error_summary;
    OnlineTiming timing;
    bool converged;

    OnlineResult & with_error_summary(ErrorSummary summary)
    {
        this->error_summary = summary;
        return *this;
    }

    OnlineResult & with_timing(OnlineTiming timing)
    {
        this->timing = timing;
        return *this;
    }

    OnlineResult & with_convergence(bool converged)
    {
        this->converged = converged;
        return *this;
    }

    OnlineResult() : converged(false) {}
};

struct ExperimentResult
{
    std::string                         name;
    OfflineParameters                   offline_parameters;
    boost::optional<OnlineParameters>   online_parameters;
    OfflineResult                       offline_result;
    boost::optional<OnlineResult>       online_result;

    ExperimentResult & with_name(std::string name)
    {
        this->name = name;
        return *this;
    }

    ExperimentResult & with_offline_parameters(OfflineParameters param)
    {
        this->offline_parameters = param;
        return *this;
    }

    ExperimentResult & with_online_parameters(OnlineParameters param)
    {
        this->online_parameters = param;
        return *this;
    }

    ExperimentResult & with_online_parameters(boost::optional<OnlineParameters> param)
    {
        this->online_parameters = param;
        return *this;
    }

    ExperimentResult & with_offline_result(OfflineResult result)
    {
        this->offline_result = result;
        return *this;
    }

    ExperimentResult & with_online_result(OnlineResult result)
    {
        this->online_result = result;
        return *this;
    }

    ExperimentResult & with_online_result(boost::optional<OnlineResult> result)
    {
        this->online_result = result;
        return *this;
    }
};

void verify_parameter_validity(const OfflineParameters & parameters);
void verify_parameter_validity(const OnlineParameters & parameters);

class Experiment
{
public:
    virtual ~Experiment() {}

    OfflineResult run_offline(const OfflineParameters & parameters)
    {
        crest::Timer timer;
        verify_parameter_validity(parameters);
        auto result = solve_offline(parameters);
        result.timing.total = timer.elapsed();
        return result;
    }

    OnlineResult run_online(const OnlineParameters & parameters,
                            crest::wave::Integrator<double> & integrator)
    {
        verify_parameter_validity(parameters);
        return solve_online(parameters, integrator);
    }

protected:
    virtual OfflineResult solve_offline(const OfflineParameters & parameters) = 0;
    virtual OnlineResult solve_online(const OnlineParameters & parameters,
                                      crest::wave::Integrator<double> & integrator) = 0;

    /**
     * Helper function for implementing solve()
     */
    template <typename BasisImpl, typename Function2dt, typename Function2dt_x, typename Function2dt_y>
    OnlineResult solve_and_analyze(const Function2dt & u,
                                   const Function2dt_x & u_x,
                                   const Function2dt_y & u_y,
                                   const OnlineParameters & parameters,
                                   const crest::Basis<double, BasisImpl> & basis,
                                   const crest::wave::InitialConditions<double> & initial_conditions,
                                   const crest::wave::ConstrainedSystem<double> & system,
                                   crest::wave::Integrator<double> & integrator,
                                   const crest::wave::Initializer<double> & initializer,
                                   double assembly_time) const
    {
        const auto dt = parameters.dt();
        crest::wave::Parameters<double> param;
        param.num_samples = parameters.sample_count;
        param.dt = dt;

        if (parameters.measure_error) {
            const auto error_transformer = crest::wave::make_error_transformer<4>(
                    basis, u, u_x, u_y, parameters.max_error_before_abort);
            auto result = crest::wave::solve(system, initial_conditions,
                                             integrator, initializer, param, error_transformer);

            // Augment the result with the given assembly time. Yes, I know, this is very hacky and
            // very poorly designed (last minute additions)
            result.timing.assembly_time = assembly_time;

            if (result.converged) {


                std::vector<double> l2_error_at_each_step;
                std::vector<double> h1_semi_error_at_each_step;
                std::vector<double> h1_error_at_each_step;

                for (const auto sample_error : result.result)
                {
                    l2_error_at_each_step.push_back(sample_error.l2);
                    h1_semi_error_at_each_step.push_back(sample_error.h1_semi);
                    h1_error_at_each_step.push_back(sample_error.h1);
                }

                ErrorSummary errors;

                switch (param.num_samples) {
                    case 1:
                        errors.h1 = h1_error_at_each_step.back();
                        errors.l2 = l2_error_at_each_step.back();
                        errors.h1_semi = h1_semi_error_at_each_step.back();
                        break;

                    default:
                        // TODO: Support even number of samples. Simpsons will currently throw when it
                        // is given an even number of samples
                        errors.l2 = crest::composite_simpsons(l2_error_at_each_step.begin(),
                                                              l2_error_at_each_step.end(),
                                                              dt);
                        errors.h1_semi = crest::composite_simpsons(h1_semi_error_at_each_step.begin(),
                                                                   h1_semi_error_at_each_step.end(),
                                                                   dt);
                        errors.h1 = crest::composite_simpsons(h1_error_at_each_step.begin(),
                                                              h1_error_at_each_step.end(),
                                                              dt);
                        break;
                }

                return OnlineResult()
                        .with_error_summary(errors)
                        .with_timing(result.timing)
                        .with_convergence(result.converged);
            } else {
                // If the simulation did not converge, we do not want to compute an error at all,
                // particularly because we might have an odd number of samples (in which case the above would throw)
                // This last part is of course unfortunate, but I don't want to spend time on fixing that at the moment.
                return OnlineResult()
                        .with_timing(result.timing)
                        .with_convergence(result.converged);
            }

        } else {
            const auto ignore_transformer = crest::wave::IgnoreTransformer<double>();
            auto result = crest::wave::solve(system, initial_conditions, integrator,
                                             initializer, param, ignore_transformer);
            // Augment the result with the given assembly time. Yes, I know, this is very hacky and
            // very poorly designed (last minute additions)
            result.timing.assembly_time = assembly_time;
            return OnlineResult()
                    .with_timing(result.timing)
                    .with_convergence(result.converged);
        };
    };
};

inline void verify_parameter_validity(const OnlineParameters & parameters)
{
    if (parameters.end_time <= 0.0 || !std::isfinite(parameters.end_time))
    {
        throw std::invalid_argument("Final time T must be finite and greater than zero.");
    }

    if (parameters.sample_count < 1) {
        throw std::invalid_argument("Number of samples N must be 1 or greater.");
    }

    if (parameters.sample_count % 2 == 0) {
        throw std::invalid_argument("Number of samples must be an odd number.");
    }
}

inline void verify_parameter_validity(const OfflineParameters & parameters)
{
    if (parameters.mesh_resolution <= 0.0 || !std::isfinite(parameters.mesh_resolution))
    {
        throw std::invalid_argument("Spatial resolution h must be finite and greater than zero.");
    }
}

