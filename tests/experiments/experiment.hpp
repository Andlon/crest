#pragma once

#include <crest/wave/solver.hpp>
#include <crest/geometry/indexed_mesh.hpp>
#include <crest/quadrature/simpsons.hpp>

#include <string>
#include <cmath>

struct ExperimentParameters
{
    // Run the experiment for the time interval [0, end_time]
    double end_time;
    // With mesh resolution h
    double mesh_resolution;
    // And number of time samples (including t = 0)
    uint64_t sample_count;

    int oversampling;

    std::string basis_export_file;
    std::string basis_import_file;

    // TODO: Enable this switch. For now we always compute Bochner errors
    // bool compute_bochner;

    double dt() const
    {
        return end_time / (sample_count - 1);
    }

    ExperimentParameters & with_end_time(double t)
    {
        this->end_time = t;
        return *this;
    }

    ExperimentParameters & with_mesh_resolution(double res)
    {
        this->mesh_resolution = res;
        return *this;
    }

    ExperimentParameters & with_sample_count(uint64_t count)
    {
        this->sample_count = count;
        return *this;
    }

    ExperimentParameters & with_oversampling(int oversampling)
    {
        this->oversampling = oversampling;
        return *this;
    }

    ExperimentParameters & with_basis_export_file(std::string basis_export_file)
    {
        this->basis_export_file = basis_export_file;
        return *this;
    }

    ExperimentParameters()
    {
        end_time = 0.0;
        mesh_resolution = 0.0;
        sample_count = 0;
        oversampling = 0;
    }
};

struct MeshDetails
{
    int num_vertices;
    int num_elements;
};

struct ErrorSummary
{
    double l2;
    double h1;
    double h1_semi;

    /*
     * If bochner is true, then the errors represent error in time as well as space.
     * That is, the errors are computed in terms of Bochner norms.
     * Otherwise, only the error at the final sample was measured.
     *
     * TODO: Enable this feature
     */
//    bool bochner;

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

struct ExperimentOutput
{
    MeshDetails mesh_details;
    ErrorSummary error_summary;
};

struct ExperimentResult
{
    std::string             name;
    ExperimentParameters    parameters;
    MeshDetails             mesh_details;
    ErrorSummary            error_summary;

    ExperimentResult & with_name(std::string name)
    {
        this->name = name;
        return *this;
    }

    ExperimentResult & with_parameters(const ExperimentParameters & param)
    {
        this->parameters = param;
        return *this;
    }

    ExperimentResult & with_mesh_details(const MeshDetails & details)
    {
        this->mesh_details = details;
        return *this;
    }

    ExperimentResult & with_error_summary(const ErrorSummary & summary)
    {
        this->error_summary = summary;
        return *this;
    }
};

void verify_parameter_validity(const ExperimentParameters & parameters);

class Experiment
{
public:
    virtual ~Experiment() {}

    ExperimentResult run(const ExperimentParameters & parameters,
                         crest::wave::Integrator<double> & integrator) const
    {
        verify_parameter_validity(parameters);
        const auto output = solve(parameters, integrator);
        ExperimentResult result;
        result.parameters = parameters;
        result.mesh_details = output.mesh_details;
        result.error_summary = output.error_summary;
        return result;
    }

//    virtual std::string name() const = 0;

protected:
    virtual ExperimentOutput solve(const ExperimentParameters & parameters,
                                   crest::wave::Integrator<double> & integrator) const = 0;

    /**
     * Helper function for implementing solve()
     */
    template <typename BasisImpl, typename Function2dt, typename Function2dt_x, typename Function2dt_y>
    ExperimentOutput solve_and_analyze(const Function2dt & u,
                                       const Function2dt_x & u_x,
                                       const Function2dt_y & u_y,
                                       const ExperimentParameters & parameters,
                                       const crest::IndexedMesh<double, int> mesh,
                                       const crest::Basis<double, BasisImpl> & basis,
                                       const crest::wave::InitialConditions<double> & initial_conditions,
                                       const crest::wave::ConstrainedSystem<double> & system,
                                       crest::wave::Integrator<double> & integrator,
                                       const crest::wave::Initializer<double> & initializer) const
    {
        const auto dt = parameters.dt();
        const auto error_transformer = crest::wave::make_error_transformer<4>(basis, u, u_x, u_y);

        crest::wave::Parameters<double> param;
        param.num_samples = parameters.sample_count;
        param.dt = dt;
        const auto result = crest::wave::solve(system, initial_conditions,
                                               integrator, initializer, param, error_transformer);

        std::vector<double> l2_error_at_each_step;
        std::vector<double> h1_semi_error_at_each_step;
        std::vector<double> h1_error_at_each_step;

        for (const auto sample_error : result.result)
        {
            l2_error_at_each_step.push_back(sample_error.l2);
            h1_semi_error_at_each_step.push_back(sample_error.h1_semi);
            h1_error_at_each_step.push_back(sample_error.h1);
        }

        MeshDetails mesh_details;
        mesh_details.num_elements = mesh.num_elements();
        mesh_details.num_vertices = mesh.num_vertices();

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

        ExperimentOutput output;
        output.error_summary = errors;
        output.mesh_details = mesh_details;

        return output;
    };
};

inline void verify_parameter_validity(const ExperimentParameters & parameters)
{
    if (parameters.end_time <= 0.0 || !std::isfinite(parameters.end_time))
    {
        throw std::invalid_argument("Final time T must be finite and greater than zero.");
    }

    if (parameters.sample_count < 1) {
        throw std::invalid_argument("Number of samples N must be 1 or greater.");
    }

    if (parameters.mesh_resolution <= 0.0 || !std::isfinite(parameters.mesh_resolution))
    {
        throw std::invalid_argument("Spatial resolution h must be finite and greater than zero.");
    }
}

