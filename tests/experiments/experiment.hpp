#pragma once

#include <crest/wave/solver.hpp>

#include <string>
#include <cmath>

struct ExperimentParameters
{
    // Run the experiment for the time interval [0, T]
    double T;
    // With mesh resolution h
    double h;
    // And number of time steps N
    uint64_t N;

    // TODO: Enable this switch. For now we always compute Bochner errors
    // bool compute_bochner;

    double dt() const
    {
        return T / (N - 1);
    }
};

struct MeshDetails
{
    int num_vertices;
    int num_elements;
};

struct ErrorSummary
{
    double l2_error;
    double h1_error;
    double h1_error_semi;

    /*
     * If bochner is true, then the errors represent error in time as well as space.
     * That is, the errors are computed in terms of Bochner norms.
     * Otherwise, only the error at the final sample was measured.
     */
    bool bochner;
};

struct ExperimentOutput
{
    MeshDetails mesh_details;
    ErrorSummary error_summary;
};

struct ExperimentResult
{
    ExperimentParameters    parameters;
    MeshDetails             mesh_details;
    ErrorSummary            error_summary;
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
                                       const crest::wave::LoadProvider<double> & load_provider,
                                       const crest::wave::ConstraintHandler<double> & constraint_handler,
                                       crest::wave::Integrator<double> & integrator,
                                       const crest::wave::Initializer<double> & initializer) const
    {
        const auto dt = parameters.dt();
        const auto error_transformer = crest::wave::make_error_transformer<4>(basis, u, u_x, u_y);

        crest::wave::Parameters<double> param;
        param.num_samples = parameters.N;
        param.dt = dt;
        const auto result = crest::wave::solve(basis, initial_conditions, load_provider, constraint_handler,
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

        const auto total_l2_error = crest::composite_simpsons(l2_error_at_each_step.begin(),
                                                              l2_error_at_each_step.end(),
                                                              dt);
        const auto total_h1_semi_error = crest::composite_simpsons(h1_semi_error_at_each_step.begin(),
                                                                   h1_semi_error_at_each_step.end(),
                                                                   dt);
        const auto total_h1_error = crest::composite_simpsons(h1_error_at_each_step.begin(),
                                                              h1_error_at_each_step.end(),
                                                              dt);

        MeshDetails mesh_details;
        mesh_details.num_elements = mesh.num_elements();
        mesh_details.num_vertices = mesh.num_vertices();

        ErrorSummary errors;
        errors.h1_error = total_h1_error;
        errors.l2_error = total_l2_error;
        errors.h1_error_semi = total_h1_semi_error;

        ExperimentOutput output;
        output.error_summary = errors;
        output.mesh_details = mesh_details;

        return output;
    };
};

inline void verify_parameter_validity(const ExperimentParameters & parameters)
{
    if (parameters.T <= 0.0 || !std::isfinite(parameters.T))
    {
        throw std::invalid_argument("Final time T must be finite and greater than zero.");
    }

    if (parameters.N < 1) {
        throw std::invalid_argument("Number of samples N must be 1 or greater.");
    }

    if (parameters.h <= 0.0 || !std::isfinite(parameters.h))
    {
        throw std::invalid_argument("Spatial resolution h must be finite and greater than zero.");
    }
}

