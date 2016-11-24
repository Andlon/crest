#include <crest/wave/solver.hpp>

#include <crest/wave/constraints.hpp>

#include <crest/basis/lagrange_basis2d.hpp>
#include <crest/geometry/refinement.hpp>
#include <crest/quadrature/simpsons.hpp>

#include <experiments/experiments.hpp>

#include <iostream>
#include <iomanip>
#include <string>

using std::cout;
using std::endl;
using std::cerr;
using std::setw;


void homogeneous_standard_lagrange(const ExperimentParameters & param)
{
    HomogeneousLoadUnitSquare experiment;
    crest::wave::CrankNicolson<double> integrator;
    const auto result = experiment.run(param, integrator);

    cout << endl
         << "# vertices:    " << result.mesh_details.num_vertices << endl
         << "# elements:    " << result.mesh_details.num_elements << endl
         << "dt:            " << result.parameters.dt() << endl
         << "L2 error:      " << result.error_summary.l2_error << endl
         << "H1-semi error: " << result.error_summary.h1_error_semi << endl
         << "H1 error:      " << result.error_summary.h1_error << endl;
}

ExperimentParameters parse_cmd_arguments(std::string T_str, std::string h_str, std::string N_str)
{
    const auto T = std::stof(T_str);
    const auto h = std::stof(h_str);
    const uint64_t num_samples = std::stoull(N_str);

    if (T > 0.0 && h > 0.0 && num_samples >= 2)
    {
        ExperimentParameters param;
        param.T = T;
        param.h = h;
        param.N = num_samples;

        return param;
    }

    throw std::invalid_argument("Invalid parameters.");
}

int main(int argc, char ** argv)
{
    if (argc != 4)
    {
        cout << "Invalid number of arguments." << endl;
        return 1;
    }

    ExperimentParameters params;

    try {
        params = parse_cmd_arguments(std::string(argv[1]), std::string(argv[2]), std::string(argv[3]));
    }
    catch (const std::invalid_argument & e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    cout << "Parameters: " << endl
         << "T: " << setw(8) << params.T << " (end time)" << endl
         << "h: " << setw(8) << params.h << " (spatial discretization)" << endl
         << "N: " << setw(8) << params.N << " (number of temporal samples)" << endl;

    try {
        homogeneous_standard_lagrange(params);
    }
    catch (const std::exception & e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
