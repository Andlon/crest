#include <crest/wave/solver.hpp>

#include <crest/wave/constraints.hpp>

#include <crest/basis/lagrange_basis2d.hpp>
#include <crest/geometry/refinement.hpp>
#include <crest/quadrature/simpsons.hpp>

#include <iostream>
#include <iomanip>
#include <string>

using std::cout;
using std::endl;
using std::cerr;
using std::setw;

constexpr double PI = 3.1415926535897932385;
constexpr double PI_SQUARED = PI * PI;

crest::IndexedMesh<double, int> minimal_unit_square()
{
    typedef crest::IndexedMesh<double, int>::Vertex Vertex;
    typedef crest::IndexedMesh<double, int>::Element Element;

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

    return crest::IndexedMesh<double, int>(std::move(vertices), std::move(elements));
};

void homogeneous_standard_lagrange(double T, double h, uint64_t num_samples)
{
    using std::sin;
    using std::cos;

    const auto u = [] (auto t, auto x, auto y)
    {
        return cos(2 * PI * t) * sin(2 * PI * x) * sin(2 * PI * y);
    };

    const auto u_x = [] (auto t, auto x, auto y)
    {
        return 2 * PI * cos(2 * PI * t) * cos(2 * PI * x) * sin(2 * PI * y);
    };

    const auto u_y = [] (auto t, auto x, auto y)
    {
        return 2 * PI * cos(2 * PI * t) * sin(2 * PI * x) * cos(2 * PI * y);
    };

    const auto u0 = [] (auto x, auto y)
    {
        return sin(2.0 * PI * x) * sin(2.0 * PI * y);
    };

    const auto v0 = [] (auto, auto)
    {
        return 0.0;
    };

    const auto u0_tt = [] (auto x, auto y)
    {
        return - 8.0 * PI_SQUARED * sin(2.0 * PI * x) * sin(2.0 * PI * y);
    };

    const auto f = [] (auto t, auto x, auto y)
    {
        return 4.0 * PI_SQUARED * cos(2.0 * PI * t) * sin(2.0 * PI * x) * sin(2.0 * PI * y);
    };

    const auto dt = T / (num_samples - 1);

    const auto mesh = crest::bisect_to_tolerance<double>(minimal_unit_square(), h);
    const auto basis = crest::LagrangeBasis2d<double>(mesh);

    crest::wave::InitialConditions<double> ic;
    ic.u0_h = basis.interpolate(u0);
    ic.v0_h = basis.interpolate(v0);
    ic.u0_tt_h = basis.interpolate(u0_tt);

    const auto bc = crest::wave::HomogeneousDirichlet<double>(basis);
    const crest::wave::SeriesExpansionInitializer<double> initializer;
    crest::wave::CrankNicolson<double> integrator;

    const auto load = crest::wave::make_basis_load_function<4>(f, basis);

    crest::wave::Parameters<double> param;
    param.num_samples = num_samples;
    param.dt = dt;

    const auto error_transformer = crest::wave::make_error_transformer<4>(basis, u, u_x, u_y);

    const auto result = crest::wave::solve(basis, ic, load, bc, integrator, initializer, param, error_transformer);

    std::vector<double> l2_error_at_each_step;
    std::vector<double> h1_error_at_each_step;

    for (uint64_t i = 0; i < param.num_samples; ++i)
    {
        const auto e_i = result.result[i];
        l2_error_at_each_step.push_back(e_i.l2);
        h1_error_at_each_step.push_back(e_i.h1);
    }

    const auto total_l2_error = crest::composite_simpsons(l2_error_at_each_step.begin(),
                                                          l2_error_at_each_step.end(),
                                                          dt);
    const auto total_h1_error = crest::composite_simpsons(h1_error_at_each_step.begin(),
                                                          h1_error_at_each_step.end(),
                                                          dt);

    cout << endl
         << "# vertices: " << mesh.num_vertices() << endl
         << "# elements: " << mesh.num_elements() << endl
         << "dt:         " << dt << endl
         << "L2 error:   " << total_l2_error << endl
         << "H1 error:   " << total_h1_error << endl;
}

struct ExperimentParameters
{
    double T;
    double h;
    uint64_t N;
};

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
        homogeneous_standard_lagrange(params.T, params.h, params.N);
    }
    catch (const std::exception & e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
