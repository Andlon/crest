#pragma once

#include "experiment.hpp"

#include <crest/wave/solver.hpp>

#include <crest/wave/constraints.hpp>

#include <crest/basis/lagrange_basis2d.hpp>
#include <crest/geometry/refinement.hpp>
#include <crest/quadrature/simpsons.hpp>

// TODO: Move this somewhere else
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

class HomogeneousLoadUnitSquare : public Experiment
{
protected:
    virtual ExperimentOutput solve(const ExperimentParameters & parameters,
                                   crest::wave::Integrator<double> & integrator) const override
    {
        constexpr double PI = 3.1415926535897932385;
        constexpr double PI_SQUARED = PI * PI;

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

        const auto u0 = [&u] (auto x, auto y)
        {
            return u(0.0, x, y);
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

        const auto h = parameters.mesh_resolution;

        const auto mesh = crest::bisect_to_tolerance<double>(minimal_unit_square(), h);
        const auto basis = crest::LagrangeBasis2d<double>(mesh);

        crest::wave::InitialConditions<double> ic;
        ic.u0_h = basis.interpolate(u0);
        ic.v0_h = basis.interpolate(v0);
        ic.u0_tt_h = basis.interpolate(u0_tt);

        const auto bc = crest::wave::HomogeneousDirichlet<double>(basis);
        const auto initializer = crest::wave::SeriesExpansionInitializer<double>();
        const auto load = crest::wave::make_basis_load_function<4>(f, basis);

        return solve_and_analyze(u, u_x, u_y, parameters, mesh, basis, ic, load, bc, integrator, initializer);
    }
};
