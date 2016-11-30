#pragma once

#include "experiment.hpp"

#include <crest/wave/solver.hpp>

#include <crest/wave/constraints.hpp>

#include <crest/basis/lagrange_basis2d.hpp>
#include <crest/basis/homogenized_basis.hpp>
#include <crest/geometry/refinement.hpp>
#include <crest/quadrature/simpsons.hpp>

std::pair<crest::IndexedMesh<double, int>, std::vector<crest::ReentrantCorner<double, int>>>
minimal_l_shaped()
{
    typedef crest::IndexedMesh<double, int>::Vertex Vertex;
    typedef crest::IndexedMesh<double, int>::Element Element;

    const auto a = 0.5;

    const std::vector<Vertex> vertices {
            Vertex(0.0, 0.0),
            Vertex(a, 0.0),
            Vertex(a, a),
            Vertex(0.0, a),
            Vertex(-a, a),
            Vertex(-a, 0.0),
            Vertex(-a, -a),
            Vertex(0.0, -a)
    };

    const std::vector<Element> elements {
            Element({0, 1, 2}),
            Element({0, 3, 2}),
            Element({0, 3, 4}),
            Element({0, 5, 4}),
            Element({0, 5, 6}),
            Element({0, 7, 6})
    };

    const auto mesh = crest::IndexedMesh<double, int>(std::move(vertices), std::move(elements));

    constexpr double PI = 3.1415926535897932385;

    const std::vector<crest::ReentrantCorner<double, int>> corners {
            crest::ReentrantCorner<double, int>(0, 3 * PI / 2.0)
    };

    return std::make_pair(mesh, corners);
};

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

        const auto load = crest::wave::make_basis_load_function<4>(f, basis);
        const auto bc = crest::wave::HomogeneousDirichletBC<double>::assemble(basis, load);
        const auto initializer = crest::wave::SeriesExpansionInitializer<double>();

        return solve_and_analyze(u, u_x, u_y, parameters, mesh, basis, ic, bc, integrator, initializer);
    }
};

class InhomogeneousLoadUnitSquare : public Experiment
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
            return cos(2 * PI * t * x) + cos(2 * PI * t) * sin(2 * PI * x) * sin(2 * PI * y);
        };

        const auto u_x = [] (auto t, auto x, auto y)
        {
            return -2 * PI * t * sin(2 * PI * t * x) + 2 * PI * cos(2 * PI * t) * cos(2 * PI * x) * sin(2 * PI * y);
        };

        const auto u_y = [] (auto t, auto x, auto y)
        {
            return 2 * PI * cos(2 * PI * t) * sin(2 * PI * x) * cos(2 * PI * y);
        };

        const auto u0 = [] (auto x, auto y)
        {
            return 1 + sin(2 * PI * x) * sin(2 * PI * y);
        };

        const auto v0 = [] (auto  , auto  )
        {
            return 0.0;
        };

        const auto f = [] (auto t, auto x, auto y)
        {
            return 4 * PI_SQUARED * cos(2 * PI * t) * sin(2 * PI * x) * sin(2 * PI * y) +
                   4 * PI_SQUARED * cos(2 * PI * t * x) * (t * t - x * x);
        };

        const auto u0_tt = [] (auto x, auto y)
        {
            return - 4 * PI_SQUARED * (x * x) - 8 * PI_SQUARED * sin(2 * PI * x) * sin(2 * PI * y);
        };


        const auto g = [] (auto t, auto x, auto  )
        {
            return cos(2 * PI * t * x);
        };

        const auto g_tt = [] (auto t, auto x, auto  )
        {
            return - 4 * PI_SQUARED * (x * x) * cos(2 * PI * t * x);
        };

        const auto h = parameters.mesh_resolution;

        const auto mesh = crest::bisect_to_tolerance<double>(minimal_unit_square(), h);
        const auto basis = crest::LagrangeBasis2d<double>(mesh);

        crest::wave::InitialConditions<double> ic;
        ic.u0_h = basis.interpolate(u0);
        ic.v0_h = basis.interpolate(v0);
        ic.u0_tt_h = basis.interpolate(u0_tt);

        const auto load = crest::wave::make_basis_load_function<4>(f, basis);
        // TODO: Fix this factory function mess. Generalize it.
        const auto bc = crest::wave::make_inhomogeneous_dirichlet(basis, load, g, g_tt);
        const auto initializer = crest::wave::SeriesExpansionInitializer<double>();

        return solve_and_analyze(u, u_x, u_y, parameters, mesh, basis, ic, bc, integrator, initializer);
    }
};

class HomogenizedLShaped : public Experiment
{
protected:
    virtual ExperimentOutput solve(const ExperimentParameters & parameters,
                                   crest::wave::Integrator<double> & integrator) const override
    {
        constexpr double PI = 3.1415926535897932385;
        constexpr double PI_SQUARED = PI * PI;

        using std::sin;
        using std::cos;
        using std::sqrt;
        using std::pow;

        const auto map_to_theta = [] (auto x, auto y)
        {
            // atan2 returns values in the interval (-pi, pi), but we need [0, 2pi)
            const auto theta_symmetric_interval = std::atan2(y, x);
            if (theta_symmetric_interval < 0.0) {
                return 2 * PI + theta_symmetric_interval;
            } else {
                return theta_symmetric_interval;
            }
        };

        const auto u = [map_to_theta] (auto t, auto x, auto y)
        {
            const auto r2 = x * x + y * y;
            const auto r_to_2_by_3 = std::pow(r2, 1.0 / 3.0);
            const auto theta = map_to_theta(x, y);
            return cos(2 * PI * t) * sin(2 * theta / 3.0) * r_to_2_by_3;
        };

        const auto u_tt = [u] (auto t, auto x, auto y)
        {
            return - 4 * PI_SQUARED * u(t, x, y);
        };

        const auto f = [u_tt] (auto t, auto x, auto y)
        {
            return u_tt(t, x, y);
        };

        const auto u0 = [u] (auto x, auto y)
        {
            return u(0, x, y);
        };

        const auto v0 = [] (auto  , auto  )
        {
            return 0.0;
        };

        const auto u0_tt = [u_tt] (auto x, auto y)
        {
            return u_tt(0, x, y);
        };

        const auto g = [u] (auto t, auto x, auto y)
        {
            return u(t, x, y);
        };

        const auto g_tt = [u_tt] (auto t, auto x, auto y)
        {
            return u_tt(t, x, y);
        };

        const auto u_r = [] (auto t, auto r, auto theta)
        {
            return (2.0 / 3.0) * cos(2 * PI * t) * sin(2 * theta / 3.0) * pow(r, - 1.0 / 3.0);
        };

        const auto u_theta = [] (auto t, auto r, auto theta)
        {
            return (2.0 / 3.0) * cos(2 * PI * t) * cos(2 * theta / 3.0) * pow(r, 2.0 / 3.0);
        };

        const auto u_x = [map_to_theta, u_theta, u_r] (auto t, auto x, auto y)
        {
            const auto r2 = x * x + y * y;
            const auto r = sqrt(r2);
            const auto theta = map_to_theta(x, y);
            // Use chain rule
            return - (y / r2) * u_theta(t, r, theta) + (x / r) * u_r(t, r, theta);
        };

        const auto u_y = [map_to_theta, u_theta, u_r] (auto t, auto x, auto y)
        {
            const auto r2 = x * x + y * y;
            const auto r = sqrt(r2);
            const auto theta = map_to_theta(x, y);
            // Use chain rule
            return (x / r2) * u_theta(t, r, theta) + (y / r) * u_r(t, r, theta);
        };

        const auto h = parameters.mesh_resolution;

        const auto l_shaped = minimal_l_shaped();
        const auto initial_mesh = l_shaped.first;
        const auto initial_mesh_corners = l_shaped.second;

        const auto coarse_fine_meshes = crest::threshold(initial_mesh, h, initial_mesh_corners);
        const auto oversampling = 4;
        const auto basis = crest::HomogenizedBasis<double>(coarse_fine_meshes.coarse,
                                                           coarse_fine_meshes.fine,
                                                           oversampling);

        crest::wave::InitialConditions<double> ic;
        ic.u0_h = basis.interpolate(u0);
        ic.v0_h = basis.interpolate(v0);
        ic.u0_tt_h = basis.interpolate(u0_tt);

        const auto load = crest::wave::make_basis_load_function<4>(f, basis);
        // TODO: Fix this factory function mess. Generalize it.
        const auto bc = crest::wave::make_inhomogeneous_dirichlet(basis, load, g, g_tt);
        const auto initializer = crest::wave::SeriesExpansionInitializer<double>();

        return solve_and_analyze(u, u_x, u_y, parameters, coarse_fine_meshes.coarse,
                                 basis, ic, bc, integrator, initializer);
    }
};

class StandardLShaped : public Experiment
{
protected:
    virtual ExperimentOutput solve(const ExperimentParameters & parameters,
                                   crest::wave::Integrator<double> & integrator) const override
    {
        constexpr double PI = 3.1415926535897932385;
        constexpr double PI_SQUARED = PI * PI;

        using std::sin;
        using std::cos;
        using std::sqrt;
        using std::pow;

        const auto map_to_theta = [] (auto x, auto y)
        {
            // atan2 returns values in the interval (-pi, pi), but we need [0, 2pi)
            const auto theta_symmetric_interval = std::atan2(y, x);
            if (theta_symmetric_interval < 0.0) {
                return 2 * PI + theta_symmetric_interval;
            } else {
                return theta_symmetric_interval;
            }
        };

        const auto u = [map_to_theta] (auto t, auto x, auto y)
        {
            const auto r2 = x * x + y * y;
            const auto r_to_2_by_3 = std::pow(r2, 1.0 / 3.0);
            const auto theta = map_to_theta(x, y);
            return cos(2 * PI * t) * sin(2 * theta / 3.0) * r_to_2_by_3;
        };

        const auto u_tt = [u] (auto t, auto x, auto y)
        {
            return - 4 * PI_SQUARED * u(t, x, y);
        };

        const auto f = [u_tt] (auto t, auto x, auto y)
        {
            return u_tt(t, x, y);
        };

        const auto u0 = [u] (auto x, auto y)
        {
            return u(0, x, y);
        };

        const auto v0 = [] (auto  , auto  )
        {
            return 0.0;
        };

        const auto u0_tt = [u_tt] (auto x, auto y)
        {
            return u_tt(0, x, y);
        };

        const auto g = [u] (auto t, auto x, auto y)
        {
            return u(t, x, y);
        };

        const auto g_tt = [u_tt] (auto t, auto x, auto y)
        {
            return u_tt(t, x, y);
        };

        const auto u_r = [] (auto t, auto r, auto theta)
        {
            return (2.0 / 3.0) * cos(2 * PI * t) * sin(2 * theta / 3.0) * pow(r, - 1.0 / 3.0);
        };

        const auto u_theta = [] (auto t, auto r, auto theta)
        {
            return (2.0 / 3.0) * cos(2 * PI * t) * cos(2 * theta / 3.0) * pow(r, 2.0 / 3.0);
        };

        const auto u_x = [map_to_theta, u_theta, u_r] (auto t, auto x, auto y)
        {
            const auto r2 = x * x + y * y;
            const auto r = sqrt(r2);
            const auto theta = map_to_theta(x, y);
            // Use chain rule
            return - (y / r2) * u_theta(t, r, theta) + (x / r) * u_r(t, r, theta);
        };

        const auto u_y = [map_to_theta, u_theta, u_r] (auto t, auto x, auto y)
        {
            const auto r2 = x * x + y * y;
            const auto r = sqrt(r2);
            const auto theta = map_to_theta(x, y);
            // Use chain rule
            return (x / r2) * u_theta(t, r, theta) + (y / r) * u_r(t, r, theta);
        };

        const auto h = parameters.mesh_resolution;

        const auto l_shaped = minimal_l_shaped();
        const auto initial_mesh = l_shaped.first;
        const auto initial_mesh_corners = l_shaped.second;

        const auto coarse_fine_meshes = crest::threshold(initial_mesh, h, initial_mesh_corners);
        const auto mesh = coarse_fine_meshes.coarse;

        const auto basis = crest::LagrangeBasis2d<double>(coarse_fine_meshes.fine);

        crest::wave::InitialConditions<double> ic;
        ic.u0_h = basis.interpolate(u0);
        ic.v0_h = basis.interpolate(v0);
        ic.u0_tt_h = basis.interpolate(u0_tt);

        const auto load = crest::wave::make_basis_load_function<4>(f, basis);
        // TODO: Fix this factory function mess. Generalize it.
        const auto bc = crest::wave::make_inhomogeneous_dirichlet(basis, load, g, g_tt);
        const auto initializer = crest::wave::SeriesExpansionInitializer<double>();

        return solve_and_analyze(u, u_x, u_y, parameters, coarse_fine_meshes.fine,
                                 basis, ic, bc, integrator, initializer);
    }
};
