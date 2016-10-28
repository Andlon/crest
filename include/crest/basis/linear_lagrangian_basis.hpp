#pragma once

#include <cstdint>
#include <cmath>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/util/eigen_extensions.hpp>
#include <crest/quadrature/triquad.hpp>

namespace crest
{
    /**
     * Represents a linear Lagrangian basis for a 2D mesh. It is assumed that the boundary conditions imposed on the
     * finite element space spanned by the basis are homogeneous Dirichlet conditions. However, inhomogeneous
     * Dirichlet boundary conditions can be accomodated by writing the weak formulation in terms of
     * a sum of a homogeneous solution and a known function. The boundary_ versions of the stiffness_matrix and
     * mass_matrix functions are provided with this in mind.
     */
    class LagrangeBasis2d
    {
    public:
        Eigen::SparseMatrix<double> stiffness_matrix() const { return _stiffness_matrix; }

        Eigen::SparseMatrix<double> mass_matrix() const { return _mass_matrix; }

        Eigen::SparseMatrix<double> boundary_stiffness_matrix() const { return _boundary_stiffness_matrix; }

        Eigen::SparseMatrix<double> boundary_mass_matrix() const { return _boundary_mass_matrix; }

        size_t ndof() const { return _mesh.num_interior_vertices(); }

        template <typename Function2d>
        Eigen::VectorXd interpolate_free_nodes(const Function2d & f) const;

        template <int Strength=2, typename Function2d>
        Eigen::VectorXd compute_load(const Function2d & f) const;

        static LagrangeBasis2d assemble_from_mesh(IndexedMesh<double, int> mesh);

    private:
        explicit LagrangeBasis2d(IndexedMesh<double, int> mesh,
                                 Eigen::SparseMatrix<double> mass_interior_matrix,
                                 Eigen::SparseMatrix<double> mass_boundary_matrix,
                                 Eigen::SparseMatrix<double> stiffness_interior_matrix,
                                 Eigen::SparseMatrix<double> stiffness_boundary_matrix);

        IndexedMesh<double, int> _mesh;
        Eigen::SparseMatrix<double> _mass_matrix;
        Eigen::SparseMatrix<double> _boundary_mass_matrix;
        Eigen::SparseMatrix<double> _stiffness_matrix;
        Eigen::SparseMatrix<double> _boundary_stiffness_matrix;
    };

    namespace detail
    {
        template <typename Scalar>
        struct assembly_triplets {
            std::vector<Eigen::Triplet<Scalar>> stiffness_triplets;
            std::vector<Eigen::Triplet<Scalar>> mass_triplets;
        };

        template <typename Scalar>
        assembly_triplets<Scalar> assemble_linear_lagrangian_stiffness_triplets(
                const crest::IndexedMesh<Scalar, int> &mesh);

        struct interior_boundary_matrices {
            Eigen::SparseMatrix<double> interior_matrix;
            Eigen::SparseMatrix<double> boundary_matrix;
        };

        interior_boundary_matrices interior_and_boundary_matrices_from_triplets(
                std::vector<Eigen::Triplet<double>> triplets,
                const IndexedMesh<double, int> & mesh);
    }

    /*
     * IMPLEMENTATION BELOW
     */

    template <typename Scalar>
    detail::assembly_triplets<Scalar> detail::assemble_linear_lagrangian_stiffness_triplets(
            const crest::IndexedMesh<Scalar, int> &mesh)
    {
        const static Eigen::Matrix<Scalar, 3, 3> M_LOCAL_REF = (1.0 / 24.0) * (Eigen::Matrix3d()
                <<
                2.0, 1.0, 1.0,
                1.0, 2.0, 1.0,
                1.0, 1.0, 2.0
        ).finished().cast<Scalar>();

        const static Eigen::Matrix<Scalar, 3, 3> A11 = (1.0 / 2.0) * (Eigen::Matrix3d()
                <<
                1.0, 0.0, -1.0,
                0.0, 0.0, 0.0,
                -1.0, 0.0, 1.0
        ).finished().cast<Scalar>();

        const static Eigen::Matrix<Scalar, 3, 3> A12 = (1.0 / 2.0) * (Eigen::Matrix3d()
                <<
                0.0, 1.0, -1.0,
                1.0, 0.0, -1.0,
                -1.0, -1.0, 2.0
        ).finished().cast<Scalar>();

        const static Eigen::Matrix<Scalar, 3, 3> A22 = (1.0 / 2.0) * (Eigen::Matrix3d()
                <<
                0.0, 0.0, 0.0,
                0.0, 1.0, -1.0,
                0.0, -1.0, 1.0
        ).finished().cast<Scalar>();

        std::vector<Eigen::Triplet<Scalar>> mass_triplets;
        std::vector<Eigen::Triplet<Scalar>> stiffness_triplets;
        mass_triplets.reserve(3 * mesh.num_elements());
        stiffness_triplets.reserve(3 * mesh.num_elements());

        for (const auto & element : mesh.elements())
        {
            const auto a = mesh.vertices()[element.vertex_indices[0]];
            const auto b = mesh.vertices()[element.vertex_indices[1]];
            const auto c = mesh.vertices()[element.vertex_indices[2]];

            const auto v1 = a - c;
            const auto v2 = b - c;

            const Eigen::Matrix2d jacobian = (Eigen::Matrix2d() << v1.x, v2.x, v1.y, v2.y).finished();
            const Eigen::Matrix2d jacobian_inverse = jacobian.inverse();
            const Eigen::Matrix2d C = jacobian_inverse * jacobian_inverse.transpose();
            const auto abs_det_jacobian = std::abs(jacobian.determinant());

            const Eigen::Matrix<Scalar, 3, 3> A_local = abs_det_jacobian * (C(0, 0) * A11 + C(0, 1) * A12 + C(1, 1) * A22);
            const Eigen::Matrix<Scalar, 3, 3> M_local = abs_det_jacobian * M_LOCAL_REF;

            typedef Eigen::Triplet<Scalar> T;
            for (size_t i = 0; i < 3; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    const auto I = element.vertex_indices[i];
                    const auto J = element.vertex_indices[j];
                    mass_triplets.emplace_back(T(I, J, M_local(i, j)));
                    stiffness_triplets.emplace_back(T(I, J, A_local(i, j)));
                }
            }
        }

        return detail::assembly_triplets<Scalar> {
                std::move(stiffness_triplets),
                std::move(mass_triplets)
        };
    }

    inline LagrangeBasis2d LagrangeBasis2d::assemble_from_mesh(IndexedMesh<double, int> mesh)
    {
        const auto triplets = detail::assemble_linear_lagrangian_stiffness_triplets(mesh);

        const auto interior_nodes = mesh.compute_interior_vertices();
        const auto & boundary_nodes = mesh.boundary_vertices();

        // Stiffness
        Eigen::SparseMatrix<double> full_stiffness_mat(mesh.num_vertices(), mesh.num_vertices());
        full_stiffness_mat.setFromTriplets(triplets.stiffness_triplets.cbegin(), triplets.stiffness_triplets.cend());
        Eigen::SparseMatrix<double> interior_stiffness_mat = sparse_submatrix(full_stiffness_mat,
                                                                              interior_nodes,
                                                                              interior_nodes);
        Eigen::SparseMatrix<double> boundary_stiffness_mat = sparse_submatrix(full_stiffness_mat,
                                                                              interior_nodes,
                                                                              boundary_nodes);

        // Mass
        Eigen::SparseMatrix<double> full_mass_mat(mesh.num_vertices(), mesh.num_vertices());
        full_mass_mat.setFromTriplets(triplets.mass_triplets.cbegin(), triplets.mass_triplets.cend());
        Eigen::SparseMatrix<double> interior_mass_mat = sparse_submatrix(full_mass_mat,
                                                                         interior_nodes,
                                                                         interior_nodes);
        Eigen::SparseMatrix<double> boundary_mass_mat = sparse_submatrix(full_mass_mat,
                                                                         interior_nodes,
                                                                         boundary_nodes);

        return LagrangeBasis2d(std::move(mesh),
                               std::move(interior_mass_mat),
                               std::move(boundary_mass_mat),
                               std::move(interior_stiffness_mat),
                               std::move(boundary_stiffness_mat)
        );
    }

    inline LagrangeBasis2d::LagrangeBasis2d(IndexedMesh<double, int> mesh,
                                            Eigen::SparseMatrix<double> mass_interior_matrix,
                                            Eigen::SparseMatrix<double> mass_boundary_matrix,
                                            Eigen::SparseMatrix<double> stiffness_interior_matrix,
                                            Eigen::SparseMatrix<double> stiffness_boundary_matrix)
            : _mesh(std::move(mesh)),
              _mass_matrix(std::move(mass_interior_matrix)),
              _boundary_mass_matrix(std::move(mass_boundary_matrix)),
              _stiffness_matrix(std::move(stiffness_interior_matrix)),
              _boundary_stiffness_matrix(std::move(stiffness_boundary_matrix))
    {

    }


    template <int Strength, typename Function2d>
    Eigen::VectorXd LagrangeBasis2d::compute_load(const Function2d & f) const
    {
        Eigen::VectorXd load(_mesh.num_vertices());
        load.setZero();

        // See triquad.hpp for the mapping used here
        const auto a_basis = [] (auto x, auto  ) { return 0.5 * x + 0.5; };
        const auto b_basis = [] (auto  , auto y) { return 0.5 * y + 0.5; };
        const auto c_basis = [] (auto x, auto y) { return 0.5 * (-x - y); };

        for (const auto element : _mesh.elements())
        {
            const auto z0 = element.vertex_indices[0];
            const auto z1 = element.vertex_indices[1];
            const auto z2 = element.vertex_indices[2];

            const auto & a = _mesh.vertices()[z0];
            const auto & b = _mesh.vertices()[z1];
            const auto & c = _mesh.vertices()[z2];
            const auto transform = triquad_transform(a, b, c);
            const auto transformed_f = [&f, &transform] (auto x, auto y)
            {
                const auto coords = transform.transform_from_reference(x, y);
                return f(coords.x, coords.y);
            };

            const auto absdet = transform.absolute_determinant();

            load(z0) += absdet * triquad_ref<Strength, double>(
                    [&] (auto x, auto y) { return transformed_f(x, y) * a_basis(x, y); }
            );
            load(z1) += absdet * triquad_ref<Strength, double>(
                    [&] (auto x, auto y) { return transformed_f(x, y) * b_basis(x, y); }
            );
            load(z2) += absdet * triquad_ref<Strength, double>(
                    [&] (auto x, auto y) { return transformed_f(x, y) * c_basis(x, y); }
            );
        }

        return submatrix(load, _mesh.compute_interior_vertices(), { 0 });
    }
}