#pragma once

#include <cstdint>
#include <cmath>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/util/eigen_extensions.hpp>
#include <crest/quadrature/triquad.hpp>
#include <crest/basis/basis.hpp>

namespace crest
{
    /**
     * A standard linear Lagrangian basis
     */
    template <typename Scalar>
    class LagrangeBasis2d : public Basis<Scalar, LagrangeBasis2d<Scalar>>
    {
    public:
        explicit LagrangeBasis2d(const IndexedMesh<Scalar, int> & mesh) : _mesh(mesh) {}

        virtual std::vector<int> boundary_nodes() const override { return _mesh.boundary_vertices(); }
        virtual std::vector<int> interior_nodes() const override { return _mesh.compute_interior_vertices(); }

        virtual Assembly<Scalar> assemble() const override;

        virtual int num_dof() const override { return _mesh.num_vertices(); }

        template <typename Function2d>
        VectorX<Scalar> interpolate(const Function2d &f) const;

        template <int QuadStrength, typename Function2d>
        VectorX<Scalar> load(const Function2d &f) const;

        template <int QuadStrength, typename Function2d>
        Scalar error_l2(const Function2d &f, const VectorX<Scalar> & weights) const;

        template <int QuadStrength, typename Function2d_x, typename Function2d_y>
        Scalar error_h1_semi(const Function2d_x & f_x,
                             const Function2d_y & f_y,
                             const VectorX<Scalar> & weights) const;

    private:
        const IndexedMesh<double, int> & _mesh;
    };

    namespace detail
    {
        template <typename Scalar>
        struct assembly_triplets {
            std::vector<Eigen::Triplet<Scalar>> stiffness_triplets;
            std::vector<Eigen::Triplet<Scalar>> mass_triplets;
        };

        template <typename Scalar>
        assembly_triplets<Scalar> assemble_linear_lagrangian_system_triplets(
                const crest::IndexedMesh<Scalar, int> & mesh);
    }

    /*
     * IMPLEMENTATION BELOW
     */

    template <typename Scalar>
    detail::assembly_triplets<Scalar> detail::assemble_linear_lagrangian_system_triplets(
            const crest::IndexedMesh<Scalar, int> & mesh)
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

    template <typename Scalar>
    Assembly<Scalar> LagrangeBasis2d<Scalar>::assemble() const
    {
        const auto triplets = detail::assemble_linear_lagrangian_system_triplets(_mesh);

        Assembly<Scalar> assembly;

        // Stiffness
        assembly.stiffness = Eigen::SparseMatrix<Scalar>(num_dof(), num_dof());
        assembly.stiffness.setFromTriplets(triplets.stiffness_triplets.cbegin(), triplets.stiffness_triplets.cend());

        // Mass
        assembly.mass = Eigen::SparseMatrix<Scalar>(num_dof(), num_dof());
        assembly.mass.setFromTriplets(triplets.mass_triplets.cbegin(), triplets.mass_triplets.cend());

        return assembly;
    }

    template <typename Scalar>
    template <int QuadStrength, typename Function2d>
    VectorX<Scalar> LagrangeBasis2d<Scalar>::load(const Function2d & f) const
    {
        Eigen::VectorXd load(_mesh.num_vertices());
        load.setZero();

        // See triquad.hpp for the mapping used here
        const auto a_basis = [] (auto x, auto  ) { return Scalar(0.5) * x + Scalar(0.5); };
        const auto b_basis = [] (auto  , auto y) { return Scalar(0.5) * y + Scalar(0.5); };
        const auto c_basis = [] (auto x, auto y) { return Scalar(0.5) * (-x - y); };

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

            load(z0) += absdet * triquad_ref<QuadStrength, Scalar>(
                    [&] (auto x, auto y) { return transformed_f(x, y) * a_basis(x, y); }
            );
            load(z1) += absdet * triquad_ref<QuadStrength, Scalar>(
                    [&] (auto x, auto y) { return transformed_f(x, y) * b_basis(x, y); }
            );
            load(z2) += absdet * triquad_ref<QuadStrength, Scalar>(
                    [&] (auto x, auto y) { return transformed_f(x, y) * c_basis(x, y); }
            );
        }

        return load;
    }

    template <typename Scalar>
    template <typename Function2d>
    VectorX<Scalar> LagrangeBasis2d<Scalar>::interpolate(const Function2d & f) const
    {
        // Simple nodal interpolation
        auto result = VectorX<Scalar>(_mesh.num_vertices());
        for (int i = 0; i < _mesh.num_vertices(); ++i)
        {
            const auto vertex = _mesh.vertices()[i];
            result(i) = f(vertex.x, vertex.y);
        }
        return result;
    }

    template <typename Scalar>
    template <int QuadStrength, typename Function2d>
    Scalar LagrangeBasis2d<Scalar>::error_l2(const Function2d &f, const VectorX<Scalar> & weights) const
    {
        Scalar error_squared = Scalar(0);

        // See triquad.hpp for the mapping used here
        const auto basis0 = [] (auto x, auto  ) { return Scalar(0.5) * x + Scalar(0.5); };
        const auto basis1 = [] (auto  , auto y) { return Scalar(0.5) * y + Scalar(0.5); };
        const auto basis2 = [] (auto x, auto y) { return Scalar(0.5) * (-x - y); };

#pragma omp parallel for reduction(+:error_squared)
        for (int element_index = 0; element_index < _mesh.num_elements(); ++element_index)
        {
            const auto element = _mesh.elements()[element_index];
            const auto z0 = element.vertex_indices[0];
            const auto z1 = element.vertex_indices[1];
            const auto z2 = element.vertex_indices[2];

            const auto w0 = weights(z0);
            const auto w1 = weights(z1);
            const auto w2 = weights(z2);

            const auto & a = _mesh.vertices()[z0];
            const auto & b = _mesh.vertices()[z1];
            const auto & c = _mesh.vertices()[z2];
            const auto transform = triquad_transform(a, b, c);
            const auto f_ref = [&f, &transform] (auto x, auto y)
            {
                const auto coords = transform.transform_from_reference(x, y);
                return f(coords.x, coords.y);
            };

            // Computes the square of the difference of f and f_h in the reference triangle
            const auto diff_ref_squared = [&] (auto x, auto y)
            {
                const auto f_h_ref = w0 * basis0(x, y) +
                                     w1 * basis1(x, y) +
                                     w2 * basis2(x, y);

                const auto diff = f_ref(x, y) - f_h_ref;
                return diff * diff;
            };

            const auto absdet = transform.absolute_determinant();
            error_squared += absdet * triquad_ref<QuadStrength, Scalar>(diff_ref_squared);
        }

        return std::sqrt(error_squared);
    };


    template <typename Scalar>
    template <int QuadStrength, typename Function2d_x, typename Function2d_y>
    Scalar LagrangeBasis2d<Scalar>::error_h1_semi(const Function2d_x & f_x,
                                                  const Function2d_y & f_y,
                                                  const VectorX<Scalar> & weights) const
    {
        Scalar error_squared = Scalar(0);

        // See triquad.hpp for the mapping used here
        const auto basis0_x = Scalar(0.5);
        const auto basis0_y = Scalar(0.0);
        const auto basis1_x = Scalar(0.0);
        const auto basis1_y = Scalar(0.5);
        const auto basis2_x = Scalar(-0.5);
        const auto basis2_y = Scalar(-0.5);

#pragma omp parallel for reduction(+:error_squared)
        for (int element_index = 0; element_index < _mesh.num_elements(); ++element_index)
        {
            const auto element = _mesh.elements()[element_index];
            const auto z0 = element.vertex_indices[0];
            const auto z1 = element.vertex_indices[1];
            const auto z2 = element.vertex_indices[2];

            const auto w0 = weights(z0);
            const auto w1 = weights(z1);
            const auto w2 = weights(z2);

            const auto & a = _mesh.vertices()[z0];
            const auto & b = _mesh.vertices()[z1];
            const auto & c = _mesh.vertices()[z2];
            const auto transform = triquad_transform(a, b, c);

            const auto f_grad_ref = [&f_x, &f_y, &transform] (auto x, auto y)
            {
                const auto coords = transform.transform_from_reference(x, y);
                Eigen::Matrix<Scalar, 2, 1> grad;
                grad(0) = f_x(coords.x, coords.y);
                grad(1) = f_y(coords.x, coords.y);
                return grad;
            };

            // Since we have linear elements, the gradients are constants
            Eigen::Matrix<Scalar, 2, 1> f_h_grad_ref;
            f_h_grad_ref(0) = w0 * basis0_x +
                              w1 * basis1_x +
                              w2 * basis2_x;
            f_h_grad_ref(1) = w0 * basis0_y +
                              w1 * basis1_y +
                              w2 * basis2_y;

            // Due to change of variables, we have to left-apply J^-T
            const auto J_inv_t = transform.jacobian().inverse().transpose();
            const Eigen::Matrix<Scalar, 2, 1> f_h_grad_ref_transformed = J_inv_t * f_h_grad_ref;

            // Computes the square of the difference of grad(f) and grad(f_h)
            // in the reference triangle
            const auto diff_squared = [&] (auto x, auto y)
            {
                // Note that J_inv_t cancels with J_t for f_grad_ref
                const Eigen::Matrix<Scalar, 2, 1> diff = f_grad_ref(x, y) - f_h_grad_ref_transformed;
                return diff.dot(diff);
            };

            const auto absdet = transform.absolute_determinant();
            error_squared += absdet * triquad_ref<QuadStrength, Scalar>(diff_squared);
        }

        return std::sqrt(error_squared);
    };
}
