#pragma once

#include <crest/geometry/indexed_mesh.hpp>
#include <crest/geometry/mesh_algorithms.hpp>
#include <crest/geometry/biscale_mesh.hpp>
#include <crest/basis/quasi_interpolation.hpp>
#include <crest/basis/lagrange_basis2d.hpp>
#include <crest/basis/detail/homogenized_basis_detail.hpp>
#include <crest/util/eigen_extensions.hpp>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <set>
#include <cassert>

namespace crest
{
    template <typename Scalar>
    class HomogenizedBasis : public Basis<Scalar, HomogenizedBasis<Scalar>>
    {
    public:
        explicit HomogenizedBasis(const BiscaleMesh<Scalar, int> & mesh,
                                  Eigen::SparseMatrix<double> weights);

        virtual std::vector<int> boundary_nodes() const override { return _mesh.coarse_mesh().boundary_vertices(); }

        virtual std::vector<int> interior_nodes() const override { return _mesh.coarse_mesh().compute_interior_vertices(); }

        virtual Assembly<Scalar> assemble() const override;

        virtual int num_dof() const override { return _mesh.coarse_mesh().num_vertices(); }

        template <typename Function2d>
        VectorX<Scalar> interpolate(const Function2d & f) const;

        template <int QuadStrength, typename Function2d>
        VectorX<Scalar> load(const Function2d & f) const;

        template <int QuadStrength, typename Function2d>
        Scalar error_l2(const Function2d & f, const VectorX<Scalar> & weights) const;

        template <int QuadStrength, typename Function2d_x, typename Function2d_y>
        Scalar error_h1_semi(const Function2d_x & f_x,
                             const Function2d_y & f_y,
                             const VectorX<Scalar> & weights) const;

        const Eigen::SparseMatrix<Scalar> & basis_weights() const { return _basis_weights; }

    private:
        Eigen::SparseMatrix<Scalar> _basis_weights;
        const BiscaleMesh<Scalar, int> & _mesh;
    };

    /**
     * Base class for solvers that compute localized correctors for a given BiscaleMesh.
     */
    template <typename Scalar>
    class CorrectorSolver
    {
    public:
        virtual ~CorrectorSolver() {}

        crest::HomogenizedBasis<Scalar> compute_basis(const BiscaleMesh<Scalar, int> & mesh,
                                                      unsigned int oversampling) const;

        Eigen::SparseMatrix<Scalar> compute_correctors(const BiscaleMesh<Scalar, int> & mesh,
                                                       unsigned int oversampling) const;

        virtual std::vector<Eigen::Triplet<Scalar>> compute_element_correctors_for_patch(
                const BiscaleMesh<Scalar, int> & mesh,
                const std::vector<int> & fine_patch_interior,
                const Eigen::SparseMatrix<Scalar> & local_coarse_stiffness,
                const Eigen::SparseMatrix<Scalar> & local_fine_stiffness,
                const Eigen::SparseMatrix<Scalar> & local_quasi_interpolator,
                int coarse_element) const = 0;

    protected:
        VectorX<Scalar> local_rhs(const BiscaleMesh<Scalar, int> & mesh,
                                  const std::vector<int> & fine_patch_interior,
                                  int coarse_element,
                                  int local_index) const;
    };

    /**
     * The default corrector solver, which uses SparseLU to compute correctors. It is slow, but very robust.
     */
    template <typename Scalar>
    class SparseLuCorrectorSolver : public CorrectorSolver<Scalar>
    {
    public:
        virtual std::vector<Eigen::Triplet<Scalar>> compute_element_correctors_for_patch(
                const BiscaleMesh<Scalar, int> & mesh,
                const std::vector<int> & fine_patch_interior,
                const Eigen::SparseMatrix<Scalar> & local_coarse_stiffness,
                const Eigen::SparseMatrix<Scalar> & local_fine_stiffness,
                const Eigen::SparseMatrix<Scalar> & local_quasi_interpolator,
                int coarse_element) const override
        {
            (void) local_coarse_stiffness;

            std::vector<Eigen::Triplet<Scalar>> triplets;
            const auto & I_H = local_quasi_interpolator;
            const auto & A = local_fine_stiffness;
            const auto C = detail::construct_saddle_point_problem(A, I_H);

            Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
            solver.analyzePattern(C);
            solver.factorize(C);
            assert(solver.info() == Eigen::Success);

            for (int i = 0; i < 3; ++i)
            {
                const auto b_local = this->local_rhs(mesh, fine_patch_interior, coarse_element, i);

                VectorX<Scalar> c(C.rows());
                c << b_local, VectorX<Scalar>::Zero(I_H.rows());

                // Recall that the solution is of the form [x, kappa], where kappa is merely a Lagrange multiplier, so
                // we extract x as the corrector.
                const VectorX<Scalar> corrector = solver.solve(c).topRows(A.rows());

                assert(static_cast<size_t>(corrector.rows()) == fine_patch_interior.size());
                const auto global_index = mesh.coarse_mesh().elements()[coarse_element].vertex_indices[i];
                for (size_t k = 0; k < fine_patch_interior.size(); ++k)
                {
                    const auto component = corrector(k);
                    // Due to rounding issues, some components that should perhaps be zero in exact arithmetic
                    // may be non-zero, and as such we might end up with a denser matrix than we should actually
                    // have. To prevent this, we introduce a threshold which determines whether to keep the entry.
                    // TODO: Make this threshold configurable?
                    if (std::abs(component) > 1e-12)
                    {
                        triplets.emplace_back(Eigen::Triplet<Scalar>(global_index, fine_patch_interior[k], component));
                    }
                }
            }

            return triplets;
        }
    };

    template <typename Scalar>
    VectorX<Scalar> CorrectorSolver<Scalar>::local_rhs(const BiscaleMesh<Scalar, int> & mesh,
                                                       const std::vector<int> & fine_patch_interior,
                                                       int coarse_element,
                                                       int local_index) const
    {
        assert(local_index >= 0 && local_index < 3);
        // TODO: Simplify this function
        assert(std::is_sorted(fine_patch_interior.cbegin(), fine_patch_interior.cend()));

        const auto coarse_triangle = mesh.coarse_mesh().triangle_for(coarse_element);
        const Eigen::Matrix<Scalar, 3, 3> coarse_coeff = detail::basis_coefficients_for_triangle(coarse_triangle);

        const auto coarse_grad_x = coarse_coeff(0, local_index);
        const auto coarse_grad_y = coarse_coeff(1, local_index);

        VectorX<Scalar> rhs(fine_patch_interior.size());
        rhs.setZero();
        for (auto k : mesh.descendants_for(coarse_element))
        {
            const auto vertex_indices = mesh.fine_mesh().elements()[k].vertex_indices;
            const auto fine_triangle = mesh.fine_mesh().triangle_for(k);
            const Eigen::Matrix<Scalar, 3, 3> fine_coeff = detail::basis_coefficients_for_triangle(fine_triangle);

            for (size_t j = 0; j < 3; ++j)
            {
                const auto fine_grad_x = fine_coeff(0, j);
                const auto fine_grad_y = fine_coeff(1, j);

                // TODO: Fix this
                const auto product = [&] (auto  , auto  )
                {
                    return coarse_grad_x * fine_grad_x + coarse_grad_y * fine_grad_y;
                };

                // At this point, we only know the index of the vertex in the global mesh,
                // but we need the index of the vertex with respect to the patch interior. For now,
                // we just perform a binary search to recover it, though there may be much more efficient ways.
                // For example, we can construct a hashmap in the beginning of this function (which may
                // or may not be more efficient).
                const auto vertex_index = vertex_indices[j];
                const auto range = std::equal_range(fine_patch_interior.cbegin(),
                                                    fine_patch_interior.cend(),
                                                    vertex_index);

                if (range.first != range.second)
                {
                    const auto inner_product = triquad<2>(product,
                                                          fine_triangle.a,
                                                          fine_triangle.b,
                                                          fine_triangle.c);

                    const auto local_index = range.first - fine_patch_interior.cbegin();
                    rhs(local_index) += inner_product;
                }

            }
        }
        return rhs;
    }

    template <typename Scalar>
    Eigen::SparseMatrix<Scalar> CorrectorSolver<Scalar>::compute_correctors(
            const BiscaleMesh<Scalar, int> & mesh,
            unsigned int oversampling) const
    {
        const auto I_H = quasi_interpolator(mesh);

        Eigen::SparseMatrix<Scalar> A_fine(mesh.fine_mesh().num_vertices(), mesh.fine_mesh().num_vertices());
        Eigen::SparseMatrix<Scalar> A_coarse(mesh.coarse_mesh().num_elements(), mesh.coarse_mesh().num_elements());
        {
            // Currently we needlessly construct the mass matrix here too. For now we put this in a block scope
            // so that it will be deallocated shortly thereafter, but in the long-term this should
            // be remedied so that we don't redundantly compute it. TODO
            const auto fine_assembly = LagrangeBasis2d<Scalar>(mesh.fine_mesh()).assemble();
            A_fine = std::move(fine_assembly.stiffness);

            const auto coarse_assembly = LagrangeBasis2d<Scalar>(mesh.coarse_mesh()).assemble();
            A_coarse = std::move(coarse_assembly.stiffness);
        }

        std::vector<Eigen::Triplet<Scalar>> basis_triplets;
        for (int coarse_element = 0; coarse_element < mesh.coarse_mesh().num_elements(); ++coarse_element)
        {
            const auto coarse_patch = mesh.coarse_element_patch(coarse_element, oversampling);
            const auto coarse_patch_interior = coarse_patch.interior();
            const auto fine_patch = mesh.fine_patch_from_coarse(coarse_patch);
            const auto fine_patch_interior = fine_patch.interior();

            if (!fine_patch_interior.empty())
            {
                const auto I_H_local = detail::localized_quasi_interpolator(I_H,
                                                                            coarse_patch,
                                                                            fine_patch_interior);
                const auto A_fine_local = sparse_submatrix(A_fine, fine_patch_interior, fine_patch_interior);
                const auto A_coarse_local = sparse_submatrix(A_coarse, coarse_patch_interior, coarse_patch_interior);

                const auto corrector_contributions = compute_element_correctors_for_patch(
                        mesh, fine_patch.interior(), A_coarse_local, A_fine_local, I_H_local, coarse_element);
                std::copy(corrector_contributions.cbegin(),
                          corrector_contributions.cend(),
                          std::back_inserter(basis_triplets));
            }
        }

        Eigen::SparseMatrix<Scalar> basis(mesh.coarse_mesh().num_vertices(), mesh.fine_mesh().num_vertices());
        basis.setFromTriplets(basis_triplets.cbegin(), basis_triplets.cend());
        return basis;
    }

    template <typename Scalar>
    HomogenizedBasis<Scalar> CorrectorSolver<Scalar>::compute_basis(const BiscaleMesh<Scalar, int> & mesh,
                                                                    unsigned int oversampling) const
    {
        const auto corrector_weights = compute_correctors(mesh, oversampling);
        const auto lagrange_basis_weights = detail::standard_coarse_basis_in_fine_space(mesh);
        const auto basis_weights = lagrange_basis_weights - corrector_weights;
        return HomogenizedBasis<Scalar>(mesh, basis_weights);
    }

    template <typename Scalar>
    HomogenizedBasis<Scalar>::HomogenizedBasis(const BiscaleMesh<Scalar, int> & mesh,
                                               Eigen::SparseMatrix<double> weights)
            : _mesh(mesh)
    {
        if (weights.rows() == mesh.coarse_mesh().num_vertices() && weights.cols() == mesh.fine_mesh().num_vertices())
        {
            _basis_weights = std::move(weights);
        } else
        {
            throw std::invalid_argument("Dimensions of basis weights are not compatible with "
                                                "supplied coarse and fine meshes.");
        }
    }


    template <typename Scalar>
    Assembly<Scalar> HomogenizedBasis<Scalar>::assemble() const
    {
        LagrangeBasis2d<Scalar> fine_basis(_mesh.fine_mesh());
        const auto fine_assembly = fine_basis.assemble();

        const auto & M = fine_assembly.mass;
        const auto & A = fine_assembly.stiffness;
        const auto & W = _basis_weights;

        Assembly<Scalar> assembly;
        assembly.mass = W * M * W.transpose();
        assembly.stiffness = W * A * W.transpose();
        return assembly;
    }

    template <typename Scalar>
    template <typename Function2d>
    VectorX<Scalar> HomogenizedBasis<Scalar>::interpolate(const Function2d & f) const
    {
        // A slightly more accuracte way of interpolating would perhaps be to interpolate the function
        // in the fine space, and then quasi-interpolate it to weights in the coarse space.
        // However, it seems this has some unfortunate effects on the implementation of inhomogeneous
        // Dirichlet boundary conditions?
        return LagrangeBasis2d<Scalar>(_mesh.coarse_mesh()).interpolate(f);
    }

    template <typename Scalar>
    template <int QuadStrength, typename Function2d>
    VectorX<Scalar> HomogenizedBasis<Scalar>::load(const Function2d & f) const
    {
        const LagrangeBasis2d<Scalar> fine_basis(_mesh.fine_mesh());
        const auto fine_load = fine_basis.load<QuadStrength>(f);
        const auto & W = _basis_weights;
        return W * fine_load;
    };

    template <typename Scalar>
    template <int QuadStrength, typename Function2d>
    Scalar HomogenizedBasis<Scalar>::error_l2(const Function2d & f, const VectorX<Scalar> & weights) const
    {
        const LagrangeBasis2d<Scalar> fine_basis(_mesh.fine_mesh());
        const auto & W = _basis_weights;
        const VectorX<Scalar> fine_weights = W.transpose() * weights;
        return fine_basis.error_l2<QuadStrength>(f, fine_weights);
    };

    template <typename Scalar>
    template <int QuadStrength, typename Function2d_x, typename Function2d_y>
    Scalar HomogenizedBasis<Scalar>::error_h1_semi(const Function2d_x & f_x,
                                                   const Function2d_y & f_y,
                                                   const VectorX<Scalar> & weights) const
    {
        const LagrangeBasis2d<Scalar> fine_basis(_mesh.fine_mesh());
        const auto & W = _basis_weights;
        const VectorX<Scalar> fine_weights = W.transpose() * weights;
        return fine_basis.error_h1_semi<QuadStrength>(f_x, f_y, fine_weights);
    };
}
