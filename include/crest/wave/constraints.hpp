#pragma once

#include <crest/util/eigen_extensions.hpp>
#include <crest/util/algorithms.hpp>

#include <crest/basis/basis.hpp>

#include <crest/wave/load.hpp>

#include <cassert>

namespace crest
{
    namespace wave
    {
        template <typename Scalar>
        class ConstrainedSystem {
        public:
            virtual ~ConstrainedSystem() {}

            int num_free_nodes() const { return _assembly.stiffness.rows(); }
            int num_total_nodes() const { return _num_total; }

            const Assembly<Scalar> & assembly() const { return _assembly; }

            virtual VectorX<Scalar> expand_solution(Scalar t, const VectorX<Scalar> & constrained) const = 0;
            virtual VectorX<Scalar> constrain_solution(const VectorX<Scalar> & full) const = 0;
            virtual VectorX<Scalar> constrain_velocity(const VectorX<Scalar> & full) const = 0;
            virtual VectorX<Scalar> constrain_acceleration(const VectorX<Scalar> & full) const = 0;

            virtual VectorX<Scalar> load(Scalar t) const = 0;

        protected:
            explicit ConstrainedSystem(Assembly<Scalar> assembly,
                                       int num_total_nodes)
                    : _assembly(std::move(assembly)), _num_total(num_total_nodes)
            { }


        private:
            Assembly<Scalar> _assembly;
            int _num_total;
        };

        template <typename Scalar>
        class HomogeneousDirichletBC : public ConstrainedSystem<Scalar>
        {
        public:
            virtual VectorX<Scalar> expand_solution(Scalar t, const VectorX<Scalar> & reduced) const override
            {
                (void) t;

                const auto num_free = ConstrainedSystem<Scalar>::num_free_nodes();
                const auto num_total = ConstrainedSystem<Scalar>::num_total_nodes();
                VectorX<Scalar> full(num_total);
                full.setZero();

                assert(num_total >= num_free);
                for (int i = 0; i < num_free; ++i)
                {
                    const auto full_index = _interior[i];
                    full(full_index) = reduced(i);
                }

                return full;
            }

            virtual VectorX<Scalar> constrain_solution(const VectorX<Scalar> & full) const override
            {
                return submatrix(full, _interior, { 0 });
            }
            virtual VectorX<Scalar> constrain_velocity(const VectorX<Scalar> & full) const override
            {
                return submatrix(full, _interior, { 0 });
            }
            virtual VectorX<Scalar> constrain_acceleration(const VectorX<Scalar> & full) const override
            {
                return submatrix(full, _interior, { 0 });
            }

            virtual VectorX<Scalar> load(Scalar t) const override
            {
                const auto unconstrained_load = _load_provider.compute(t);
                return submatrix(unconstrained_load, _interior, { 0 });
            }

            template <typename BasisImpl>
            static HomogeneousDirichletBC<Scalar> assemble(const Basis<Scalar, BasisImpl> & basis,
                                                           const LoadProvider<Scalar> & load_provider,
                                                           const Assembly<Scalar> & assembly)
            {
                const auto interior = basis.interior_nodes();
                const auto num_total = basis.num_dof();

                Assembly<Scalar> constrained_assembly;
                constrained_assembly.stiffness = sparse_submatrix(assembly.stiffness, interior, interior);
                constrained_assembly.mass = sparse_submatrix(assembly.mass, interior, interior);

                return HomogeneousDirichletBC(std::move(constrained_assembly),
                                              load_provider,
                                              std::move(interior),
                                              num_total);
            }

        private:
            explicit HomogeneousDirichletBC(Assembly<Scalar> assembly,
                                            const LoadProvider<Scalar> & load_provider,
                                            std::vector<int> interior,
                                            int num_total_nodes)
                    : ConstrainedSystem<Scalar>(std::move(assembly), num_total_nodes),
                      _load_provider(load_provider),
                      _interior(interior)
            { }

            const LoadProvider<Scalar> & _load_provider;
            std::vector<int> _interior;
        };

        // Note: this only works for Lagrange-type bases with nodal interpolation
        template <typename Scalar, typename BasisImpl, typename Function2d, typename Function2d_tt>
        class InhomogeneousDirichletBC : public ConstrainedSystem<Scalar>
        {
        public:
            virtual VectorX<Scalar> expand_solution(Scalar t, const VectorX<Scalar> & reduced) const override
            {
                const auto num_free = ConstrainedSystem<Scalar>::num_free_nodes();
                const auto num_total = ConstrainedSystem<Scalar>::num_total_nodes();
                const auto num_boundary = static_cast<int>(_boundary.size());
                VectorX<Scalar> full(num_total);
                full.setZero();

                assert(num_total >= num_free);
                for (int i = 0; i < num_free; ++i)
                {
                    const auto full_index = _interior[i];
                    full(full_index) = reduced(i);
                }

                const auto g_h = [this, t] (auto x, auto y) { return _boundary_func(t, x, y); };
                const auto g_h_weights = _basis.interpolate_boundary(g_h);

                for (int i = 0; i < num_boundary; ++i)
                {
                    const auto full_index = _boundary[i];
                    full(full_index) = g_h_weights(i);
                }

                return full;
            }

            virtual VectorX<Scalar> constrain_solution(const VectorX<Scalar> & full) const override
            {
                return submatrix(full, _interior, { 0 });
            }
            virtual VectorX<Scalar> constrain_velocity(const VectorX<Scalar> & full) const override
            {
                return submatrix(full, _interior, { 0 });
            }
            virtual VectorX<Scalar> constrain_acceleration(const VectorX<Scalar> & full) const override
            {
                return submatrix(full, _interior, { 0 });
            }

            virtual VectorX<Scalar> load(Scalar t) const override
            {
                const auto g_h = [this, t] (auto x, auto y) { return _boundary_func(t, x, y); };
                const auto g_h_tt = [this, t] (auto x, auto y) { return _boundary_func_tt(t, x, y); };

                const auto g_h_boundary_weights = _basis.interpolate_boundary(g_h);
                const auto g_h_tt_boundary_weights = _basis.interpolate_boundary(g_h_tt);

                const auto unconstrained_load = _load_provider.compute(t);
                const auto constrained_load = submatrix(unconstrained_load, _interior, { 0 });

                const auto & A_b = _boundary_assembly.stiffness;
                const auto & M_b = _boundary_assembly.mass;
                const auto & b = constrained_load;
                const auto & beta = g_h_boundary_weights;
                const auto & beta_tt = g_h_tt_boundary_weights;

                return b - M_b * beta_tt - A_b * beta;
            }

            static InhomogeneousDirichletBC<Scalar, BasisImpl, Function2d, Function2d_tt>
            assemble(const Basis<Scalar, BasisImpl> & basis,
                     const LoadProvider<Scalar> & load_provider,
                     const Assembly<Scalar> & assembly,
                     const Function2d & boundary_func,
                     const Function2d_tt & boundary_func_second_derivative)
            {
                const auto interior = basis.interior_nodes();
                const auto boundary = basis.boundary_nodes();

                Assembly<Scalar> constrained_assembly;
                constrained_assembly.stiffness = sparse_submatrix(assembly.stiffness, interior, interior);
                constrained_assembly.mass = sparse_submatrix(assembly.mass, interior, interior);

                Assembly<Scalar> boundary_assembly;
                boundary_assembly.stiffness = sparse_submatrix(assembly.stiffness, interior, boundary);
                boundary_assembly.mass = sparse_submatrix(assembly.mass, interior, boundary);

                return InhomogeneousDirichletBC(std::move(constrained_assembly),
                                                std::move(boundary_assembly),
                                                basis,
                                                load_provider,
                                                std::move(interior),
                                                std::move(boundary),
                                                boundary_func,
                                                boundary_func_second_derivative);
            }

        private:
            explicit InhomogeneousDirichletBC(Assembly<Scalar> assembly,
                                              Assembly<Scalar> boundary_assembly,
                                              const Basis<Scalar, BasisImpl> & basis,
                                              const LoadProvider<Scalar> & load_provider,
                                              std::vector<int> interior,
                                              std::vector<int> boundary,
                                              const Function2d & boundary_function,
                                              const Function2d_tt & boundary_func_tt)
                    : ConstrainedSystem<Scalar>(std::move(assembly), static_cast<int>(interior.size() + boundary.size())),
                      _basis(basis),
                      _load_provider(load_provider),
                      _interior(interior),
                      _boundary(boundary),
                      _boundary_assembly(std::move(boundary_assembly)),
                      _boundary_func(boundary_function),
                      _boundary_func_tt(boundary_func_tt)
            { }

            const Basis<Scalar, BasisImpl> & _basis;
            const LoadProvider<Scalar> & _load_provider;
            std::vector<int> _interior;
            std::vector<int> _boundary;

            Assembly<Scalar> _boundary_assembly;
            const Function2d & _boundary_func;
            const Function2d_tt & _boundary_func_tt;
        };

        template <typename Scalar, typename BasisImpl, typename Function2d, typename Function2d_tt>
        InhomogeneousDirichletBC<Scalar, BasisImpl, Function2d, Function2d_tt>
        make_inhomogeneous_dirichlet(const Basis<Scalar, BasisImpl> & basis,
                                     const LoadProvider<Scalar> & load_provider,
                                     const Assembly<Scalar> & assembly,
                                     const Function2d & boundary_func,
                                     const Function2d_tt & boundary_func_second_derivative)
        {
            return InhomogeneousDirichletBC<Scalar, BasisImpl, Function2d, Function2d_tt>::assemble(
                    basis, load_provider, assembly, boundary_func, boundary_func_second_derivative);
        };
    }
}
