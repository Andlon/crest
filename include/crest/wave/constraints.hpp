#pragma once

#include <crest/util/eigen_extensions.hpp>
#include <crest/util/algorithms.hpp>

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

            int num_free_nodes() const { return _stiffness.rows(); }
            int num_total_nodes() const { return _num_total; }

            const Eigen::SparseMatrix<Scalar> & stiffness() const { return _stiffness; }
            const Eigen::SparseMatrix<Scalar> & mass() const { return _mass; }

            virtual VectorX<Scalar> expand_solution(const VectorX<Scalar> & constrained) const = 0;
            virtual VectorX<Scalar> constrain_solution(const VectorX<Scalar> & full) const = 0;
            virtual VectorX<Scalar> constrain_velocity(const VectorX<Scalar> & full) const = 0;
            virtual VectorX<Scalar> constrain_acceleration(const VectorX<Scalar> & full) const = 0;

            virtual VectorX<Scalar> load(Scalar t) const = 0;

        protected:
            explicit ConstrainedSystem(Eigen::SparseMatrix<Scalar> stiffness,
                                       Eigen::SparseMatrix<Scalar> mass,
                                       int num_total_nodes)
                    : _stiffness(stiffness), _mass(mass), _num_total(num_total_nodes)
            { }


        private:
            Eigen::SparseMatrix<Scalar> _stiffness;
            Eigen::SparseMatrix<Scalar> _mass;
            int _num_total;
        };

        template <typename Scalar>
        class HomogeneousDirichletBC : public ConstrainedSystem<Scalar>
        {
        public:
            virtual VectorX<Scalar> expand_solution(const VectorX<Scalar> & reduced) const override
            {
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
                                                           const LoadProvider<Scalar> & load_provider)
            {
                const auto interior = basis.interior_nodes();
                const auto assembly = basis.assemble();
                const auto num_total = basis.num_dof();

                const auto stiffness = sparse_submatrix(assembly.stiffness, interior, interior);
                const auto mass = sparse_submatrix(assembly.mass, interior, interior);

                return HomogeneousDirichletBC(std::move(stiffness),
                                              std::move(mass),
                                              load_provider,
                                              std::move(interior),
                                              num_total);
            }

        private:
            explicit HomogeneousDirichletBC(Eigen::SparseMatrix<Scalar> stiffness,
                                            Eigen::SparseMatrix<Scalar> mass,
                                            const LoadProvider<Scalar> & load_provider,
                                            std::vector<int> interior,
                                            int num_total_nodes)
                    : ConstrainedSystem<Scalar>(stiffness, mass, num_total_nodes),
                      _load_provider(load_provider),
                      _interior(interior)
            { }

            const LoadProvider<Scalar> & _load_provider;
            std::vector<int> _interior;
        };
    }

}
