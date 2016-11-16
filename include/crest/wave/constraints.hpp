#pragma once

#include <crest/util/eigen_extensions.hpp>
#include <crest/util/algorithms.hpp>

#include <cassert>

namespace crest
{
    namespace wave
    {
        template <typename Scalar>
        class ConstraintHandler
        {
        public:
            virtual ~ConstraintHandler() {}

            virtual int num_free_nodes() const = 0;

            virtual VectorX<Scalar> expand_solution(const VectorX<Scalar> & reduced) const = 0;

            virtual VectorX<Scalar> constrain_solution(const VectorX<Scalar> & full) const = 0;
            virtual VectorX<Scalar> constrain_velocity(const VectorX<Scalar> & full) const = 0;
            virtual VectorX<Scalar> constrain_acceleration(const VectorX<Scalar> & full) const = 0;

            virtual VectorX<Scalar> constrain_load(const VectorX<Scalar> & full) const = 0;

            virtual Eigen::SparseMatrix<Scalar> constrain_system_matrix(const Eigen::SparseMatrix<Scalar> & matrix) const = 0;
        };

        template <typename Scalar>
        class HomogeneousDirichlet : public ConstraintHandler<Scalar>
        {
        public:
            template <typename BasisImpl>
            explicit HomogeneousDirichlet(const Basis<Scalar, BasisImpl> & basis)
                    : _interior(basis.interior_nodes()), _num_nodes(basis.num_dof())
            {

            }

            virtual int num_free_nodes() const override
            {
                return static_cast<int>(_interior.size());
            }

            virtual VectorX<Scalar> expand_solution(const VectorX<Scalar> & reduced) const override
            {
                const auto num_free = num_free_nodes();
                VectorX<Scalar> full(_num_nodes);
                full.setZero();

                assert(_num_nodes >= num_free);
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

            virtual VectorX<Scalar> constrain_load(const VectorX<Scalar> & full) const override
            {
                return submatrix(full, _interior, { 0 });
            }

            virtual Eigen::SparseMatrix<Scalar>
            constrain_system_matrix(const Eigen::SparseMatrix<Scalar> & matrix) const override
            {
                return sparse_submatrix(matrix, _interior, _interior);
            }

        private:
            std::vector<int> _interior;
            int _num_nodes;
        };
    }

}
