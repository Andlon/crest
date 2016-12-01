#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <experiments/experiments.hpp>

using ::testing::DoubleEq;
using ::testing::DoubleNear;


class homogeneous_unit_square : public ::testing::TestWithParam<ExperimentResult> {

};

class inhomogeneous_unit_square : public ::testing::TestWithParam<ExperimentResult> {

};

TEST_P(homogeneous_unit_square, crank_nicolson_order_of_convergence)
{
    const auto expected_result = GetParam();
    crest::wave::CrankNicolson<double> integrator;
    const auto result = HomogeneousDirichletUnitSquare().run(expected_result.parameters, integrator);

    ASSERT_THAT(result.error_summary.l2, DoubleNear(expected_result.error_summary.l2, 1e-9));
    ASSERT_THAT(result.error_summary.h1_semi, DoubleNear(expected_result.error_summary.h1_semi, 1e-7));
    ASSERT_THAT(result.error_summary.h1, DoubleNear(expected_result.error_summary.h1, 1e-7));
}

TEST_P(inhomogeneous_unit_square, crank_nicolson_order_of_convergence)
{
    const auto expected_result = GetParam();
    crest::wave::CrankNicolson<double> integrator;
    const auto result = InhomogeneousDirichletUnitSquare().run(expected_result.parameters, integrator);

    ASSERT_THAT(result.error_summary.l2, DoubleNear(expected_result.error_summary.l2, 1e-9));
    ASSERT_THAT(result.error_summary.h1_semi, DoubleNear(expected_result.error_summary.h1_semi, 1e-7));
    ASSERT_THAT(result.error_summary.h1, DoubleNear(expected_result.error_summary.h1, 1e-7));
}

INSTANTIATE_TEST_CASE_P(homogeneous_unit_square,
                        homogeneous_unit_square,
                        ::testing::Values(
                                ExperimentResult()
                                        .with_parameters(ExperimentParameters()
                                                                 .with_end_time(0.5)
                                                                 .with_mesh_resolution(0.100)
                                                                 .with_sample_count(5001))
                                        .with_error_summary(ErrorSummary()
                                                                    .with_h1(0.289032060471293)
                                                                    .with_h1_semi(0.288922983708485)
                                                                    .with_l2(0.00728286774846731)),

                                ExperimentResult()
                                        .with_parameters(ExperimentParameters()
                                                                 .with_end_time(0.5)
                                                                 .with_mesh_resolution(0.0750)
                                                                 .with_sample_count(5001))
                                        .with_error_summary(ErrorSummary()
                                                                    .with_h1(0.167206590828422)
                                                                    .with_h1_semi(0.167184054654119)
                                                                    .with_l2(0.00263556985714594)),

                                ExperimentResult()
                                        .with_parameters(ExperimentParameters()
                                                                 .with_end_time(0.5)
                                                                 .with_mesh_resolution(0.0625)
                                                                 .with_sample_count(5001))
                                        .with_error_summary(ErrorSummary()
                                                                    .with_h1(0.143922838396057)
                                                                    .with_h1_semi(0.143908016134641)
                                                                    .with_l2(0.00184830572926679))
                        ));

INSTANTIATE_TEST_CASE_P(inhomogeneous_unit_square,
                        inhomogeneous_unit_square,
                        ::testing::Values(
                                ExperimentResult()
                                        .with_parameters(ExperimentParameters()
                                                                 .with_end_time(0.5)
                                                                 .with_mesh_resolution(0.100)
                                                                 .with_sample_count(5001))
                                        .with_error_summary(ErrorSummary()
                                                                    .with_h1(0.290002304690516)
                                                                    .with_h1_semi(0.289893587881787)
                                                                    .with_l2(0.00729105764949143)),

                                ExperimentResult()
                                        .with_parameters(ExperimentParameters()
                                                                 .with_end_time(0.5)
                                                                 .with_mesh_resolution(0.0750)
                                                                 .with_sample_count(5001))
                                        .with_error_summary(ErrorSummary()
                                                                    .with_h1(0.168536181026854)
                                                                    .with_h1_semi(0.168513724358073)
                                                                    .with_l2(0.00264946059116889)),

                                ExperimentResult()
                                        .with_parameters(ExperimentParameters()
                                                                 .with_end_time(0.5)
                                                                 .with_mesh_resolution(0.0625)
                                                                 .with_sample_count(5001))
                                        .with_error_summary(ErrorSummary()
                                                                    .with_h1(0.144412511159522)
                                                                    .with_h1_semi(0.144397779258504)
                                                                    .with_l2(0.00185031417795632))
                        ));
