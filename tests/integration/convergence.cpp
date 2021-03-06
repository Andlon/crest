#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <experiments/experiments.hpp>

#include <tuple>
#include <memory>

using ::testing::DoubleEq;
using ::testing::DoubleNear;

typedef std::shared_ptr<crest::wave::Integrator<double>> IntegratorSharedPtr;

/*
 * The order of convergence tests in this file simply test that the errors are close to some values
 * that have been verified manually when applying the Method of Manufactured Solutions.
 */

class homogeneous_unit_square : public ::testing::TestWithParam<std::tuple<IntegratorSharedPtr, ExperimentResult>> {

};

class inhomogeneous_unit_square : public ::testing::TestWithParam<std::tuple<IntegratorSharedPtr, ExperimentResult>> {

};

constexpr double TOL = 1e-9;

TEST_P(homogeneous_unit_square, order_of_convergence)
{
    auto test_params = GetParam();
    auto integrator = std::get<0>(test_params);
    integrator->set_tolerance(TOL);
    const auto expected_result = std::get<1>(test_params);
    auto experiment = HomogeneousDirichletUnitSquare();
    experiment.run_offline(expected_result.offline_parameters);
    const auto result = experiment.run_online(expected_result.online_parameters.get(), *integrator);

    const auto expected_online_result = expected_result.online_result.get();
    ASSERT_THAT(result.error_summary.l2, DoubleNear(expected_online_result.error_summary.l2, 1e-9));
    ASSERT_THAT(result.error_summary.h1_semi, DoubleNear(expected_online_result.error_summary.h1_semi, 1e-7));
    ASSERT_THAT(result.error_summary.h1, DoubleNear(expected_online_result.error_summary.h1, 1e-7));
}

TEST_P(inhomogeneous_unit_square, order_of_convergence)
{
    auto test_params = GetParam();
    auto integrator = std::get<0>(test_params);
    integrator->set_tolerance(TOL);
    const auto expected_result = std::get<1>(test_params);

    auto experiment = InhomogeneousDirichletUnitSquare();
    experiment.run_offline(expected_result.offline_parameters);
    const auto result = experiment.run_online(expected_result.online_parameters.get(), *integrator);

    const auto expected_online_result = expected_result.online_result.get();
    ASSERT_THAT(result.error_summary.l2, DoubleNear(expected_online_result.error_summary.l2, 1e-9));
    ASSERT_THAT(result.error_summary.h1_semi, DoubleNear(expected_online_result.error_summary.h1_semi, 1e-7));
    ASSERT_THAT(result.error_summary.h1, DoubleNear(expected_online_result.error_summary.h1, 1e-7));
}

ExperimentResult make_convergence_test_result(const OfflineParameters & offline,
                                              const OnlineParameters & online,
                                              const ErrorSummary & error_summary)
{
    return ExperimentResult()
            .with_offline_parameters(offline)
            .with_online_parameters(online)
            .with_online_result(OnlineResult().with_error_summary(error_summary));
}

INSTANTIATE_TEST_CASE_P(direct_crank_nicolson,
                        homogeneous_unit_square,
                        ::testing::Combine(
                                ::testing::Values(std::make_shared<crest::wave::DirectCrankNicolson<double>>()),
                                ::testing::Values(
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.25),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.585458541734819)
                                                        .with_h1_semi(0.584669994677925)
                                                        .with_l2(0.0289167538286227)
                                        ),
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.125),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.289671336214043)
                                                        .with_h1_semi(0.2895515729457)
                                                        .with_l2(0.00761102062470874)
                                        ),
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.0625),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.144437769866845)
                                                        .with_h1_semi(0.144415884917059)
                                                        .with_l2(0.00220311548992367)
                                        )
                                )));

INSTANTIATE_TEST_CASE_P(iterative_crank_nicolson,
                        homogeneous_unit_square,
                        ::testing::Combine(
                                ::testing::Values(std::make_shared<crest::wave::IterativeCrankNicolson<double>>()),
                                ::testing::Values(
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.25),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.585458541735016)
                                                        .with_h1_semi(0.584669994678112)
                                                        .with_l2(0.0289167538288395)
                                        ),
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.125),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.289671336213969)
                                                        .with_h1_semi(0.289551572945625)
                                                        .with_l2(0.00761102062490839)
                                        ),
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.0625),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.144437769865326)
                                                        .with_h1_semi(0.14441588491555)
                                                        .with_l2(0.002203115490062)
                                        )
                                )));

INSTANTIATE_TEST_CASE_P(direct_crank_nicolson,
                        inhomogeneous_unit_square,
                        ::testing::Combine(
                                ::testing::Values(std::make_shared<crest::wave::DirectCrankNicolson<double>>()),
                                ::testing::Values(
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.25),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.587379118139992)
                                                        .with_h1_semi(0.586591995031311)
                                                        .with_l2(0.0289494785208848)
                                        ),
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.125),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.290637174136373)
                                                        .with_h1_semi(0.290517800933425)
                                                        .with_l2(0.00761895620278423)
                                        ),
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.0625),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.144922764238598)
                                                        .with_h1_semi(0.144901003208638)
                                                        .with_l2(0.0022048870589444)
                                        )
                                )));

INSTANTIATE_TEST_CASE_P(iterative_crank_nicolson,
                        inhomogeneous_unit_square,
                        ::testing::Combine(
                                ::testing::Values(std::make_shared<crest::wave::IterativeCrankNicolson<double>>()),
                                ::testing::Values(
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.25),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.587379371767793)
                                                        .with_h1_semi(0.586592249024436)
                                                        .with_l2(0.0289494776458804)
                                        ),
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.125),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.290636831113929)
                                                        .with_h1_semi(0.290517458731212)
                                                        .with_l2(0.00761894063454651)
                                        ),
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.0625),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.144922345898392)
                                                        .with_h1_semi(0.144900585264829)
                                                        .with_l2(0.00220483987973394)
                                        )
                                )));

INSTANTIATE_TEST_CASE_P(iterative_leapfrog,
                        inhomogeneous_unit_square,
                        ::testing::Combine(
                                ::testing::Values(std::make_shared<crest::wave::IterativeLeapfrog<double>>()),
                                ::testing::Values(
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.25),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.587378020587716)
                                                        .with_h1_semi(0.586590792266803)
                                                        .with_l2(0.0289506641927545)
                                        ),
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.125),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.290625227318885)
                                                        .with_h1_semi(0.290505802110512)
                                                        .with_l2(0.00761996738873146)
                                        ),
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.0625),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.144948283932812)
                                                        .with_h1_semi(0.144926498965492)
                                                        .with_l2(0.00220611626654336)
                                        )
                                )));

INSTANTIATE_TEST_CASE_P(lumped_leapfrog,
                        inhomogeneous_unit_square,
                        ::testing::Combine(
                                ::testing::Values(std::make_shared<crest::wave::LumpedLeapfrog<double>>()),
                                ::testing::Values(
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.25),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.582666537765749)
                                                        .with_h1_semi(0.582159646452925)
                                                        .with_l2(0.0240803754481128)
                                        ),
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.125),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.288489719319695)
                                                        .with_h1_semi(0.288431743027219)
                                                        .with_l2(0.0057424077081183)
                                        ),
                                        make_convergence_test_result(
                                                OfflineParameters()
                                                        .with_mesh_resolution(0.0625),
                                                OnlineParameters()
                                                        .with_end_time(0.5)
                                                        .with_sample_count(401),
                                                ErrorSummary()
                                                        .with_h1(0.144206934687547)
                                                        .with_h1_semi(0.144197839593727)
                                                        .with_l2(0.00159275434745992)
                                        )
                                )));
