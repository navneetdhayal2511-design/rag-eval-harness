import pytest

from rageval.metrics import calibrate, cohens_kappa
from rageval.metrics.stats import bootstrap_ci, compare, paired_permutation_test


def test_bootstrap_interval_brackets_the_mean() -> None:
    values = [0.0, 1.0] * 30
    interval = bootstrap_ci(values, seed=0)
    assert interval.value == pytest.approx(0.5)
    assert interval.low < interval.value < interval.high
    assert interval.n == 60


def test_bootstrap_interval_collapses_when_every_score_is_identical() -> None:
    interval = bootstrap_ci([0.8] * 20, seed=0)
    assert (interval.low, interval.value, interval.high) == pytest.approx((0.8, 0.8, 0.8))


def test_bootstrap_is_reproducible_for_a_given_seed() -> None:
    values = [i / 40 for i in range(40)]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


def test_a_wider_interval_comes_from_noisier_scores() -> None:
    tight = bootstrap_ci([0.5] * 10 + [0.6] * 10, seed=0)
    loose = bootstrap_ci([0.0] * 10 + [1.0] * 10, seed=0)
    assert (loose.high - loose.low) > (tight.high - tight.low)


def test_identical_runs_are_not_significant() -> None:
    scores = [0.0, 1.0, 1.0, 0.0, 1.0] * 6
    assert paired_permutation_test(scores, scores) == 1.0


def test_a_consistent_improvement_is_detected() -> None:
    baseline = [0.0] * 40
    candidate = [1.0] * 40
    assert paired_permutation_test(baseline, candidate, seed=0) < 0.01


def test_a_single_lucky_question_is_not_significant() -> None:
    # The reason the gate requires significance: one question flipping on a
    # 40-question set must not fail somebody's build.
    baseline = [1.0] * 39 + [0.0]
    candidate = [1.0] * 40
    assert paired_permutation_test(baseline, candidate, seed=0) > 0.05


def test_p_value_is_never_exactly_zero() -> None:
    # A randomisation test cannot justify p = 0, and reporting it invites
    # claims the data does not support.
    assert paired_permutation_test([0.0] * 50, [1.0] * 50, seed=0) > 0.0


def test_comparison_reports_direction_and_pairing() -> None:
    result = compare("recall@5", [0.0] * 30, [1.0] * 30, seed=0)
    assert result.delta == pytest.approx(1.0)
    assert result.significant and result.direction == "improved"
    assert result.n == 30


def test_comparison_requires_matching_question_sets() -> None:
    with pytest.raises(ValueError, match="same questions"):
        compare("recall@5", [0.0, 1.0], [1.0])


def test_kappa_is_one_for_perfect_agreement() -> None:
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_kappa_is_near_zero_for_chance_agreement() -> None:
    # Both label half positive but never agree on which half.
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_kappa_matches_a_hand_computed_confusion_matrix() -> None:
    judge = [1] * 7 + [0] * 3
    human = [1] * 5 + [0] * 5
    observed = 0.8
    expected = 0.7 * 0.5 + 0.3 * 0.5
    assert cohens_kappa(judge, human) == pytest.approx((observed - expected) / (1 - expected))


def test_calibration_separates_false_positives_from_false_negatives() -> None:
    report = calibrate([1, 1, 0, 0], [1, 0, 1, 0])
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.agreement == pytest.approx(0.5)


def test_a_judge_at_chance_is_reported_as_unusable() -> None:
    report = calibrate([1, 1, 0, 0], [1, 0, 1, 0])
    assert "not usable" in report.verdict


def test_a_judge_that_agrees_is_reported_as_usable() -> None:
    report = calibrate([1] * 9 + [0] * 11, [1] * 10 + [0] * 10)
    assert report.kappa > 0.6 and report.verdict.startswith("usable")
