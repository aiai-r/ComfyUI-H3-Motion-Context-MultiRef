from h3_timing import (
    crossfade_plan,
    is_exact_av_boundary,
    is_h3_video_run,
    largest_h3_video_run,
    preferred_av_runs_through,
    video_runs_through,
)


def test_video_grid_and_floor():
    assert video_runs_through(100) == [5, 22, 39, 56, 73, 90]
    assert largest_h3_video_run(4) == 0
    assert largest_h3_video_run(5) == 5
    assert largest_h3_video_run(21) == 5
    assert largest_h3_video_run(22) == 22
    assert largest_h3_video_run(40) == 39
    assert largest_h3_video_run(96) == 90
    assert is_h3_video_run(39)
    assert not is_h3_video_run(40)


def test_exact_av_boundaries():
    assert preferred_av_runs_through(243) == [39, 90, 141, 192, 243]
    for n in [39, 90, 141, 192, 243]:
        assert is_exact_av_boundary(n)
    for n in [5, 22, 56, 73, 107, 124]:
        assert not is_exact_av_boundary(n)


def test_crossfade_plan_keeps_only_last_matching_context():
    assert crossfade_plan(90, 39) == (51, 39)
    assert crossfade_plan(39, 90) == (0, 39)
    assert crossfade_plan(40, 39) == (1, 39)
    assert crossfade_plan(39, 0) == (39, 0)
