from h3_timing import (
    crossfade_plan,
    is_exact_av_boundary,
    is_h3_video_run,
    largest_h3_video_run,
    preferred_av_runs_through,
    video_runs_through,
    sample_boundary_from_frames,
    sample_boundary_from_seconds,
    sample_span_for_frame_interval,
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


def test_absolute_pcm_boundaries_avoid_extension_seam_rounding_drift():
    sr = 44100
    # Default 5-second H3 run: 124 raw frames, 39 protected, 85 new frames.
    # Extension 1 seam is frame 124; Extension 2 seam is frame 209.
    fixed_relative_cut = round(39 / 24 * sr)
    cut1 = sample_span_for_frame_interval(124 - 39, 39, sr)
    cut2 = sample_span_for_frame_interval(209 - 39, 39, sr)
    assert fixed_relative_cut == 71662
    assert cut1 == 71662
    assert cut2 == 71663


def test_absolute_song_slice_endpoints_use_master_timeline_boundaries():
    sr = 32000
    start = 323 / 24
    duration = 362 / 24
    start_sample = sample_boundary_from_seconds(start, sr)
    end_sample = sample_boundary_from_seconds(start + duration, sr)
    assert end_sample - start_sample == 482666
    # Relative duration rounding would be one sample longer here.
    assert round(duration * sr) == 482667
