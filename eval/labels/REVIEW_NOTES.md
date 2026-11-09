# Test video label review

The 15 CSVs named after videos in `test_video/` use the event intervals below. Unlisted frames are negative for the three scored classes: `fall`, `fight`, and `crowd_density`. The evaluator checks deterministic detector outputs; it does not score Qwen's final event classifications, Telegram delivery, inactivity, or `uncertain` decisions.

Crowd-density labels mark sustained, visibly tightly packed crowds. They are visual labels, not calibrated people-per-square-meter measurements. The current detector uses person count normalized by image area, so its thresholds should be calibrated for each camera and resolution before comparing density scores.

## Positive annotations

- `istockphoto-2149294792-640_adpp_is.csv`: staged slip/fall, frames 27–286 (about 0.9–9.5 s at 29.97 fps).
- `gettyimages-704-76-640_adpp.csv`: continuous brawl, frames 1–188 (full 7.84 s clip); two rows represent the two participants.
- `Video-864.csv`: elevator fight, frames 76–436 (about 2.5–14.5 s at 30 fps); two rows represent the two participants.
- `12269404_2320_1080_30fps.csv`: dense indoor event crowd, frames 1–336.
- `14565204_2160_3840_60fps.csv`: crowded pedestrian market walkway, frames 1–802.
- `1677252-uhd_3840_2160_30fps.csv`: very dense event crowd, frames 1–1136.
- `17037852_2160_3840_30fps.csv`: concert audience, frames 1–336.
- `19980218-hd_1080_1920_30fps.csv`: packed stadium audience, frames 1–330.
- `gettyimages-2277438678-640_adpp.csv`: dense rally/protest crowd, frames 1–4040. Crowd-control activity is not labeled as a fall or fight.

I screened one-second samples across each clip and checked the candidate incident clips at half-second intervals. Frame windows are approximate and should be refined with frame-by-frame playback before using the scores as a formal benchmark.
