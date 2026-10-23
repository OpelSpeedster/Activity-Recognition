"""
Hand-made keypoint sequences for checking detector logic without a camera,
video file, or YOLO inference. They are useful for learning and for checking
that a code change did not break these examples. They are not real-world
accuracy measurements.

Each scenario returns:
    frames: list of poses-per-frame, in the exact format the detectors expect:
        [ {"id": 1, "bbox": [x1,y1,x2,y2], "keypoints": [(x,y,conf), ...17]} , ... ]
    frame_shape: (height, width, channels)
    ground_truth: list of eval.metrics.Event

Keypoint layout follows COCO-17 (same as YOLOv8-pose):
0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists,
11-12 hips, 13-14 knees, 15-16 ankles.
"""
import random

from eval.metrics import Event

FRAME_SHAPE = (720, 1280, 3)  # height, width, channels


def _standing_pose(cx, cy, jitter=0.0, rng=None):
    """A roughly upright person centered at (cx, cy). Torso is vertical."""
    r = rng or random
    j = lambda: r.uniform(-jitter, jitter)
    kp = [None] * 17
    kp[0] = (cx + j(), cy - 90 + j(), 0.9)      # nose
    kp[5] = (cx - 20 + j(), cy - 60 + j(), 0.9)  # L shoulder
    kp[6] = (cx + 20 + j(), cy - 60 + j(), 0.9)  # R shoulder
    kp[11] = (cx - 15 + j(), cy + j(), 0.9)      # L hip
    kp[12] = (cx + 15 + j(), cy + j(), 0.9)      # R hip
    kp[13] = (cx - 15 + j(), cy + 60 + j(), 0.85)  # L knee
    kp[14] = (cx + 15 + j(), cy + 60 + j(), 0.85)  # R knee
    kp[15] = (cx - 15 + j(), cy + 120 + j(), 0.8)  # L ankle
    kp[16] = (cx + 15 + j(), cy + 120 + j(), 0.8)  # R ankle
    for i in (1, 2, 3, 4, 7, 8, 9, 10):
        kp[i] = (cx + j(), cy - 70 + j(), 0.6)
    bbox = [int(cx - 30), int(cy - 100), int(cx + 30), int(cy + 130)]
    return kp, bbox


def _lying_pose(cx, cy, jitter=0.0, rng=None):
    """A horizontal person (torso roughly horizontal, wide flat bbox)."""
    r = rng or random
    j = lambda: r.uniform(-jitter, jitter)
    kp = [None] * 17
    kp[0] = (cx - 90 + j(), cy + j(), 0.85)
    kp[5] = (cx - 60 + j(), cy - 10 + j(), 0.85)
    kp[6] = (cx - 60 + j(), cy + 10 + j(), 0.85)
    kp[11] = (cx + j(), cy - 8 + j(), 0.85)
    kp[12] = (cx + j(), cy + 8 + j(), 0.85)
    kp[13] = (cx + 45 + j(), cy - 8 + j(), 0.8)
    kp[14] = (cx + 45 + j(), cy + 8 + j(), 0.8)
    kp[15] = (cx + 90 + j(), cy - 8 + j(), 0.75)
    kp[16] = (cx + 90 + j(), cy + 8 + j(), 0.75)
    for i in (1, 2, 3, 4, 7, 8, 9, 10):
        kp[i] = (cx - 70 + j(), cy + j(), 0.55)
    bbox = [int(cx - 100), int(cy - 30), int(cx + 100), int(cy + 30)]
    return kp, bbox


def fall_scenario(seed=0, n_frames=90, fall_start=30, fall_settle=40):
    """
    Person 1 stands still (frames 1-29), collapses to horizontal
    (frames 30-39, transitional), then lies still (frames 40-89).
    Ground truth: a fall event covering roughly [30, 90].
    """
    rng = random.Random(seed)
    frames = []
    cx, cy = 400, 300

    for f in range(1, n_frames + 1):
        if f < fall_start:
            kp, bbox = _standing_pose(cx, cy, jitter=1.5, rng=rng)
        elif f < fall_settle:
            # linear interpolation from standing to lying (transition = real motion)
            t = (f - fall_start) / (fall_settle - fall_start)
            skp, sbbox = _standing_pose(cx, cy, jitter=0, rng=rng)
            lkp, lbbox = _lying_pose(cx, cy, jitter=0, rng=rng)
            kp = [
                (skp[i][0] + t * (lkp[i][0] - skp[i][0]),
                 skp[i][1] + t * (lkp[i][1] - skp[i][1]),
                 0.85)
                for i in range(17)
            ]
            bbox = [int(sbbox[i] + t * (lbbox[i] - sbbox[i])) for i in range(4)]
        else:
            kp, bbox = _lying_pose(cx, cy, jitter=0.5, rng=rng)  # lying still, tiny jitter

        frames.append([{"id": 1, "bbox": bbox, "keypoints": kp}])

    ground_truth = [Event("fall", fall_start, n_frames, person_id=1)]
    return frames, FRAME_SHAPE, ground_truth


def normal_walking_scenario(seed=1, n_frames=90):
    """Person walks across frame, staying upright the whole time. No events."""
    rng = random.Random(seed)
    frames = []
    for f in range(1, n_frames + 1):
        cx = 200 + f * 8
        cy = 300 + 10 * (1 if f % 10 < 5 else -1)  # slight bob from walking
        kp, bbox = _standing_pose(cx, cy, jitter=3.0, rng=rng)
        frames.append([{"id": 1, "bbox": bbox, "keypoints": kp}])
    return frames, FRAME_SHAPE, []  # no ground-truth events


def sitting_scenario(seed=2, n_frames=90):
    """
    Person sits down and stays seated. This is a KNOWN HARD CASE: bbox
    aspect ratio widens somewhat and torso can tilt, but it should NOT be
    classified as a fall. Used to test false-positive rate on ambiguous
    postures.
    """
    rng = random.Random(seed)
    frames = []
    cx, cy = 500, 350
    for f in range(1, n_frames + 1):
        kp, bbox = _standing_pose(cx, cy, jitter=1.0, rng=rng)
        if f > 15:
            # sitting: torso tilts slightly forward (~30-40 deg, below fall threshold),
            # legs fold under (knees rise toward hips), bbox compresses vertically
            kp = list(kp)
            kp[13] = (kp[13][0], kp[13][1] - 40, kp[13][2])  # knees rise
            kp[14] = (kp[14][0], kp[14][1] - 40, kp[14][2])
            kp[11] = (kp[11][0] + 8, kp[11][1] + 5, kp[11][2])  # slight hip shift
            kp[12] = (kp[12][0] + 8, kp[12][1] + 5, kp[12][2])
            bbox = [bbox[0], bbox[1] + 20, bbox[2], bbox[3] - 10]
        frames.append([{"id": 1, "bbox": bbox, "keypoints": kp}])
    return frames, FRAME_SHAPE, []  # no ground-truth fall


def fight_scenario(seed=3, n_frames=90, fight_start=30, fight_end=70):
    """
    Two people stand apart (no fight), move close together and exchange
    rapid, large motion (fight_start..fight_end), then separate.
    Ground truth: a fight event covering [fight_start, fight_end] for both IDs.
    """
    rng = random.Random(seed)
    frames = []
    for f in range(1, n_frames + 1):
        if f < fight_start:
            cx1, cy1 = 300, 300
            cx2, cy2 = 700, 300  # far apart, no interaction
            j1 = j2 = 1.0
        elif f <= fight_end:
            # close together, large rapid relative motion (jitter simulates
            # fast limb movement / grappling)
            cx1, cy1 = 480 + rng.uniform(-15, 15), 300 + rng.uniform(-15, 15)
            cx2, cy2 = 520 + rng.uniform(-15, 15), 300 + rng.uniform(-15, 15)
            j1 = j2 = 25.0
        else:
            cx1, cy1 = 300, 300
            cx2, cy2 = 700, 300
            j1 = j2 = 1.0

        kp1, bbox1 = _standing_pose(cx1, cy1, jitter=j1, rng=rng)
        kp2, bbox2 = _standing_pose(cx2, cy2, jitter=j2, rng=rng)
        frames.append([
            {"id": 1, "bbox": bbox1, "keypoints": kp1},
            {"id": 2, "bbox": bbox2, "keypoints": kp2},
        ])

    ground_truth = [
        Event("fight", fight_start, fight_end, person_id=1),
        Event("fight", fight_start, fight_end, person_id=2),
    ]
    return frames, FRAME_SHAPE, ground_truth


def two_people_passing_scenario(seed=4, n_frames=90):
    """
    Two people walk past each other closely (near-miss in a corridor) but
    with NORMAL walking motion, not fight-level motion. Should NOT trigger
    a fight. Tests false-positive rate for "close proximity != aggression".
    """
    rng = random.Random(seed)
    frames = []
    for f in range(1, n_frames + 1):
        cx1 = 200 + f * 6
        cx2 = 900 - f * 6
        kp1, bbox1 = _standing_pose(cx1, 300, jitter=2.0, rng=rng)
        kp2, bbox2 = _standing_pose(cx2, 300, jitter=2.0, rng=rng)
        frames.append([
            {"id": 1, "bbox": bbox1, "keypoints": kp1},
            {"id": 2, "bbox": bbox2, "keypoints": kp2},
        ])
    return frames, FRAME_SHAPE, []


ALL_SCENARIOS = {
    "fall": fall_scenario,
    "normal_walking": normal_walking_scenario,
    "sitting": sitting_scenario,
    "fight": fight_scenario,
    "two_people_passing": two_people_passing_scenario,
}
