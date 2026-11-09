# How to create ground-truth labels for evaluation

To measure real accuracy (not just "it ran"), you need to tell the system
what *actually* happened in a test video, so predictions can be checked
against ground truth. This is normal in ML — it's called **annotation** or
**labeling**.

## Format

Create one CSV per test video: `eval/labels/<video_name>.csv`

```csv
event_type,start_frame,end_frame,person_id
fall,120,180,1
fight,300,340,2
fight,300,340,3
```

- `event_type`: `fall`, `fight`, or `crowd_density` (these are the event types
  the current real-video evaluator scores). A crowd-density interval means a
  sustained, visibly tightly packed crowd; ordinary groups of pedestrians do
  not count. The detector uses normalized person count, so its thresholds need
  calibration for each camera and resolution.
- `start_frame` / `end_frame`: the frame range where the event is actually
  happening (open the video in VLC or any player with frame-step, note the
  frame numbers — most players show timestamp; multiply by fps to get frame
  number, or use `ffprobe`/`opencv` frame counter)
- `person_id`: optional; see the scoring policy below.

## Scoring policy

The evaluator matches events of the same type when their labeled and predicted
frame ranges overlap, allowing a default boundary tolerance of 10 frames.
Each prediction and each label can match at most once; unmatched predictions
count as false positives and unmatched labels count as false negatives.
Repeated detector hits for the same person are grouped into one episode, and
the episode closes after more than two frames without another hit. Use
--tolerance-frames N to change the timing allowance for a video.

When both a label and prediction include a person ID, their IDs must match.
Leave the label ID blank to score whether the event was detected regardless of
which person was involved.

## How to make one quickly (10-15 min per clip)

1. Record or download a short (30-60s) clip with a staged fall, a staged
   push/shove, and some normal walking — mix all three so the eval isn't
   trivial.
2. Step through frame-by-frame (in VLC: `E` key steps one frame at a time)
   and note down the frame number where the fall/fight visibly starts and
   ends.
3. Fill the CSV as shown above.
4. Also add "negative" clips (no incidents). They are important for
   measuring how often the system creates a false alert for a reviewer.

## Where to get free clips if you have none

- Record yourself/friends staging a fall onto a mat and a play-fight —
  this is useful for a controlled class demonstration.
- Public fall-detection research datasets exist (e.g. "Le2i Fall Detection
  Dataset", "UR Fall Detection Dataset") — search for them if you want a
  larger benchmark. A labeled test set gives scores for those clips, but a
  small student dataset cannot prove performance in other camera views.

## Why frame ranges, not single frames

A fall is not an instant — it's an event with a start and end. Detectors
naturally take a few frames to confirm (that's the whole point of
`min_frames` debouncing to avoid false positives). Matching a prediction
against a *window* rather than one exact frame is standard for event
detection tasks (same idea used in speech activity detection, object
tracking evaluation, etc.).
