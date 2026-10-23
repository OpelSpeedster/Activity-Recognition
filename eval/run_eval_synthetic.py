"""
Check the FallDetector / FightDetector rules with hand-made keypoint examples.

This does not require YOLO, a webcam, or any video file. It checks the rule
logic only; it does not measure camera, pose-model, or real-world accuracy.
Use labeled real videos for that.

Usage:
    python -m eval.run_eval_synthetic
"""
from detectors.fall_detector import FallDetector
from detectors.fight_detector import FightDetector
from eval.metrics import Event, evaluate_events, summarize, print_report, accumulate_events
from eval.synthetic_scenarios import ALL_SCENARIOS


def run_scenario(name, scenario_fn):
    frames, frame_shape, ground_truth = scenario_fn()

    fall_detector = FallDetector()
    fight_detector = FightDetector()

    predictions = []
    open_events = {}

    for frame_id, poses in enumerate(frames, start=1):
        falls, _inactivity = fall_detector.update(poses)
        fights = fight_detector.update(poses, frame_shape)
        accumulate_events(predictions, open_events, "fall", falls, frame_id)
        accumulate_events(predictions, open_events, "fight", fights, frame_id)

    last_frame = len(frames)
    for (etype, pid), start in open_events.items():
        predictions.append(Event(etype, start, last_frame, pid))

    return ground_truth, predictions


def main():
    all_results = []
    for name, scenario_fn in ALL_SCENARIOS.items():
        ground_truth, predictions = run_scenario(name, scenario_fn)
        results = [
            evaluate_events(ground_truth, predictions, "fall"),
            evaluate_events(ground_truth, predictions, "fight"),
        ]
        overall = summarize(results)
        print_report(name, results, overall)
        all_results.extend(results)

        # Explain what happened in plain terms for scenarios with no ground truth
        if not ground_truth:
            fp_total = sum(r.false_positives for r in results)
            verdict = "PASS (no false alarms)" if fp_total == 0 else f"FAIL ({fp_total} false alarm(s))"
            print(f"  -> {verdict}")

    print("\n=== Aggregate across all synthetic scenarios ===")
    fall_results = [r for r in all_results if r.event_type == "fall"]
    fight_results = [r for r in all_results if r.event_type == "fight"]
    print_report("ALL SCENARIOS", [summarize(fall_results), summarize(fight_results)], summarize(all_results))


if __name__ == "__main__":
    main()
