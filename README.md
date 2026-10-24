# Crowd Safety Learning Project

This is a student prototype for learning how video detection works. It looks
for a few movement patterns and creates an alert for a person to review. It
does **not** identify people, decide that someone committed a crime, predict
intent, or contact emergency services.

## What works in this version

- Detect people and estimate body keypoints with a small pose model.
- Flag a **possible fall** from body posture and movement.
- Flag **fight-like movement** when people are close and moving quickly. This
  is only a rough candidate signal; dancing, play, and occlusion can confuse it.
- Estimate people count and a relative crowd level for one camera view.
- Save event details and optional images/clips so a person can review them.

Weapons, fire/smoke, theft, disasters, and contacting emergency services are
not implemented. Each needs its own carefully tested detector and data. A
camera cannot establish a person's intent. Any real response must be decided
by a trained person using approved procedures.

## How the program works

```text
webcam or video
      -> pose model finds people and body points
      -> fall and fight-like movement rules check short sequences
      -> crowd counter estimates people in the frame
      -> rule agent chooses: no alert, alert, or human review
      -> program logs the result and may save a clip
```

The rule agent is the best place to start: it is local, predictable, and does
not need an API key. The optional LLM agent receives a short text summary of
the detected events, not the video. Local rules choose the alert; the LLM can
only add a short review note.

## Start here

Use a short video first; it is easier to repeat than a live webcam. From
PowerShell in this folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py --source path\to\your_video.mp4 --agent rule
```

For an NVIDIA Desktop, install the CUDA-enabled PyTorch build into the active
environment before `pip install -r requirements.txt`. Use the official
[PyTorch installer selector](https://pytorch.org/get-started/locally/) and
choose Windows, Pip, Python, and a CUDA option. CPU-only computers and the Pi
can skip this and install `requirements.txt` directly. OpenCV is included in
the requirements and supplies the video-reading support; a separate decoder
install is normally unnecessary.

To use a USB webcam on a computer:

```powershell
python main.py --source webcam --camera 0 --agent rule
```

On a CUDA computer, leave `--device` out to let the program select CUDA
automatically. The default CUDA profile uses `yolov8m-pose.pt` at 960 pixels,
confidence 0.10, and a higher NMS IoU threshold to retain more overlapping
people. CPU-only computers use `yolov8n-pose.pt` at 640 pixels. The first run
downloads the selected model weights. Override these settings with `--model`,
`--imgsz`, `--device`, or the matching values in `.env`.

To force the CPU profile on a computer, add `--device cpu`. Do not add that
option when you want the CUDA profile.

For the supplied 2320×1080 dense crowd clip, the larger input preserves more
detail than the previous 640-pixel nano pose profile. It may improve the count,
but a pose detector still misses people who are very small or hidden behind
others. A dedicated density-map crowd counter is a separate future module and
should be measured against hand-labeled frames from the target camera.

Press `q` in the video window to stop. Output is saved in `logs/`, `frames/`,
and `clips/` beside the project. Only use footage you have permission to
process, and delete test clips you no longer need.

## Raspberry Pi 4

The Pi 4 can capture USB webcam video and run a small CPU demonstration, but
it is much less powerful than a computer with a GPU. The automatic CPU profile
uses the nano pose model at 640 pixels; it may run slowly, especially in a
crowded scene. Start with a short video and measure the actual speed before
calling it real time.
For a stronger student demo, let the Pi capture video and run the heavier
analysis on your computer. This project does not yet include that networked
Pi-to-computer connection.

On Raspberry Pi OS, create and activate a Python virtual environment, install
the requirements, then try:

```bash
python3 main.py --source webcam --camera 0 --agent rule --device cpu
```

If installation fails, check that the OS is 64-bit and that the installed
PyTorch/Ultralytics packages support that OS and Python version. Camera and
model performance depend on the particular Pi, webcam, and scene.

## Learn it in small steps

1. Run the program on a short video and look at the displayed boxes/keypoints.
2. Read `main.py`: it connects the input, detectors, agent, and logger.
3. Read `detectors/fall_detector.py`, then `detectors/fight_detector.py`.
   These are simple rules over pose estimates, not trained fight/fall models.
4. Read `agents/rule_agent.py` to see how event candidates become review
   alerts. `utils/action_executor.py` only prints and logs; it does not call
   anyone.
5. Change one threshold in `.env`, run again, and compare what changed.
6. Label separate real clips and use the real-video evaluation below before
   making any claim about accuracy.

## Evaluation and accuracy

Run the synthetic logic exercise:

```bash
python -m eval.run_eval_synthetic
```

This feeds hand-made keypoint sequences directly to the fall/fight rules. It
checks that the rules respond to those examples. It does **not** test the
webcam, pose model, crowd footage, or real-world accuracy. A perfect score on
those synthetic examples is not a real accuracy result; see
`eval/sample_accuracy_report.txt`.

To evaluate a real video, create a matching CSV using
`eval/LABELING_GUIDE.md`, then run:

```bash
python -m eval.run_eval --video path\to\clip.mp4 --labels eval\labels\clip.csv
```

Report precision, recall, F1, missed events, false alerts, and alert delay
for each event. Keep some entire videos out of threshold tuning so the final
result is measured on footage the system has not already seen. Results from
one camera do not guarantee results from another camera. There is no honest
accuracy percentage to promise until this evaluation has been done on your
intended camera and scene.

## Folder guide

- `main.py` - connects the steps and runs the video loop.
- `config.py` and `.env.example` - settings and thresholds.
- `detectors/` - pose, fall, fight-like movement, and crowd-count logic.
- `agents/` - local rules and optional text-only LLM suggestion.
- `utils/` - camera input, tracking, clips, drawing, and event logs.
- `eval/` - synthetic examples, real-video evaluation, and labeling guide.
- `tests/` - small checks for individual pieces of logic.

## Optional LLM mode

Copy `.env.example` to `.env`, add an API key, and run with `--agent llm`.
This sends a text summary of detector outputs to the configured API. Do not
use this mode if you do not want event metadata sent to an external service.
If the API call fails or returns an invalid response, the program uses the
local rule agent instead.

## Agentic second-opinion verification

When `GROQ_API_KEY` is set in `.env`, a **high-risk decision that has been
sustained for `VERIFY_MIN_HIGH_RISK_SECONDS` (default 2.0)** is also sent to a
two-agent verifier on the Groq API. A **vision agent** (Qwen) receives the
actual snapshot frame as an image plus the detector measurements, and a
separate **decision agent** (GPT-OSS) turns that observation into an
independent `{risk_level, action, reasoning, alert_message}` verdict that is
compared against the local rule decision:

- **Agree + high risk** -> a wa.me WhatsApp message and a `tel:` dialer link
  with the evidence are opened for the designated authority. The message leads
  with `alert_message`, the plain-language incident summary the decision agent
  wrote, followed by the structured verdict lines.
- **Disagree, or an agent fails / returns bad JSON / exceeds its timeout** ->
  a review request wa.me link is opened for `REVIEWER_WHATSAPP_NUMBER`, and
  the program pauses and asks you to type `approve` or `reject` in the
  console. Approving continues the normal flow (escalating if risk is high);
  rejecting skips the action.

Every other frame runs the normal local path with **no network call**: a brief
spike (one fall frame, a momentary density jump) never reaches the LLMs, and
neither does medium risk. Set `VERIFY_MIN_HIGH_RISK_SECONDS=0` to verify every
high-risk frame the moment it appears.

Local rules remain the decision authority; the agents never override them and
never choose actions on their own. Every comparison, including the verifier
latency, is recorded in `logs/verification.csv` so the latency target can be
tracked.

### Setup

One Groq key covers both agents. Create one at https://console.groq.com/keys,
then in `.env`:

```
GROQ_API_KEY=gsk_your_key_here
VISION_MODEL=qwen/qwen3.8-27b
DECISION_MODEL=openai/gpt-oss-120b
```

Both defaults are live Groq model IDs and need no other configuration — the
client is the standard `openai` SDK pointed at `https://api.groq.com/openai/v1`,
so no extra dependency is required beyond what is already in
`requirements.txt`.

### The two agents

The split is deliberate. The two jobs need different capabilities, and keeping
them separate is what makes the comparison in `agents/verifier.py` meaningful:

- **`VisionAgent`** — `qwen/qwen3.8-27b`, multimodal, 131K context. Returns a
  factual observation (`scene`, `activity`, `people_count`, `risk_signals`,
  `confidence`, `notes`) and **is never shown the local verdict**, so it cannot
  echo the answer back. It is also told not to pick a risk level or an action;
  that is the other agent's job. It uses `strict: false` structured outputs
  because Groq only offers advisory schema enforcement for this model, so
  `parse_observation` still has to tolerate an occasional malformed reply.
- **`DecisionAgent`** — `openai/gpt-oss-120b`, text only, ~500 tps. Sees the
  observation plus the detector measurements and returns an independent verdict
  plus `alert_message`. It uses `strict: true`, which is constrained decoding
  that **guarantees** the reply matches the schema, so a malformed verdict is
  not a failure mode. It is also never shown the local verdict.

Switch `DECISION_MODEL` to `openai/gpt-oss-20b` for a cheaper, weaker
alternative, or change `VISION_MODEL` to another Groq multimodal model — the
only hard requirement is that it accepts image input. Both agents degrade to
`insufficient_output` on any failure, which routes to human review instead of
killing the video loop.

### Non-blocking verification

With `VERIFY_ASYNC=1` (default) the verifier runs on a single worker thread and
the video loop keeps reading frames instead of stalling for the duration of
both agent calls. Verdicts are picked up on a later frame, and each one carries
the evidence and decision from the frame that produced it. Only the newest
pending frame is verified — a stale frame is not worth two slow LLM calls.

Set `VERIFY_ASYNC=0` to verify inline (the original behavior), which blocks
the loop for up to `2 * VERIFY_TIMEOUT_SECONDS`.

`VERIFY_LATENCY_TARGET_MS` defaults to 2000 because both agents run on Groq at
roughly 450-500 tps, so decode time should be well under a second.

### Evidence delivery

`wa.me` links are text-only, so a plain escalation message can only carry a
file path. Set `EVIDENCE_SERVER_ENABLED=1` and `EVIDENCE_PUBLIC_HOST=<your
LAN IP>` to start a small static server on `frames/` — the message then
carries a tappable link so the authority sees the photo on their phone
(same network required). Off by default.

### Safety and limits

This layer only notifies designated humans via `AUTHORITY_WHATSAPP_NUMBER`,
`AUTHORITY_PHONE_NUMBER` and `REVIEWER_WHATSAPP_NUMBER` (digits only, country
code included) — it still does **not** contact emergency services, and neither
agent ever places a call or sends a message on its own. WhatsApp receiving is
out of scope: `wa.me` is send-only, so approve/reject happens in the console.

Set `ESCALATION_DRY_RUN=1` to print the links instead of opening them, or
`VERIFY_ENABLED=0` to turn the whole layer off. If either agent call is slower
than `VERIFY_TIMEOUT_SECONDS` it is treated as a failure and routed to human
review rather than stalling the video loop.
