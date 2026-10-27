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
      -> pose model and deterministic rules find candidate events
      -> only a non-none rule action starts event review
      -> save the before, event, and after clip
      -> Qwen reviews three sampled images; GPT-OSS chooses type, risk, summary
      -> Telegram sends the clip and alert to a human reviewer
```

The rule agent is local, predictable, and does not need an API key. It gates
event review even when `--agent llm` is selected. The optional LLM mode is a
separate text-only display suggestion; it cannot start a Telegram alert by
itself. For event review, one Groq API key is shared by Qwen and GPT-OSS.

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
4. Read `agents/rule_agent.py` and `main.py` to see how deterministic
   candidates become event clips, Groq reviews, and Telegram alerts.
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
- `agents/` - deterministic rules, Groq event reviewers, and the optional text-only LLM suggestion.
- `utils/` - camera input, tracking, clips, drawing, and event logs.
- `eval/` - synthetic examples, real-video evaluation, and labeling guide.
- `tests/` - small checks for individual pieces of logic.

## Optional LLM mode

Copy `.env.example` to `.env`, add an API key, and run with `--agent llm`.
This sends a text summary of detector outputs to the configured API. Do not
use this mode if you do not want event metadata sent to an external service.
If the API call fails or returns an invalid response, the program uses the
local rule agent instead.

## Groq event review and Telegram alerts

The event-review path runs after the deterministic `RuleBasedAgent` returns an
action other than `none`. It creates an event-centered clip using
`CLIP_SECONDS` (3 seconds by default), then samples an image from before, at,
and after the candidate event. Qwen reviews those three images plus detector
context; GPT-OSS returns an alert type, risk level, and short summary. The
MP4 is sent to Telegram, not to Groq.

Low-risk decisions are logged without a Telegram alert. Medium- and high-risk
decisions, plus `uncertain`, are sent to the reviewer. If Groq is unavailable
or the review fails, the system sends the deterministic candidate with an
`AI review unavailable` note. If video upload fails or the clip is over
Telegram's 50 MB Bot API limit, it sends a text notice with the upload failure.
The person receiving the Telegram message reviews the clip and decides what to
do. The app does not read Telegram replies or call emergency services.

### Setup

One `GROQ_API_KEY` authenticates both Groq models. Copy `.env.example` to
`.env`, create a Groq API key, create a Telegram bot with BotFather, and set the
bot token and destination chat ID:

```dotenv
GROQ_API_KEY=gsk_your_key_here
VISION_MODEL=qwen/qwen3.8-27b
DECISION_MODEL=openai/gpt-oss-120b
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TELEGRAM_TIMEOUT_SECONDS=10
```

Start the bot in a private chat (or add it to the review group) so it can send
messages there. Keep `.env` private and never commit bot or Groq tokens. The
model defaults already match the values above; set them only if you need to
override a model. Both models use the existing OpenAI-compatible client pointed
at Groq, so no additional SDK is needed.

Run a short video first:

```powershell
python main.py --source path\to\your_video.mp4 --agent rule
```

The deterministic rule gate also runs when `--agent llm` is selected. The
optional LLM mode is separate and still uses `OPENAI_API_KEY`; it does not
create Telegram alerts. Without `GROQ_API_KEY`, a qualifying local rule event
uses the `AI review unavailable` fallback. Without Telegram credentials, the
application records the delivery failure in `logs/verification.csv` but cannot
send a message.

### Review records and limits

`logs/verification.csv` records the local rule decision, AI status, alert type,
risk, summary, model latency, and Telegram delivery status. The review runs on
one background worker so video capture continues while Groq and Telegram are
responding. Repeated frames for the same event are deduplicated by the rule
action cooldown. If the bounded queue is full, the event is logged as
`review_queue_full`.

This is a student prototype. Fall, fight-like movement, crowd density, and
inactivity are candidate signals; lighting, occlusion, dancing, and ordinary
movement can produce errors. A human must review alerts and choose any
response.
