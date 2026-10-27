# Groq and Telegram Event Review Design

**Status:** Approved by the user on 2026-10-27

## Goal

After the project's deterministic rules identify a candidate alert, use Groq's Qwen vision model to inspect the event, GPT-OSS to decide its type and risk, and a Telegram bot to send the reviewer a concise summary together with the event clip.

This is a human-review workflow. It does not identify people, infer intent or guilt, dispatch emergency services, or act on Telegram replies.

## Existing project context

The real-time loop in `main.py` runs pose estimation and local fall, fight-like movement, inactivity, and crowd-density detectors. `RuleBasedAgent` assigns a local risk/action. The existing Groq verifier is a second opinion for sustained high-risk events and sends one JPEG snapshot to Qwen, then detector context to GPT-OSS. `ClipBuffer` currently retains pre-trigger frames; event-centered post-trigger capture is needed for a clip that includes what happened after the candidate was detected. The current notification utilities open prefilled WhatsApp links rather than sending messages.

## Chosen approach

Use an event-triggered, asynchronous workflow. The deterministic rule agent remains the first gate. Do not call Groq for a `none` action. For each distinct non-`none` rule decision, capture a single incident record and run one Qwen-to-GPT-OSS review. Do not run vision inference continuously on every frame.

For candidates handled by this workflow, Telegram is the only outbound notification channel: do not also open WhatsApp/tel links or block on a console approval prompt. Other existing behavior outside this event-review path should remain scoped and unchanged.

## Event data flow

1. **Detect and gate locally.** The frame loop runs the existing detectors and `RuleBasedAgent`. Only a non-`none` deterministic action creates an AI review candidate. The deterministic gate must still apply when the optional `--agent llm` mode is selected; that mode cannot create a Telegram event by itself.
2. **Build event evidence.** Maintain a rolling frame buffer. For each candidate, preserve pre-event frames, the candidate frame, and post-event frames, then finalize an event-centered MP4. Use the existing `CLIP_SECONDS` setting as the clip duration (3 seconds by default), split between pre-event and post-event footage. If a source ends before post-roll completes, finalize with the frames available. Save the completed clip in `clips/`.
3. **Sample frames for Groq.** Select three JPEGs from the event timeline: before the candidate, at the candidate, and after it. Send these images and the detector measurements to Qwen `qwen/qwen3.8-27b`. Groq's documented vision interface accepts image inputs and supports at most three images in one request, so this design sends sampled images rather than an MP4 to Groq. [Groq vision documentation](https://console.groq.com/docs/vision)
4. **Make a structured decision.** Qwen returns factual visual observations only. GPT-OSS `openai/gpt-oss-120b` receives that observation plus detector context and returns validated fields: `alert_type`, `risk_level`, and `summary`. Supported types are the project's existing candidate classes: `possible_fall`, `fight_like_movement`, `crowd_density`, and `inactivity`, plus `uncertain` when the images do not support a clear classification. Risk is `low`, `medium`, or `high`. The summary is short, factual, and avoids identity or intent claims.
5. **Notify the reviewer.** If the GPT-OSS result is medium/high risk, or is uncertain, send one Telegram video message with the event MP4 and a caption containing alert type, risk level, summary, and timestamp. Low/no-visible-risk results are logged without a notification. Telegram's Bot API supports video upload through `sendVideo`; the hosted Bot API currently limits uploaded videos to 50 MB. If the file exceeds the limit or cannot be attached, send the text alert and log that the clip upload failed. [Telegram Bot API](https://core.telegram.org/bots/api)
6. **Keep the live loop responsive.** Capture, Groq review, and Telegram delivery must not block frame acquisition. Queue distinct incidents rather than replacing a waiting event with a newer frame. Deduplicate repeated detections for the same tracked event; use a cooldown for events without a stable track ID. Drain outstanding work when a video file ends.

## Failure handling

- If Qwen, GPT-OSS, or structured-output validation fails, send the reviewer the deterministic candidate type and local risk with a clear `AI review unavailable` label, plus the saved clip when possible. Do not present a fallback as an AI classification.
- If Telegram credentials are missing, keep processing and record the candidate and evidence path locally; report that delivery is disabled.
- If Telegram returns an error or times out, log the failure and continue video processing. Do not retry indefinitely or block the capture loop.
- If the source ends before all three temporal samples are available, do not claim a complete AI review. Send the deterministic candidate with an `AI review unavailable` note and the clip when possible. If a candidate clip cannot be finalized, send a text notification with the local evidence path and an explicit missing-clip note.

## Configuration and privacy

Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to the local `.env` setup and add blank placeholders to `.env.example`. Never commit tokens. The bot must be started by the reviewer and added to the destination chat before it can send there. Keep Telegram delivery disabled when either required setting is absent.

Only three sampled event images and detector context go to Groq. The MP4 goes to the configured Telegram chat; it is not sent to Groq. Telegram messages are one-way notifications: the program does not poll for reviewer replies or treat replies as approval.

## Scope and success criteria

The feature classifies only candidate types already represented by this project's rules. It does not add weapon, fire, theft, or other unsupported detectors. It adds no automatic response beyond notifying the configured human reviewer.

The design is satisfied when a no-alert local decision makes no Groq or Telegram request; a deterministic candidate produces one event-centered clip and three temporal images; Qwen observation feeds GPT-OSS structured classification; an eligible result reaches Telegram with its clip and requested summary fields; and API failures do not stop the video loop or silently discard an incident.

## Alternatives considered

- **Continuous vision review:** rejected because it would call Groq on routine frames and add latency and API traffic.
- **Send the MP4 to Groq:** the current documented Groq vision interface uses image inputs. The chosen flow sends three sampled images to Groq and the full event clip to Telegram.


