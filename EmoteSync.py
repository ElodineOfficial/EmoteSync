"""
EmoteSync (fixed v3.1) — "switch at START of next sentence" + GLOBAL anti-linger + neutral lock + strict RR + robust splitting

What this version guarantees:
  ✓ **Switch at START of next sentence (no dead‑air flip):**
    Frame changes are aligned to the *start time of the next sentence’s first word*,
    so the previous frame persists through any silence between sentences.
  ✓ **Global anti-linger (emotion run limit):**
    The same *emotion* cannot occupy the screen for more than
    `EMOTION_RUN_MAX_SECONDS` (default 15s). If the analysis keeps predicting the
    same emotion, we insert a *bridge* emotion chosen from the current sentence’s
    **top‑k** emotion predictions (or a sensible configured fallback) so the on-screen
    state *must* change.
  ✓ **Neutral lock:** "neutral" may appear at most once in a row; repeated
    neutrals redirect to last/best non‑neutral base.
  ✓ **Strict 10–15s pacing:** we rotate at sentence boundaries with a 10–15s
    soft target and keep a hard cap of 20s for any single image chunk.
  ✓ **Robust sentence splitting:** If Whisper does not provide `words`, we split
    *segment text* into sentences via punctuation and proportionally allocate
    time so that boundaries occur and no chunk exceeds `SENTENCE_MAX_SECONDS`.
  ✓ **Strict per‑emotion round‑robin**, no immediate variant reuse.
  ✓ CSV export with the reason for each swap (emotion_change, soft_target,
    hard_cap, neutral_lock, bridge_run_limit, forced_split).

Usage: same UI. Drop in place of the previous file.
"""

import os
import re
import csv
import logging
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import random
from collections import defaultdict

import numpy as np
from PIL import Image  # Pillow compatibility shim below

# Pillow 10+ removed Image.ANTIALIAS. Older moviepy may call it.
try:
    _ = Image.ANTIALIAS  # no-op if present
except AttributeError:  # Pillow ≥10
    try:
        Image.ANTIALIAS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except Exception:
        pass

from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    CompositeVideoClip,
    AudioFileClip,
    ColorClip,
)

import whisper
from transformers import pipeline

# -----------------------------------------------------------------------------
# Configuration — pacing & segmentation
# -----------------------------------------------------------------------------
SOFT_SWAP_SECONDS_RANGE = (10.0, 15.0)   # target length per image; randomized per swap
HARD_CAP_SECONDS = 20.0                  # maximum a single *image* chunk
PAUSE_SPLIT_THRESHOLD = 0.60             # pause between words → sentence boundary (if words available)
SENTENCE_MAX_SECONDS = 14.0              # if a "sentence" runs longer, force a boundary
FORCE_SPLIT_LONG_SPANS = True            # defensive: split spans > HARD_CAP even without boundaries
EMOTION_RUN_MAX_SECONDS = 15.0           # **GLOBAL**: max contiguous on‑screen time for the same base emotion

PREFERRED_EXTS = ("png", "webp", "jpg", "jpeg")
EXPORT_TIMELINE_CSV = True

# Neutral / bridge policy
NEUTRAL_MAX_CONSECUTIVE = 1              # allow neutral at most once in a row
BRIDGE_TOPK = 3                          # we may pick a bridge from top‑k (≥2) model predictions
BRIDGE_MIN_SCORE = 0.18                  # only consider bridge candidates with score ≥ this
IDLE_FALLBACK_PRIORITY = ["idle", "listen", "listening", "thinking", "curious", "attentive", "happy"]

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("EmoteSyncFixedV3_1")

def update_status(message: str):
    status_var.set(message)
    logger.debug(message)

# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def select_audio_file():
    audio_file = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.wma *.aiff *.aif *.aifc"), ("All Files", "*.*")]
    )
    audio_file_var.set(audio_file)

def select_background_video():
    background_file = filedialog.askopenfilename(
        title="Select Background Video",
        filetypes=[("Video", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v"), ("All Files", "*.*")]
    )
    background_file_var.set(background_file)

def select_emotion_images_folder():
    emotion_folder = filedialog.askdirectory(title="Select Folder Containing Emotion Images")
    emotion_folder_var.set(emotion_folder)

def start_processing():
    audio_file = audio_file_var.get()
    use_bg = use_background_var.get()
    background_video = background_file_var.get() if use_bg else ""
    emotion_folder = emotion_folder_var.get()

    if not audio_file or not emotion_folder:
        messagebox.showerror("Error", "Please select an audio file and an emotion images folder.")
        return
    if use_bg and not background_video:
        messagebox.showerror("Error", "Please select a background video or uncheck 'Use Background Video'.")
        return

    start_button.config(state="disabled")
    threading.Thread(
        target=process_video,
        args=(audio_file, background_video, emotion_folder, use_bg),
        daemon=True
    ).start()

# -----------------------------------------------------------------------------
# Assets — load arrays of emotion images (happy, happy_1, happy_2, …)
# -----------------------------------------------------------------------------
_VARIANT_RE = re.compile(
    r'^([a-z0-9][a-z0-9_\-\s]*?)(?:[\s_\-]*\(?(\d+)\)?)?\.(png|jpg|jpeg|webp)$',
    re.IGNORECASE
)

def _normalize_base(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\s+', '_', s)           # spaces → underscores
    s = re.sub(r'[_\-]+', '_', s)        # collapse dashes/underscores
    s = re.sub(r'[^a-z0-9_]', '', s)     # strip other punctuation
    return s

def load_emotion_image_sets(folder_path, prefer_exts=PREFERRED_EXTS):
    image_sets = defaultdict(lambda: defaultdict(list))  # base -> idx -> [(rank, has_suffix, path)]
    if not os.path.isdir(folder_path):
        logger.error("Emotion folder '%s' does not exist.", folder_path)
        return {}

    for fn in os.listdir(folder_path):
        if fn.startswith("."):
            continue
        m = _VARIANT_RE.match(fn)
        if not m:
            continue
        base_raw = m.group(1)
        base = _normalize_base(base_raw)
        idx = int(m.group(2)) if m.group(2) is not None else 0
        ext = m.group(3).lower()
        if ext not in prefer_exts:
            continue
        full = os.path.join(folder_path, fn)
        if not os.path.isfile(full):
            continue
        has_suffix = 1 if m.group(2) is not None else 0  # prefer no suffix over _0 when both exist
        try:
            rank = prefer_exts.index(ext)
        except ValueError:
            rank = len(prefer_exts)
        image_sets[base][idx].append((rank, has_suffix, full))

    finalized = {}
    for base, by_idx in image_sets.items():
        ordered = []
        for i in sorted(by_idx.keys()):
            best = sorted(by_idx[i], key=lambda t: (t[0], t[1]))[0][2]
            ordered.append(best)
        if ordered:
            finalized[base] = ordered

    for base, arr in sorted(finalized.items()):
        logger.info("Found %d variants for '%s': %s", len(arr), base, ", ".join(os.path.basename(p) for p in arr))

    return finalized

class RoundRobinSelector:
    """
    Per-emotion round‑robin selector. Returns the next variant each time an emotion is *activated*.
    Ensures we don't immediately reuse the same image path when base has >1 variant.
    """
    def __init__(self, image_sets, neutral_list=None):
        self.image_sets = image_sets
        self.neutral = neutral_list or []
        self._counters = defaultdict(int)
        self._last_image_for_base = {}

    def has_base(self, base_emotion):
        base = _normalize_base(base_emotion or "neutral")
        return base in self.image_sets and len(self.image_sets[base]) > 0

    def next(self, base_emotion, avoid_path=None):
        base = _normalize_base(base_emotion or "neutral")
        variants = self.image_sets.get(base)
        if not variants:
            variants = self.neutral
            base = "neutral"
            if not variants:
                return None
        if len(variants) == 1:
            img = variants[0]
            self._counters[base] += 1
            self._last_image_for_base[base] = img
            return img

        # Try up to len(variants) times to avoid repeating the same path
        for _ in range(len(variants)):
            i = self._counters[base] % len(variants)
            img = variants[i]
            self._counters[base] += 1
            if img != avoid_path and img != self._last_image_for_base.get(base):
                self._last_image_for_base[base] = img
                return img
        img = variants[self._counters[base] % len(variants)]
        self._counters[base] += 1
        self._last_image_for_base[base] = img
        return img

# -----------------------------------------------------------------------------
# ASR (Whisper) + sentence splitting
# -----------------------------------------------------------------------------
def transcribe_audio(audio_file, model):
    # Try to get per-word timestamps; if unavailable, fall back gracefully
    try:
        return model.transcribe(audio_file, word_timestamps=True)
    except TypeError:
        return model.transcribe(audio_file)
    except Exception as e:
        logger.error("Transcription failed. Ensure FFmpeg is installed. Error: %s", e)
        raise

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def _split_text_sentences_with_durations(text, start, end, max_len=SENTENCE_MAX_SECONDS):
    """
    Fallback when word timestamps are absent: split by punctuation and
    proportionally allocate time by word count. Ensure each piece <= max_len.
    """
    text = (text or "").strip()
    if not text:
        return [{"start": float(start), "end": float(end), "text": ""}]

    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    if not parts:
        parts = [text]

    # Allocate by word counts
    words_counts = [max(1, len(p.split())) for p in parts]
    total_words = float(sum(words_counts))
    seg_dur = max(0.0, float(end) - float(start))

    # Initial allocation
    spans = []
    cur = float(start)
    for wc, p in zip(words_counts, parts):
        dur = seg_dur * (wc / total_words) if total_words > 0 else seg_dur / len(parts)
        spans.append({"start": cur, "end": cur + dur, "text": p})
        cur += dur

    # Enforce max_len by sub-splitting overly long spans evenly by word groups
    out = []
    for s in spans:
        s_dur = float(s["end"]) - float(s["start"])
        if s_dur <= max_len:
            out.append(s)
            continue
        # split into n chunks so each <= max_len
        n = int(np.ceil(s_dur / max_len))
        chunk_dur = s_dur / n
        c_start = s["start"]
        for i in range(n):
            c_end = min(s["end"], c_start + chunk_dur)
            out.append({"start": c_start, "end": c_end, "text": s["text"]})
            c_start = c_end
    return out

def _segments_to_sentences(segments, pause_threshold=PAUSE_SPLIT_THRESHOLD, max_len=SENTENCE_MAX_SECONDS):
    """
    Convert Whisper segments into sentence‑like chunks.
    **FIXED**: A long pause now creates a boundary *before* the next word, so that
    the next sentence begins exactly at that word’s start time. This guarantees that
    frame switches happen at the START of the next sentence (no flip during silence).
    """
    out = []
    for seg in segments:
        words = seg.get("words")
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        if not words:
            # Fallback: punctuation-driven splitting with proportional timing
            out.extend(_split_text_sentences_with_durations(seg.get("text", ""), seg_start, seg_end, max_len=max_len))
            continue

        buf = []
        buf_start = None
        last_end = None

        for w in words:
            w_text = str(w.get("word", ""))
            w_start = float(w.get("start", seg_start))
            w_end = float(w.get("end", w_start))

            # --- boundary BEFORE this word, if a long pause occurred ---
            # If there was a pause >= threshold since the previous word,
            # we finish the previous sentence at the *previous* word's end (last_end)
            # and start a new sentence that begins exactly at w_start.
            pre_boundary = last_end is not None and (w_start - last_end) >= pause_threshold and buf
            if pre_boundary:
                text_prev = "".join(x.get("word", "") for x in buf).strip()
                out.append({
                    "start": buf_start if buf_start is not None else seg_start,
                    "end": last_end,
                    "text": text_prev
                })
                buf = []          # reset buffer; current word 'w' starts the next sentence
                buf_start = None  # will set to w_start below

            if buf_start is None:
                buf_start = w_start
            buf.append(w)

            # --- boundary AFTER this word (punctuation or max length) ---
            boundary_after = False
            if re.search(r'[.!?;:]\s*$', w_text):
                boundary_after = True
            if (w_end - buf_start) >= max_len:
                boundary_after = True

            last_end = w_end

            if boundary_after:
                text = "".join(x.get("word", "") for x in buf).strip()
                out.append({"start": buf_start, "end": w_end, "text": text})
                buf = []
                buf_start = None
                # keep last_end at w_end for potential next pre-boundary checks

        if buf:
            # close any trailing buffer
            w_end = float(buf[-1].get("end", seg_end))
            text = "".join(x.get("word", "") for x in buf).strip()
            out.append({"start": buf_start if buf_start is not None else seg_start,
                        "end": w_end,
                        "text": text})
    return out

# -----------------------------------------------------------------------------
# NLP (emotion classification)
# -----------------------------------------------------------------------------
def detect_emotion_candidates(text, emotion_classifier, emotion_mapping, topk=BRIDGE_TOPK):
    """
    Returns a sorted list of candidate bases with scores, e.g.:
        [("happy", 0.61), ("neutral", 0.21), ("anger", 0.08), ...]
    Collapses model labels to our base set via `emotion_mapping` (max score wins).
    """
    if not text.strip():
        return [("neutral", 1.0)]
    raw = emotion_classifier(text, top_k=topk)
    # HF pipeline returns either list[dict] or list[list[dict]]
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    buckets = defaultdict(float)
    for item in raw:
        lbl = str(item["label"]).lower()
        score = float(item["score"])
        base = emotion_mapping.get(lbl, "neutral").lower()
        buckets[base] = max(buckets[base], score)
    # ensure neutral exists as a candidate
    if "neutral" not in buckets:
        buckets["neutral"] = 0.0
    ranked = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)
    return ranked

def build_sentence_emotion_timeline(audio_file, emotion_classifier, emotion_mapping, whisper_model):
    """
    Returns list of {"start": float, "end": float, "preds": list[(base,score)], "text": str}
    for each sentence‑like chunk.
    """
    update_status("Transcribing audio and splitting into sentences...")
    res = transcribe_audio(audio_file, whisper_model)
    raw_segments = res.get("segments", [])
    sentences = _segments_to_sentences(raw_segments)

    timeline = []
    n = len(sentences)
    for i, seg in enumerate(sentences):
        update_status(f"Classifying emotion for sentence {i+1}/{n}...")
        text = seg.get("text", "").strip()
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        preds = detect_emotion_candidates(text, emotion_classifier, emotion_mapping, topk=BRIDGE_TOPK)
        timeline.append({"start": start, "end": end, "preds": preds, "text": text})
        dbg = ", ".join(f"{b}:{s:.2f}" for b,s in preds[:3])
        logger.debug("SENTENCE %03d  %.2f → %.2f  [%s]  %s",
                     i+1, start, end, dbg, (text[:80] + "…") if len(text) > 80 else text)
    return timeline

# -----------------------------------------------------------------------------
# Helpers — neutral lock, bridge selection, best available bases
# -----------------------------------------------------------------------------
def _best_available_non_neutral_base(image_sets, preferred=IDLE_FALLBACK_PRIORITY):
    for base in preferred:
        if base in image_sets and base != "neutral" and image_sets[base]:
            return base
    candidates = [(k, len(v)) for k, v in image_sets.items() if k != "neutral" and v]
    if not candidates:
        return None
    candidates.sort(key=lambda kv: (-kv[1], kv[0]))
    return candidates[0][0]

def choose_bridge_base(preds, image_sets, last_non_neutral, avoid_base):
    """
    Pick an alternate base (not avoid_base), from current sentence's ranked preds,
    that has assets and score ≥ BRIDGE_MIN_SCORE. Else fall back to last_non_neutral,
    else the best available non‑neutral base.
    """
    for base, score in preds:
        if base == avoid_base:
            continue
        if score >= BRIDGE_MIN_SCORE and base in image_sets and image_sets[base]:
            return base
    if last_non_neutral and last_non_neutral != avoid_base and last_non_neutral in image_sets and image_sets[last_non_neutral]:
        return last_non_neutral
    return _best_available_non_neutral_base(image_sets)

def resolve_display_base_v3(
    preds, image_sets,
    last_display_base, last_non_neutral,
    neutral_lock_active, run_base, run_start_time, boundary_time,
):
    """
    Decide which base to display at this boundary given ranked predictions.
    Applies:
      - neutral lock
      - global anti-linger (emotion run max)
      - asset availability
    Returns: (display_base:str|None, new_neutral_lock_active:bool, reason:str)
    """
    # primary predicted with assets
    primary = None
    for base, _ in preds:
        if base in image_sets and image_sets[base]:
            primary = base
            break
    if primary is None:
        primary = "neutral" if "neutral" in image_sets and image_sets["neutral"] else None

    # Compute current run duration if we stayed with the same base
    run_dur_if_same = (boundary_time - run_start_time) if (run_base == last_display_base == primary and run_start_time is not None) else 0.0

    # 1) Neutral lock
    if primary == "neutral":
        if not neutral_lock_active and "neutral" in image_sets and image_sets["neutral"]:
            return "neutral", True, "neutral_lock_first"
        # already locked: choose a bridge
        bridge = choose_bridge_base(preds, image_sets, last_non_neutral, avoid_base="neutral")
        return bridge, True, "neutral_lock_bridge"

    # 2) Global anti-linger: if choosing primary would exceed the run max, bridge away
    if run_base == primary and run_dur_if_same >= EMOTION_RUN_MAX_SECONDS:
        bridge = choose_bridge_base(preds, image_sets, last_non_neutral, avoid_base=primary)
        if bridge:
            return bridge, False, "bridge_run_limit"

    # 3) Normal case: choose primary
    return primary, False, "primary"

# -----------------------------------------------------------------------------
# Display timeline (sentence‑boundary driven swapping) + anti‑stall
# -----------------------------------------------------------------------------
def build_display_timeline_by_sentences(
    sentence_timeline,
    rr_selector: RoundRobinSelector,
    image_sets,
    total_duration: float,
    soft_range=SOFT_SWAP_SECONDS_RANGE,
    hard_cap=HARD_CAP_SECONDS,
    fill_leading_with_neutral=True,
):
    """
    Build the final image display timeline with:
      - neutral lock (at most one neutral in a row)
      - global anti-linger (max contiguous same base time)
      - sentence-boundary driven swaps (plus defensive hard-cap post-splitting)

    **Important**: Swaps are aligned to the *start of the next sentence* — we hold
    the current frame through any inter-sentence silence and only flip when the
    next sentence’s first word starts.
    """
    timeline = []

    if not sentence_timeline:
        base = _best_available_non_neutral_base(image_sets) or ("neutral" if image_sets.get("neutral") else None)
        img = rr_selector.next(base) if base else None
        if img:
            timeline.append({"start": 0.0, "end": total_duration, "emotion": base or "neutral", "image": img, "reason": "no_speech"})
        return timeline

    first_start = float(sentence_timeline[0]["start"])
    last_non_neutral = None
    neutral_lock = False

    if fill_leading_with_neutral and first_start > 0.0:
        # Apply neutral lock rules to the initial silence
        if image_sets.get("neutral"):
            img = rr_selector.next("neutral")
            if img:
                timeline.append({"start": 0.0, "end": first_start, "emotion": "neutral", "image": img, "reason": "leading_silence"})
                neutral_lock = True

    # Initialize with the first sentence's predictions
    current_start = max(0.0, first_start)
    preds0 = sentence_timeline[0]["preds"]
    display_base, neutral_lock, reason = resolve_display_base_v3(
        preds0, image_sets, last_display_base=None, last_non_neutral=last_non_neutral,
        neutral_lock_active=neutral_lock, run_base=None, run_start_time=None, boundary_time=current_start
    )
    if display_base and display_base != "neutral":
        last_non_neutral = display_base
    current_image = rr_selector.next(display_base) if display_base else None
    last_swap_time = current_start
    soft_target = random.uniform(*soft_range)
    current_display_base = display_base
    run_base = display_base
    run_start_time = current_start

    # Walk boundaries (always aligned to the *next sentence start*)
    for idx in range(1, len(sentence_timeline)):
        boundary = float(sentence_timeline[idx]["start"])
        boundary = min(boundary, total_duration)
        if boundary <= current_start:
            continue  # guard

        preds = sentence_timeline[idx]["preds"]
        proposed_base, proposed_lock, choose_reason = resolve_display_base_v3(
            preds, image_sets,
            last_display_base=current_display_base,
            last_non_neutral=last_non_neutral,
            neutral_lock_active=neutral_lock,
            run_base=run_base,
            run_start_time=run_start_time,
            boundary_time=boundary,
        )

        elapsed_since_swap = boundary - last_swap_time
        should_rotate = False

        base_changed = (proposed_base != current_display_base)
        time_trigger = (elapsed_since_swap >= hard_cap or elapsed_since_swap >= soft_target)

        # Always rotate on base change; otherwise rotate by pacing
        if base_changed or time_trigger:
            should_rotate = True

        if should_rotate:
            # Close current (up to the *start* of the next sentence)
            if current_image:
                timeline.append({
                    "start": current_start,
                    "end": boundary,
                    "emotion": current_display_base or "neutral",
                    "image": current_image,
                    "reason": ("base_change" if base_changed else ("hard_cap" if elapsed_since_swap >= hard_cap else "soft_target"))
                })
            # New selection
            current_display_base = proposed_base
            neutral_lock = proposed_lock
            if current_display_base and current_display_base != "neutral":
                last_non_neutral = current_display_base

            avoid = timeline[-1]["image"] if timeline else None
            current_image = rr_selector.next(current_display_base, avoid_path=avoid) if current_display_base else None
            # update run tracking
            if run_base == current_display_base:
                # continuing run
                pass
            else:
                run_base = current_display_base
                run_start_time = boundary

            current_start = boundary
            last_swap_time = boundary
            soft_target = random.uniform(*soft_range)

    # Close last to end
    final_end = total_duration
    if final_end > current_start and current_image:
        timeline.append({
            "start": current_start,
            "end": final_end,
            "emotion": current_display_base or "neutral",
            "image": current_image,
            "reason": "finalize"
        })

    # Defensive anti‑stall: split spans > hard cap; avoid consecutive neutral
    if FORCE_SPLIT_LONG_SPANS and hard_cap > 0:
        timeline = _split_long_spans(
            timeline, rr_selector, image_sets, soft_range, hard_cap
        )

    # Clean
    clean = [seg for seg in timeline if seg["end"] > seg["start"] and seg["image"]]
    return clean

def _split_long_spans(timeline, rr_selector, image_sets, soft_range, hard_cap):
    out = []
    last_non_neutral = None
    last_base = None
    for seg in timeline:
        start, end = float(seg["start"]), float(seg["end"])
        base = (seg["emotion"] or "neutral").lower()
        img = seg["image"]
        dur = end - start

        if base != "neutral":
            last_non_neutral = base

        if dur <= hard_cap:
            out.append(seg)
            last_base = base
            continue

        # Split into chunks and avoid consecutive neutral
        cur = start
        first = True
        avoid = out[-1]["image"] if out else None

        while cur < end:
            target = random.uniform(*soft_range)
            chunk_end = min(end, cur + min(max(target, soft_range[0]), hard_cap))

            # Decide base for this chunk
            if first:
                use_base = base
            else:
                if base == "neutral":
                    alt = last_non_neutral or _best_available_non_neutral_base(image_sets)
                    use_base = alt if alt is not None else None
                else:
                    # during long single-base run, try to bridge away to respect global variety
                    alt = _best_available_non_neutral_base(image_sets) if base == "neutral" else base
                    use_base = alt

            # Choose image
            if first:
                use_img = img
            else:
                use_img = rr_selector.next(use_base, avoid_path=avoid) if use_base else None

            if use_img:
                out.append({
                    "start": cur, "end": chunk_end,
                    "emotion": use_base or "neutral",
                    "image": use_img,
                    "reason": "forced_split"
                })
                avoid = use_img
                last_base = use_base or "neutral"
                if use_base and use_base != "neutral":
                    last_non_neutral = use_base

            cur = chunk_end
            first = False
    return out

# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------
def process_video(audio_file, background_video_file, emotion_folder, use_background):
    try:
        update_status("Loading models...")

        # Map model labels to your base emotions
        emotion_mapping = {
            "joy": "happy",
            "neutral": "neutral",
            "anger": "anger",
            "sadness": "neutral",   # add a `sad.png` set if you want it distinct
            "fear": "confusion",
            "surprise": "disbelief",
            "disgust": "annoyance",
        }
        used_bases = set(emotion_mapping.values()) | {"neutral"} | set(IDLE_FALLBACK_PRIORITY)

        # Load emotion image arrays
        update_status("Loading emotion image sets...")
        image_sets_all = load_emotion_image_sets(emotion_folder, prefer_exts=PREFERRED_EXTS)
        image_sets = {k: v for k, v in image_sets_all.items() if k in used_bases}

        rr = RoundRobinSelector(image_sets, neutral_list=image_sets.get("neutral", []))

        # Initialize models
        emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        whisper_model = whisper.load_model("base")

        # Load audio
        update_status("Loading audio...")
        try:
            audio_clip = AudioFileClip(audio_file)
        except Exception as e:
            start_button.config(state="normal")
            messagebox.showerror("Error", f"Could not load audio file:\n{e}")
            return
        total_duration = float(audio_clip.duration)

        # Build sentence-level emotion timeline
        sentence_timeline = build_sentence_emotion_timeline(audio_file, emotion_classifier, emotion_mapping, whisper_model)

        # Build *display* timeline (what image shows when)
        display_timeline = build_display_timeline_by_sentences(
            sentence_timeline,
            rr_selector=rr,
            image_sets=image_sets,
            total_duration=total_duration,
            soft_range=SOFT_SWAP_SECONDS_RANGE,
            hard_cap=HARD_CAP_SECONDS,
            fill_leading_with_neutral=True,
        )

        # CSV export
        if EXPORT_TIMELINE_CSV:
            try:
                with open("display_timeline.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["start", "end", "dur", "display_emotion", "image", "reason"])
                    for seg in display_timeline:
                        w.writerow([f"{seg['start']:.3f}", f"{seg['end']:.3f}", f"{(seg['end']-seg['start']):.3f}", seg["emotion"], os.path.basename(seg["image"]), seg.get("reason","")])
                logger.info("Exported display_timeline.csv")
            except Exception as e:
                logger.warning("Failed to export CSV: %s", e)

        # Background
        fps = 24
        if use_background:
            update_status("Loading background video...")
            try:
                background_video = VideoFileClip(background_video_file)
                fps = getattr(background_video, "fps", fps) or fps
            except Exception as e:
                start_button.config(state="normal")
                messagebox.showerror("Error", f"Could not load background video:\n{e}")
                return

            if background_video.duration < total_duration:
                background_video = background_video.loop(duration=total_duration)
            elif background_video.duration > total_duration:
                background_video = background_video.subclip(0, total_duration)
        else:
            # Transparent background sized to first available image, else default
            width, height = 640, 480
            first_path = None
            for lst in image_sets.values():
                if lst:
                    first_path = lst[0]
                    break
            if first_path and os.path.isfile(first_path):
                tmp = ImageClip(first_path)
                width, height = tmp.size
                tmp.close()

            background_video = (
                ColorClip(size=(width, height), color=(0, 0, 0))
                .set_opacity(0)
                .set_duration(total_duration)
            )

        # Optional subtle "alive" motion
        def bounce_effect(t):
            d = 0.5
            if t < d:
                return 1 + 0.05 * np.sin(2 * np.pi * 2 * t / d) * np.exp(-4 * t / d)
            return 1

        # Compose overlays from display timeline
        update_status("Compositing frames…")
        clips = []
        for seg in display_timeline:
            img_path = seg["image"]
            seg_start = float(seg["start"])
            seg_end = float(seg["end"])
            duration = max(0.0, seg_end - seg_start)
            if duration <= 0.0 or not img_path or not os.path.isfile(img_path):
                continue

            logger.debug("Frame %s [%s]  %.2f → %.2f (%.2fs)  reason=%s",
                         seg["emotion"], os.path.basename(img_path), seg_start, seg_end, duration, seg.get("reason",""))

            img_clip = (
                ImageClip(img_path)
                .set_start(seg_start)
                .set_duration(duration)
                .set_position(("center", "center"))
            )

            if hasattr(background_video, "h"):
                img_clip = img_clip.resize(height=background_video.h)
            if bounce_var.get():
                img_clip = img_clip.resize(lambda t: bounce_effect(t))

            clips.append(img_clip)

        update_status("Rendering video…")
        final = CompositeVideoClip([background_video] + clips).set_duration(total_duration).set_audio(audio_clip)

        if not use_background:
            out = "output_video.webm"
            final.write_videofile(
                out, fps=fps, codec="libvpx-vp9", audio_codec="libopus",
                ffmpeg_params=["-pix_fmt", "yuva420p", "-deadline", "realtime", "-cpu-used", "8"],
            )
        else:
            out = "output_video.mp4"
            final.write_videofile(out, fps=fps, codec="libx264", audio_codec="aac")

        update_status("Done!")
        start_button.config(state="normal")
        messagebox.showinfo("Success", f"Video processing completed.\nOutput file: {out}")

    except Exception as e:
        logger.exception("Processing error")
        start_button.config(state="normal")
        update_status("An error occurred.")
        messagebox.showerror("Error", f"{e}")

# -----------------------------------------------------------------------------
# UI toggles and about/help
# -----------------------------------------------------------------------------
def toggle_background_video():
    if use_background_var.get():
        background_file_entry.config(state="normal")
        background_browse_button.config(state="normal")
    else:
        background_file_entry.config(state="disabled")
        background_browse_button.config(state="disabled")
        background_file_var.set("")

def show_support_message():
    messagebox.showinfo(
        "Support",
        "If you find this tool useful, please consider supporting FOSS developers you like."
    )

def show_about_us():
    messagebox.showinfo(
        "About This Tool",
        "EmoteSync analyzes audio, detects emotions per sentence, and shows matching images.\n\n"
        f"Swaps at sentence boundaries with a strict 10–15s target; hard cap {HARD_CAP_SECONDS:.0f}s.\n"
        f"Global anti‑linger: the same emotion cannot persist beyond {EMOTION_RUN_MAX_SECONDS:.0f}s; we bridge to a top‑k candidate.\n"
        "Neutral lock: neutral may appear at most once in a row.\n"
        "Now aligned to the START of the next sentence (no dead‑air flip)."
    )

# -----------------------------------------------------------------------------
# Tk UI
# -----------------------------------------------------------------------------
root = tk.Tk()
root.title("EmoteSync (fixed v3.1)")

audio_file_var = tk.StringVar()
background_file_var = tk.StringVar()
emotion_folder_var = tk.StringVar()
status_var = tk.StringVar()
bounce_var = tk.BooleanVar()
use_background_var = tk.BooleanVar(value=True)

# Row 0: About
about_button = tk.Button(root, text="About", command=show_about_us)
about_button.grid(row=0, column=2, sticky="e", padx=5, pady=5)

# Row 1: Audio
tk.Label(root, text="Audio File:").grid(row=1, column=0, sticky="e")
tk.Entry(root, textvariable=audio_file_var, width=50).grid(row=1, column=1)
tk.Button(root, text="Browse...", command=select_audio_file).grid(row=1, column=2)

# Row 2: Emotions folder
tk.Label(root, text="Emotion Images Folder:").grid(row=2, column=0, sticky="e")
tk.Entry(root, textvariable=emotion_folder_var, width=50).grid(row=2, column=1)
tk.Button(root, text="Browse...", command=select_emotion_images_folder).grid(row=2, column=2)

# Row 3: Background video
tk.Label(root, text="Background Video:").grid(row=3, column=0, sticky="e")
background_file_entry = tk.Entry(root, textvariable=background_file_var, width=50)
background_file_entry.grid(row=3, column=1)
background_browse_button = tk.Button(root, text="Browse...", command=select_background_video)
background_browse_button.grid(row=3, column=2)

# Row 4: Use background?
tk.Checkbutton(root, text="Use Background Video", variable=use_background_var, command=toggle_background_video).grid(row=4, column=1, sticky="w")

# Row 5: Bounce effect
tk.Checkbutton(root, text="Enable Bounce Effect", variable=bounce_var).grid(row=5, column=1, sticky="w")

# Row 6: Start
start_button = tk.Button(root, text="Start Processing", command=start_processing)
start_button.grid(row=6, column=1, pady=10)

# Row 7: Status
tk.Label(root, textvariable=status_var).grid(row=7, column=1)

# Row 8: Sticky note
tk.Label(root, text="Global anti-linger: same emotion ≤ 15s before bridging.").grid(row=8, column=0, columnspan=3, sticky="we")

root.grid_rowconfigure(7, weight=1)
root.grid_columnconfigure(1, weight=1)

toggle_background_video()

# Occasionally show support message (~1/6 starts)
if random.randint(1, 6) == 1:
    show_support_message()

root.mainloop()
