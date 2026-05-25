import os
import re
import sys
import glob
import logging
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from dataclasses import dataclass
from typing import Dict, List, Tuple

from moviepy.editor import (
    VideoFileClip, ImageClip, CompositeVideoClip, AudioFileClip, ColorClip
)
from pydub import AudioSegment
import whisper
from transformers import pipeline

import numpy as np

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EmoteSync")

# -----------------------------
# Constants
# -----------------------------
SUPPORTED_EMOTIONS = ["angry", "annoyed", "confused", "happy", "neutral"]
IMG_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
DEFAULT_FPS = 24

# Break up long runs of the same emotion by inserting a neutral frame
DEFAULT_MAX_SAME_EMOTION_RUN = 2  # after N consecutive same-emotion sentences, insert neutral

# Default now: always use top emotion from the model; only go neutral if the
# *model itself* says neutral, or you raise this threshold in the UI.
DEFAULT_CONFIDENCE_THRESHOLD = 0.0

DEFAULT_MIN_SENTENCE_SECONDS = 0.70  # too-short sentences are merged forward to avoid flicker
DEFAULT_PAUSE_BREAK_SECONDS = 0.75   # if a pause longer than this occurs, we break the sentence


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Word:
    start: float
    end: float
    word: str

@dataclass
class EmotionSpan:
    start: float
    end: float
    emotion: str
    confidence: float


# -----------------------------
# Frame Loader (round-robin support)
# -----------------------------
class FrameLibrary:
    """
    Load all frames for each supported emotion.
    Allows round-robin selection so we don't always pick the same file.

    Accepts files named like:
      angry.png, angry_1.png, angry 1.png, angry(1).png, angry1.png  (same for other emotions)
    """
    def __init__(self, folder: str):
        self.folder = folder
        self.frames: Dict[str, List[str]] = {e: [] for e in SUPPORTED_EMOTIONS}

        # Index is tracked per *base* emotion group (e.g. 'neutral'), so that any emotions
        # falling back to the same set of files share a single round-robin counter.
        self._indices: Dict[str, int] = {e: 0 for e in SUPPORTED_EMOTIONS}

        # Map each emotion to the base emotion whose frames it should use
        # (e.g. "angry" -> "angry", but if no angry frames exist, "angry" -> "neutral").
        self._fallback: Dict[str, str] = {e: e for e in SUPPORTED_EMOTIONS}

        self._load()

    def _load(self):
        if not os.path.isdir(self.folder):
            raise FileNotFoundError(f"Emotion images folder not found: {self.folder}")

        # Build regex patterns for each emotion.
        # More forgiving now: supports
        #   happy.png, happy_1.png, happy-1.png, happy 1.png, happy(1).png, happy1.png
        patterns = {
            e: re.compile(rf"^{e}(?:[\s_\-\(\)]*\d+)?$", re.IGNORECASE)
            for e in SUPPORTED_EMOTIONS
        }

        for path in sorted(glob.glob(os.path.join(self.folder, "*"))):
            if not os.path.isfile(path):
                continue
            name, ext = os.path.splitext(os.path.basename(path))
            if ext.lower() not in IMG_EXTS:
                continue
            for emotion, pat in patterns.items():
                if pat.match(name):
                    self.frames[emotion].append(path)
                    break

        # Fallbacks: if an emotion has no frames, fall back to neutral (if available).
        neutral_count = len(self.frames["neutral"])
        if neutral_count == 0:
            raise RuntimeError(
                "No 'neutral' frames found. Please include at least one neutral image "
                "named like 'neutral.png' (alternates like 'neutral_1.png' are supported)."
            )

        for emotion in SUPPORTED_EMOTIONS:
            if len(self.frames[emotion]) == 0 and emotion != "neutral":
                logger.warning(
                    f"No frames found for '{emotion}'. Falling back to 'neutral' frames."
                )
                # Don't copy the list – share the *sequence* with neutral so the round‑robin
                # continues across all emotions that use the neutral set.
                self._fallback[emotion] = "neutral"

        # Log a summary (what each emotion will actually use)
        for emotion in SUPPORTED_EMOTIONS:
            base = self._fallback.get(emotion, emotion)
            count = len(self.frames[base])
            if emotion == base:
                logger.info(f"Loaded {count} frame(s) for '{emotion}'.")
            else:
                logger.info(
                    f"Loaded {count} frame(s) for '{emotion}' "
                    f"(sharing frames with '{base}')."
                )

    def next_for(self, emotion: str) -> str:
        """Return the next frame for an emotion using round-robin selection."""
        if emotion not in SUPPORTED_EMOTIONS:
            emotion = "neutral"

        base = self._fallback.get(emotion, "neutral")
        files = self.frames.get(base) or self.frames["neutral"]

        idx = self._indices.get(base, 0) % len(files)
        self._indices[base] = idx + 1
        return files[idx]


# -----------------------------
# Whisper transcription utilities
# -----------------------------
def transcribe_with_segments(audio_file: str, model: whisper.Whisper) -> List[Segment]:
    """
    Transcribe entire audio and return Whisper's time-aligned segments.
    We'll then stitch these into sentence-like spans.
    """
    logger.info("Transcribing audio with Whisper... (this can take a bit)")
    result = model.transcribe(
        audio_file,
        verbose=False,
        word_timestamps=False,
        condition_on_previous_text=True,
    )
    segs = []
    for s in result.get("segments", []):
        segs.append(Segment(start=float(s["start"]), end=float(s["end"]), text=s["text"].strip()))
    # Fallback if Whisper didn't provide segments
    if not segs:
        text = result.get("text", "").strip()
        if text:
            # Single segment using full duration from pydub
            audio = AudioSegment.from_file(audio_file)
            segs = [Segment(0.0, audio.duration_seconds, text)]
    logger.info(f"Whisper produced {len(segs)} raw segments.")
    return segs




# -----------------------------
# Whisper word-timestamp utilities (more accurate sentence boundary timing)
# -----------------------------
def transcribe_with_words(audio_file: str, model: whisper.Whisper) -> List[Word]:
    """
    Transcribe entire audio and return a flat list of Whisper word timestamps.

    Why: Whisper's *segment* timings are often longer chunks that can contain multiple
    sentences AND the pause between them. If we split that chunk into sentences by
    punctuation and then *guess* timings (char-based), the sentence boundary can drift
    into the pause, causing avatar switches before speech starts.

    Word timestamps let us snap sentence starts/ends to actual spoken word times.
    """
    logger.info("Transcribing audio with Whisper (word timestamps)... (this can take a bit)")

    try:
        result = model.transcribe(
            audio_file,
            verbose=False,
            word_timestamps=True,
            condition_on_previous_text=True,
        )
    except TypeError:
        # Older whisper versions may not accept word_timestamps
        logger.warning(
            "This Whisper build does not support word_timestamps=True; "
            "falling back to segment timings."
        )
        return []
    except Exception as e:
        logger.warning(f"Whisper word-timestamp transcription failed: {e}. Falling back to segment timings.")
        return []

    words: List[Word] = []
    for s in result.get("segments", []) or []:
        for w in (s.get("words") or []):
            try:
                ws = float(w.get("start"))
                we = float(w.get("end"))
                wt = str(w.get("word", "")).replace("\n", " ")
                if wt.strip():
                    words.append(Word(start=ws, end=we, word=wt))
            except Exception:
                continue

    words.sort(key=lambda x: x.start)
    logger.info(f"Whisper produced {len(words)} word timestamp(s).")
    return words


def merge_words_into_sentences(words: List[Word]) -> List[Segment]:
    """
    Build sentence-like spans from word timestamps.

    Split conditions:
      - End-of-sentence punctuation on the *word* (., !, ?, …) (with basic abbreviation guard)
      - A pause between words larger than DEFAULT_PAUSE_BREAK_SECONDS
    """
    if not words:
        return []

    SENT_END_CHARS = ".?!…"
    CLOSING_QUOTES = '"\')]}”’'
    ABBREVS = {
        "e.g.", "i.e.", "etc.", "dr.", "mr.", "mrs.", "ms.",
        "prof.", "sr.", "jr.", "u.s.", "u.s.a.", "vs.",
        "p.m.", "a.m.",
    }

    def ends_sentence(word_text: str) -> bool:
        t = (word_text or "").strip()
        if not t:
            return False

        # Trim closing quotes/brackets so 'hello."' still counts.
        t = t.rstrip(CLOSING_QUOTES)

        # Collect run of sentence-end chars at end (e.g. "?!", "…", "...")
        m = re.search(rf"([{re.escape(SENT_END_CHARS)}]+)$", t)
        if not m:
            return False
        punct = m.group(1)

        # If it ends with dots, guard against abbreviations like "e.g."
        if all(c == "." for c in punct):
            token = t.lower()
            if token in ABBREVS:
                return False
        return True

    merged: List[Segment] = []
    buf: List[str] = []
    start_t = words[0].start
    prev_end = words[0].end

    def flush(end_t: float):
        nonlocal buf, start_t
        txt = "".join(buf).strip()
        if txt:
            merged.append(Segment(start=float(start_t), end=float(end_t), text=txt))
        buf = []

    for w in words:
        # New sentence after a long pause between spoken words
        if buf:
            gap = float(w.start) - float(prev_end)
            if gap > DEFAULT_PAUSE_BREAK_SECONDS:
                flush(prev_end)

        if not buf:
            start_t = w.start

        buf.append(w.word)
        prev_end = w.end

        if ends_sentence(w.word):
            flush(prev_end)

    if buf:
        flush(prev_end)

    # Enforce minimum duration (same as merge_into_sentences) to avoid flicker
    if len(merged) > 1:
        fixed: List[Segment] = []
        i = 0
        while i < len(merged):
            cur = merged[i]
            dur = cur.end - cur.start
            if dur < DEFAULT_MIN_SENTENCE_SECONDS and i + 1 < len(merged):
                nxt = merged[i + 1]
                combined = Segment(
                    start=cur.start,
                    end=nxt.end,
                    text=(cur.text + " " + nxt.text).strip(),
                )
                fixed.append(combined)
                i += 2
            else:
                fixed.append(cur)
                i += 1
        merged = fixed

    logger.info(f"Built {len(merged)} sentence-like spans from {len(words)} word timestamps.")
    return merged


def transcribe_and_build_sentences(audio_file: str, model: whisper.Whisper) -> List[Segment]:
    """
    Prefer word timestamps (best timing), but fall back to segment timings if unavailable.
    """
    words = transcribe_with_words(audio_file, model)
    if words:
        return merge_words_into_sentences(words)

    raw_segments = transcribe_with_segments(audio_file, model)
    return merge_into_sentences(raw_segments)

# -----------------------------
# Sentence splitting (improved)
# -----------------------------
def split_text_into_sentences(text: str) -> List[str]:
    """
    Lightweight, punctuation-based sentence splitter.

    - Splits on '.', '!', '?', '…'
    - Tries NOT to split inside common abbreviations like 'e.g.', 'i.e.', 'p.m.'
    - Keeps surrounding quotes/brackets attached to the sentence.
    """
    SENT_END_CHARS = ".?!…"
    CLOSING_QUOTES = '"\')]}”’'
    ABBREVS = {
        "e.g.", "i.e.", "etc.", "dr.", "mr.", "mrs.", "ms.",
        "prof.", "sr.", "jr.", "u.s.", "u.s.a.", "vs.",
        "p.m.", "a.m.",
    }

    text = (text or "").strip()
    if not text:
        return []

    sentences: List[str] = []
    n = len(text)
    start_idx = 0
    i = 0

    while i < n:
        ch = text[i]
        if ch in SENT_END_CHARS:
            # Skip '.' that is clearly part of an abbreviation like "e.g." or "p.m."
            if ch == "." and i >= 1 and i + 2 < n:
                if text[i - 1].isalpha() and text[i + 1].isalpha() and text[i + 2] == ".":
                    i += 1
                    continue

            # Collect a run of sentence-end chars, e.g. "?!", "..."
            punct_start = i
            j = i
            while j + 1 < n and text[j + 1] in SENT_END_CHARS:
                j += 1
            punct_end = j
            punct_run = text[punct_start: punct_end + 1]

            is_end = True
            if all(c == "." for c in punct_run):
                # Check if the token ending at this '.' is a known abbreviation
                back = punct_end
                while back > start_idx and not text[back - 1].isspace():
                    back -= 1
                token = text[back: punct_end + 1].lower()
                if token in ABBREVS:
                    is_end = False

            if is_end:
                k = punct_end + 1
                # Attach closing quotes/brackets
                while k < n and text[k] in CLOSING_QUOTES:
                    k += 1
                # Cut the sentence
                sent = text[start_idx:k].strip()
                if sent:
                    sentences.append(sent)
                # Skip whitespace before next sentence
                while k < n and text[k].isspace():
                    k += 1
                start_idx = k
                i = k
                continue
        i += 1

    # Tail without final punctuation
    if start_idx < n:
        tail = text[start_idx:].strip()
        if tail:
            sentences.append(tail)

    return sentences


def merge_into_sentences(segments: List[Segment]) -> List[Segment]:
    """
    Combine Whisper segments into sentence-like spans with a good balance
    between speed and natural sentence boundaries.
    """
    if not segments:
        return []

    # Step 1: expand each Whisper segment into smaller sentence-ish segments
    expanded: List[Segment] = []
    for seg in segments:
        pieces = split_text_into_sentences(seg.text)
        if not pieces:
            continue

        # If there's only one piece, keep timings exactly as Whisper gave them
        if len(pieces) == 1:
            expanded.append(Segment(start=seg.start, end=seg.end, text=pieces[0].strip()))
            continue

        # Multiple sentences inside the same Whisper chunk – distribute time
        clean_pieces = [p.strip() for p in pieces if p.strip()]
        if not clean_pieces:
            continue

        total_chars = sum(len(p) for p in clean_pieces) or 1
        seg_dur = max(0.0, seg.end - seg.start)
        cur_start = seg.start

        for idx, sentence_text in enumerate(clean_pieces):
            if idx == len(clean_pieces) - 1:
                cur_end = seg.end  # last one takes remaining time
            else:
                frac = len(sentence_text) / total_chars
                cur_end = cur_start + seg_dur * frac

            expanded.append(
                Segment(
                    start=cur_start,
                    end=cur_end,
                    text=sentence_text,
                )
            )
            cur_start = cur_end

    if not expanded:
        return []

    # Step 2: stitch into final sentence spans
    merged: List[Segment] = []

    def ends_with_punct(t: str) -> bool:
        return bool(re.search(r"[\.!\?…]$", (t or "").strip()))

    buf_text = expanded[0].text
    start = expanded[0].start
    prev_end = expanded[0].end

    for seg in expanded[1:]:
        gap = seg.start - prev_end

        # If there's a clear pause OR the current buffer already looks like
        # a complete sentence, close it and start a new one.
        if buf_text and (gap > DEFAULT_PAUSE_BREAK_SECONDS or ends_with_punct(buf_text)):
            merged.append(Segment(start=start, end=prev_end, text=buf_text.strip()))
            buf_text = seg.text
            start = seg.start
            prev_end = seg.end
            continue

        # Otherwise, keep extending the current sentence
        buf_text = (buf_text + " " + seg.text).strip()
        prev_end = seg.end

    # Final buffer
    if buf_text:
        merged.append(Segment(start=start, end=prev_end, text=buf_text.strip()))

    # Step 3: enforce minimum duration so we don't flicker fast on tiny spans
    if len(merged) > 1:
        fixed: List[Segment] = []
        i = 0
        while i < len(merged):
            cur = merged[i]
            dur = cur.end - cur.start
            if dur < DEFAULT_MIN_SENTENCE_SECONDS and i + 1 < len(merged):
                nxt = merged[i + 1]
                combined = Segment(
                    start=cur.start,
                    end=nxt.end,
                    text=(cur.text + " " + nxt.text).strip(),
                )
                fixed.append(combined)
                i += 2
            else:
                fixed.append(cur)
                i += 1
        merged = fixed

    logger.info(
        f"Built {len(merged)} sentence-like spans from {len(segments)} Whisper segments "
        f"(expanded to {len(expanded)} sub-segments)."
    )
    return merged


# -----------------------------
# Emotion classification
# -----------------------------
def build_emotion_classifier():
    """
    Returns a Hugging Face pipeline for emotion classification.

    Uses j-hartmann/emotion-english-distilroberta-base, which is downloaded
    once and then cached locally, so after the first run it's fully local.
    """
    logger.info("Loading emotion classifier (Hugging Face pipeline)...")
    clf = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        framework="pt",
    )

    # Log the label space to help debug cases where a different model loads
    try:
        labels = list(getattr(clf.model.config, "id2label", {}).values())
        if labels:
            logger.info(f"Emotion model labels: {labels}")
    except Exception:
        pass

    return clf


def map_emotion(model_label: str) -> str:
    """
    Map model labels to our five emotions.

    Handles:
      - Hartmann emotion labels: anger, disgust, fear, joy, neutral, sadness, surprise
      - Generic sentiment labels: POSITIVE / NEGATIVE
      - Common 'LABEL_0' / 'LABEL_1' style sentiment models
      - Other variants via substring heuristics
    """
    lbl = (model_label or "").lower().strip()

    # ---- Direct mappings (Hartmann model) ----
    if lbl == "anger":
        return "angry"
    if lbl == "disgust":
        return "annoyed"
    if lbl == "fear":
        return "confused"
    if lbl == "joy":
        return "happy"
    if lbl == "neutral":
        return "neutral"
    if lbl == "sadness":
        # You could also map sadness -> annoyed or confused if you want more variety.
        return "neutral"
    if lbl == "surprise":
        return "confused"

    # ---- Sentiment-style labels ----
    if lbl == "positive" or lbl == "label_1":
        return "happy"
    if lbl == "negative" or lbl == "label_0":
        # Mildly negative → annoyed, strongly negative → angry; we don't have the score here,
        # so err on the more expressive side.
        return "angry"

    # ---- Substring heuristics for other models ----
    if "anger" in lbl:
        return "angry"
    if "angry" in lbl:
        return "angry"
    if "disgust" in lbl:
        return "annoyed"
    if "annoy" in lbl:
        return "annoyed"
    if "fear" in lbl or "anxiety" in lbl:
        return "confused"
    if "joy" in lbl or "happy" in lbl or "happiness" in lbl or "positive" in lbl:
        return "happy"
    if "surprise" in lbl or "astonish" in lbl:
        return "confused"
    if "sad" in lbl or "depress" in lbl:
        return "neutral"

    # Fallback – truly unknown label
    logger.debug(f"Unknown emotion label '{model_label}', falling back to neutral.")
    return "neutral"


def classify_sentence_emotion(text: str, clf, confidence_threshold: float) -> Tuple[str, float]:
    """
    Run the classifier and return (mapped_emotion, confidence_of_top_label).

    Uses truncation=True to avoid running past the model's max length.
    """
    t = (text or "").strip()
    if not t:
        return ("neutral", 1.0)

    try:
        outputs = clf(t, truncation=True)
    except TypeError:
        # Some very old transformers versions don't accept truncation here.
        outputs = clf(t)

    # Handle both pipeline output formats:
    #   Older transformers: [[{label, score}, ...]] (nested - one list per input)
    #   Newer transformers: [{label, score}, ...]   (flat for single-string input)
    if outputs and isinstance(outputs[0], list):
        result = list(outputs[0])
    else:
        result = list(outputs)

    # Sort by score, highest first
    result.sort(key=lambda d: d["score"], reverse=True)
    top = result[0]
    mapped = map_emotion(top["label"])
    conf = float(top["score"])

    if conf < confidence_threshold:
        return ("neutral", conf)
    return (mapped, conf)


def classify_sentence_emotion_force_non_neutral(text: str, clf) -> Tuple[str, float]:
    """
    Re-run classification but explicitly try to pick a NON-NEUTRAL emotion.

    Used by the "no back-to-back neutral" pass:
      - Look at all labels from the model.
      - Map them to our 5 emotions.
      - Return the highest-scoring one that is not 'neutral', if any.
      - If every label maps to 'neutral', fall back to the normal top label.
    """
    t = (text or "").strip()
    if not t:
        return ("neutral", 0.0)

    try:
        outputs = clf(t, truncation=True)
    except TypeError:
        outputs = clf(t)

    # Handle both pipeline output formats (see classify_sentence_emotion).
    if outputs and isinstance(outputs[0], list):
        result = list(outputs[0])
    else:
        result = list(outputs)

    result.sort(key=lambda d: d["score"], reverse=True)

    # Try to find the best non-neutral mapped emotion
    for cand in result:
        e = map_emotion(cand["label"])
        if e != "neutral":
            return (e, float(cand["score"]))

    # Everything mapped to neutral; fall back to top
    top = result[0]
    return (map_emotion(top["label"]), float(top["score"]))


def build_emotion_timeline(sentences: List[Segment], clf, confidence_threshold: float) -> List[EmotionSpan]:
    timeline: List[EmotionSpan] = []
    for s in sentences:
        emotion, conf = classify_sentence_emotion(s.text, clf, confidence_threshold)
        timeline.append(EmotionSpan(start=s.start, end=s.end, emotion=emotion, confidence=conf))

        # Helpful debug output so you can confirm it's not "all neutral" anymore
        preview = s.text.replace("\n", " ")[:100]
        logger.info(
            f"[EMOTION] {s.start:6.2f}-{s.end:6.2f}s -> {emotion:8s} (conf={conf:0.3f}) :: {preview}"
        )

    return timeline


def break_up_long_runs(timeline: List[EmotionSpan], max_run: int) -> List[EmotionSpan]:
    """
    Insert neutral in long runs:
      If we see N consecutive sentences with the same non-neutral emotion,
      the (N+1)-th sentence is forced to 'neutral' (but keeps its original timing).
    """
    if max_run <= 0 or not timeline:
        return timeline

    out: List[EmotionSpan] = []
    run_emotion = None
    run_len = 0

    for span in timeline:
        e = span.emotion
        if e != "neutral" and e == run_emotion:
            run_len += 1
        else:
            run_emotion = e if e != "neutral" else None
            run_len = 1 if e != "neutral" else 0

        if run_emotion and run_len > max_run:
            # force a neutral breaker
            out.append(EmotionSpan(start=span.start, end=span.end, emotion="neutral", confidence=span.confidence))
            # reset run (we just inserted neutral)
            run_emotion = None
            run_len = 0
        else:
            out.append(span)

    return out


def avoid_back_to_back_neutral(
    sentences: List[Segment],
    timeline: List[EmotionSpan],
    clf
) -> List[EmotionSpan]:
    """
    Post-process timeline to avoid 'neutral' immediately followed by 'neutral'.

    For every second+ neutral in a row, we re-run classification for that
    sentence and pick the best non-neutral emotion (if one exists).
    """
    if not timeline:
        return timeline

    prev_emotion = None
    for i, span in enumerate(timeline):
        if span.emotion == "neutral" and prev_emotion == "neutral":
            # Try to force a non-neutral emotion for variety
            alt_emotion, alt_conf = classify_sentence_emotion_force_non_neutral(
                sentences[i].text,
                clf,
            )
            if alt_emotion != "neutral":
                logger.info(
                    f"[VARIETY] {span.start:6.2f}-{span.end:6.2f}s neutral->"
                    f"{alt_emotion} (conf={alt_conf:0.3f})"
                )
                timeline[i] = EmotionSpan(
                    start=span.start,
                    end=span.end,
                    emotion=alt_emotion,
                    confidence=alt_conf,
                )
                prev_emotion = alt_emotion
                continue

        prev_emotion = span.emotion

    return timeline


# -----------------------------
# Image fitting helper (scale + trim from top/right)
# -----------------------------
def fit_image_to_background_cover_bottom_left(img_clip: ImageClip, bg_w: int, bg_h: int) -> ImageClip:
    """
    Scale the image to fully cover the background (like CSS 'cover'),
    then trim overflow from the TOP and RIGHT sides (anchored to bottom-left).
    The resulting clip has exactly the background's size.
    """
    try:
        iw, ih = img_clip.size
        if iw <= 0 or ih <= 0 or bg_w <= 0 or bg_h <= 0:
            # Fallback: no-op if sizes are invalid
            return img_clip

        # Scale to ensure the image covers the entire background
        scale = max(bg_w / iw, bg_h / ih)
        resized = img_clip.resize(scale)

        rw, rh = resized.size

        # Compute crop rectangle anchored to bottom-left:
        #  - keep x from 0 to bg_w  (crop overflow on the RIGHT)
        #  - keep y from (rh - bg_h) to rh  (crop overflow on the TOP)
        x1 = 0
        y1 = max(0, rh - bg_h)
        x2 = min(rw, x1 + bg_w)
        y2 = min(rh, y1 + bg_h)

        # Safety correction in case of rounding
        if x2 - x1 < bg_w and rw >= bg_w:
            x1 = max(0, rw - bg_w)
            x2 = rw
        if y2 - y1 < bg_h and rh >= bg_h:
            y1 = max(0, rh - bg_h)
            y2 = rh

        fitted = resized.crop(x1=x1, y1=y1, x2=x2, y2=y2)

        # Ensure final size matches exactly bg dimensions
        fw, fh = fitted.size
        if fw != bg_w or fh != bg_h:
            # Last-ditch resize without changing aspect beyond one pixel rounding
            fitted = fitted.resize(newsize=(bg_w, bg_h))

        # Anchor at bottom-left in the composite
        return fitted.set_position(("left", "bottom"))
    except Exception as e:
        logger.warning(f"fit_image_to_background_cover_bottom_left failed: {e}. Falling back to centered fit.")
        # Fallback: center-fit to height
        return img_clip.resize(height=bg_h).set_position(("center", "center"))


# -----------------------------
# Timeline gap-filling helper (hold last frame across pauses)
# -----------------------------
def fill_timeline_gaps_with_hold(
    timeline: List[EmotionSpan],
    total_duration: float,
    epsilon: float = 1e-3,
) -> List[EmotionSpan]:
    """
    Ensure there are no small gaps between emotion spans by "holding" the last
    frame until the next one starts. Also extends the final span to the end of
    the audio so the avatar never disappears while audio is playing.
    """
    if not timeline:
        return timeline

    # Make sure spans are in chronological order
    spans = sorted(timeline, key=lambda s: s.start)
    filled: List[EmotionSpan] = []

    for idx, span in enumerate(spans):
        # Clamp to the valid time range first
        start = max(0.0, span.start)
        end = min(span.end, total_duration)

        if idx + 1 < len(spans):
            next_span = spans[idx + 1]
            # If there's a forward gap, stretch this span to cover it
            if end + epsilon < next_span.start:
                end = min(next_span.start, total_duration)
        else:
            # Last span: extend to cover any trailing audio
            if end + epsilon < total_duration:
                end = total_duration

        # Guard against negative or zero-length spans due to bad timings
        if end < start:
            end = start

        filled.append(
            EmotionSpan(
                start=start,
                end=end,
                emotion=span.emotion,
                confidence=span.confidence,
            )
        )

    return filled


# -----------------------------
# Video composition
# -----------------------------
def compose_video(
    audio_file: str,
    background_video_file: str,
    use_background: bool,
    frames: FrameLibrary,
    timeline: List[EmotionSpan],
    fps: int = DEFAULT_FPS,
    bounce: bool = False,
) -> Tuple[CompositeVideoClip, float]:
    """
    Create the final CompositeVideoClip and return (clip, total_duration).
    """
    # Load audio
    try:
        audio_clip = AudioFileClip(audio_file)
    except Exception as e:
        raise RuntimeError(f"Could not load audio file '{audio_file}'. {e}")

    total_duration = audio_clip.duration

    # NEW: fill small gaps so the avatar holds across pauses
    timeline = fill_timeline_gaps_with_hold(timeline, total_duration)

    # Prepare background
    if use_background:
        if not background_video_file:
            raise RuntimeError("Background video file not specified.")
        try:
            background_video = VideoFileClip(background_video_file)
            fps = int(background_video.fps) if getattr(background_video, "fps", None) else fps
        except Exception as e:
            raise RuntimeError(f"Could not load video file '{background_video_file}'. {e}")

        if background_video.duration < total_duration:
            background_video = background_video.loop(duration=total_duration)
        elif background_video.duration > total_duration:
            background_video = background_video.subclip(0, total_duration)
    else:
        # If we don't have a background, make a simple black canvas.
        width, height = 1280, 720
        background_video = ColorClip(size=(width, height), color=(0, 0, 0)).set_duration(total_duration)
        background_video.fps = fps

    # Bounce effect that NEVER goes below 1.0 (so we never expose blank edges)
    def bounce_effect_ge1(t: float) -> float:
        d = 0.5
        if t < d:
            # Positive-only oscillation, decaying, min 1.0
            return 1.0 + 0.05 * (np.sin(2 * np.pi * 2 * t / d) ** 2) * np.exp(-4 * t / d)
        return 1.0

    clips = [background_video]

    bg_w = int(background_video.w)
    bg_h = int(background_video.h)

    for span in timeline:
        frame_path = frames.next_for(span.emotion)
        if not os.path.isfile(frame_path):
            logger.warning(f"Missing frame '{frame_path}', falling back to neutral.")
            frame_path = frames.next_for("neutral")

        base_img = (
            ImageClip(frame_path)
            .set_start(span.start)
            .set_duration(max(0.001, span.end - span.start))
        )

        # Fit to background using "scale + trim from top/right" (anchored bottom-left)
        img_clip = fit_image_to_background_cover_bottom_left(base_img, bg_w, bg_h)

        # Apply (optional) bounce AFTER fitting, ensuring it never shrinks below coverage
        if bounce:
            img_clip = img_clip.resize(lambda t: bounce_effect_ge1(t)).set_position(("left", "bottom"))

        clips.append(img_clip)

    final = CompositeVideoClip(clips).set_duration(total_duration).set_audio(audio_clip)
    final.fps = fps
    return final, total_duration


# -----------------------------
# Main processing (thread target)
# -----------------------------
def process_video(
    audio_file: str,
    background_video_file: str,
    emotion_folder: str,
    use_background: bool,
    confidence_threshold: float,
    max_run: int,
    bounce: bool,
):
    try:
        update_status("Loading models and files...")

        # Load frames (round-robin aware)
        frame_lib = FrameLibrary(emotion_folder)

        # Emotion classifier (local transformer model)
        clf = build_emotion_classifier()

        # Whisper
        whisper_model = whisper.load_model("base")

        update_status("Transcribing & building sentence timeline...")
        sentences = transcribe_and_build_sentences(audio_file, whisper_model)

        update_status("Classifying emotions...")
        timeline = build_emotion_timeline(sentences, clf, confidence_threshold)

        # Break up long runs of same emotion using neutral as a spacer
        timeline = break_up_long_runs(timeline, max_run=max_run)

        # Enforce "no back-to-back neutral" to increase variety
        timeline = avoid_back_to_back_neutral(sentences, timeline, clf)

        # Compose the video
        update_status("Composing final video...")
        final_video, total_dur = compose_video(
            audio_file=audio_file,
            background_video_file=background_video_file,
            use_background=use_background,
            frames=frame_lib,
            timeline=timeline,
            fps=DEFAULT_FPS,
            bounce=bounce,
        )

        # Export
        update_status("Exporting video...")
        output_filename = "output_video.mp4"
        final_video.write_videofile(
            output_filename,
            fps=final_video.fps,
            codec="libx264",
            audio_codec="aac",
        )

        update_status("Processing completed!")
        start_button.config(state="normal")
        try:
            messagebox.showinfo("Success", f"Video processing completed.\nOutput: {output_filename}")
        except Exception:
            pass
        logger.info(f"Done. Output: {output_filename}")

    except Exception as e:
        logger.exception("An error occurred during processing.")
        update_status("An error occurred.")
        start_button.config(state="normal")
        try:
            messagebox.showerror("Error", f"Processing error:\n{e}")
        except Exception:
            pass
        print(f"Error: {e}", file=sys.stderr)


# -----------------------------
# Tkinter UI
# -----------------------------
def select_audio_file():
    audio_file = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("Audio / Video", "*.*")],
    )
    audio_file_var.set(audio_file)


def select_background_video():
    background_file = filedialog.askopenfilename(
        title="Select Background Video",
        filetypes=[("Video", "*.*")],
    )
    background_file_var.set(background_file)


def select_emotion_images_folder():
    emotion_folder = filedialog.askdirectory(title="Select Folder Containing Emotion Frames")
    emotion_folder_var.set(emotion_folder)


def start_processing():
    audio_file = audio_file_var.get()
    use_bg = use_background_var.get()
    background_video = background_file_var.get() if use_bg else ""
    emotion_folder = emotion_folder_var.get()

    if not audio_file or not emotion_folder:
        try:
            messagebox.showerror("Error", "Please select an audio file and an emotion frames folder.")
        except Exception:
            pass
        update_status("Please select an audio file and an emotion frames folder.")
        return

    if use_bg and not background_video:
        try:
            messagebox.showerror("Error", "Please select a background video or uncheck 'Use Background Video'.")
        except Exception:
            pass
        update_status("Please select a background video or uncheck 'Use Background Video'.")
        return

    # Disable start to prevent multiple kicks
    start_button.config(state="disabled")

    conf = float(confidence_threshold_var.get())
    max_run = int(max_run_var.get())
    bounce = bool(bounce_var.get())

    threading.Thread(
        target=process_video,
        args=(audio_file, background_video, emotion_folder, use_bg, conf, max_run, bounce),
        daemon=True,
    ).start()


def update_status(message: str):
    status_var.set(message)
    logger.info(message)


def toggle_background_video():
    if use_background_var.get():
        background_file_entry.config(state="normal")
        background_browse_button.config(state="normal")
    else:
        background_file_entry.config(state="disabled")
        background_browse_button.config(state="disabled")
        background_file_var.set("")


def show_about():
    text = (
        "EmoteSync v2\n\n"
        "• Uses Whisper to transcribe audio and switch frames at sentence starts.\n"
        "• Uses a local transformer model to classify sentences into\n"
        "  {angry, annoyed, confused, happy, neutral}.\n"
        "• Loads multiple frames per emotion (angry, angry_1, angry_2, ...), and rotates round‑robin.\n"
        "• Breaks up long runs of the same emotion by inserting neutral frames.\n"
        "• Avoids back-to-back neutral frames for more visual variety.\n"
        "• Image fit: scale to cover, trim from TOP/RIGHT (anchored bottom‑left) to match background exactly.\n"
        "• Holds the last avatar frame across short pauses so it doesn't drop out between sentences.\n"
    )
    try:
        messagebox.showinfo("About", text)
    except Exception:
        logger.info(text)


# Build UI
root = tk.Tk()
root.title("EmoteSync v2")

audio_file_var = tk.StringVar()
background_file_var = tk.StringVar()
emotion_folder_var = tk.StringVar()
status_var = tk.StringVar()
bounce_var = tk.BooleanVar(value=False)
use_background_var = tk.BooleanVar(value=True)
confidence_threshold_var = tk.DoubleVar(value=DEFAULT_CONFIDENCE_THRESHOLD)
max_run_var = tk.IntVar(value=DEFAULT_MAX_SAME_EMOTION_RUN)

# Row 0: Title + About
title_lbl = tk.Label(root, text="EmoteSync v2 — sentence-aligned emotion frames", font=("TkDefaultFont", 12, "bold"))
title_lbl.grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=(8, 4))
about_btn = tk.Button(root, text="About", command=show_about)
about_btn.grid(row=0, column=2, sticky="e", padx=5, pady=(8, 4))

# Row 1: Audio
tk.Label(root, text="Audio File:").grid(row=1, column=0, sticky="e", padx=5, pady=4)
tk.Entry(root, textvariable=audio_file_var, width=54).grid(row=1, column=1, padx=5, pady=4)
tk.Button(root, text="Browse...", command=select_audio_file).grid(row=1, column=2, padx=5, pady=4)

# Row 2: Emotion Frames Folder
tk.Label(root, text="Emotion Frames Folder:").grid(row=2, column=0, sticky="e", padx=5, pady=4)
tk.Entry(root, textvariable=emotion_folder_var, width=54).grid(row=2, column=1, padx=5, pady=4)
tk.Button(root, text="Browse...", command=select_emotion_images_folder).grid(row=2, column=2, padx=5, pady=4)

# Row 3: Background Video
tk.Label(root, text="Background Video:").grid(row=3, column=0, sticky="e", padx=5, pady=4)
background_file_entry = tk.Entry(root, textvariable=background_file_var, width=54)
background_file_entry.grid(row=3, column=1, padx=5, pady=4)
background_browse_button = tk.Button(root, text="Browse...", command=select_background_video)
background_browse_button.grid(row=3, column=2, padx=5, pady=4)

# Row 4: Options
tk.Checkbutton(root, text="Use Background Video", variable=use_background_var, command=toggle_background_video).grid(
    row=4, column=1, sticky="w", padx=5, pady=(6, 2)
)
tk.Checkbutton(root, text="Enable Subtle Bounce", variable=bounce_var).grid(
    row=5, column=1, sticky="w", padx=5, pady=2
)

# Row 6: Controls for quality
tk.Label(root, text="Neutral Fallback Threshold (0–1):").grid(row=6, column=0, sticky="e", padx=5, pady=4)
tk.Spinbox(
    root,
    from_=0.0,
    to=1.0,
    increment=0.05,
    textvariable=confidence_threshold_var,
    width=6,
).grid(row=6, column=1, sticky="w", padx=5, pady=4)

tk.Label(root, text="Max Same‑Emotion Run:").grid(row=7, column=0, sticky="e", padx=5, pady=4)
tk.Spinbox(
    root,
    from_=0,
    to=5,
    increment=1,
    textvariable=max_run_var,
    width=6,
).grid(row=7, column=1, sticky="w", padx=5, pady=4)

# Row 8: Start
start_button = tk.Button(root, text="Start Processing", command=start_processing)
start_button.grid(row=8, column=1, pady=10)

# Row 9: Status
tk.Label(root, textvariable=status_var).grid(row=9, column=0, columnspan=3, sticky="we", padx=5, pady=(2, 8))

root.grid_columnconfigure(1, weight=1)

# Initialize background fields
toggle_background_video()

# Start UI loop
if __name__ == "__main__":
    root.mainloop()
