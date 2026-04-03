"""
Gray Color App — Image Classification Tool
===========================================
Classifies images in a source folder into:
  - Before  (grayscale)
  - After   (colored)
  - Uncertain

Uses pixel-level RGB channel analysis to decide, not image-mode metadata.
"""

import csv
import os
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Maximum mean absolute difference between any two channels to still call the
# image "grayscale".  Range: [0, 255].  Tweak here to adjust sensitivity.
GRAY_THRESHOLD: float = 5.0

# If the score falls between GRAY_THRESHOLD and UNCERTAIN_THRESHOLD the image
# is put in the "Uncertain" bucket.  Above UNCERTAIN_THRESHOLD → colored.
UNCERTAIN_THRESHOLD: float = 15.0

# Output sub-folder names
FOLDER_BEFORE = "Before"
FOLDER_AFTER = "After"
FOLDER_UNCERTAIN = "Uncertain"

# CSV log filename
LOG_FILENAME = "classification_log.csv"


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def compute_color_score(image_bgr: np.ndarray) -> float:
    """Return a score in [0, 255] that measures how 'colorful' an image is.

    The score is the mean of the per-pixel maximum absolute difference across
    the three BGR channels.  A purely grayscale image gives 0; a saturated
    color image gives a high value.

    Parameters
    ----------
    image_bgr:
        NumPy array of shape (H, W, 3) in BGR uint8 format (as returned by
        cv2.imread).

    Returns
    -------
    float
        Color score.  Lower → more gray, higher → more color.
    """
    # Convert to float32 to avoid uint8 wrap-around during subtraction
    img = image_bgr.astype(np.float32)

    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Per-pixel max absolute difference across channels
    diff_rg = np.abs(r - g)
    diff_rb = np.abs(r - b)
    diff_gb = np.abs(g - b)

    max_diff = np.maximum(diff_rg, np.maximum(diff_rb, diff_gb))
    return float(np.mean(max_diff))


def classify_image(filepath: str) -> Tuple[str, float, str]:
    """Classify a single image file.

    Returns
    -------
    classification : str
        One of ``"Before"``, ``"After"``, or ``"Uncertain"``.
    score : float
        The color score (0 = perfectly gray, higher = more color).
    error : str
        Empty string on success; error message on failure.
    """
    try:
        # cv2.IMREAD_COLOR always returns a 3-channel BGR image, even for
        # images stored with a single channel or a palette — this is exactly
        # what we want so the channel-difference analysis is always valid.
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)

        if img is None:
            return "Error", 0.0, f"cv2.imread returned None for '{filepath}'"

        score = compute_color_score(img)

        if score <= GRAY_THRESHOLD:
            classification = FOLDER_BEFORE       # grayscale
        elif score >= UNCERTAIN_THRESHOLD:
            classification = FOLDER_AFTER        # colored
        else:
            classification = FOLDER_UNCERTAIN    # borderline

        return classification, score, ""

    except (cv2.error, IOError, ValueError, MemoryError) as exc:
        return "Error", 0.0, str(exc)
    except Exception as exc:  # noqa: BLE001 — catch-all for unexpected failures
        return "Error", 0.0, f"Unexpected error: {exc}"


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def ensure_output_dirs(output_root: str) -> None:
    """Create the three output sub-directories if they do not exist."""
    for name in (FOLDER_BEFORE, FOLDER_AFTER, FOLDER_UNCERTAIN):
        os.makedirs(os.path.join(output_root, name), exist_ok=True)


def safe_destination(dest_dir: str, filename: str) -> str:
    """Return a destination path that does not overwrite an existing file.

    If *filename* already exists in *dest_dir*, appends ``_1``, ``_2`` … to
    the stem until a free name is found.
    """
    dest = os.path.join(dest_dir, filename)
    if not os.path.exists(dest):
        return dest

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        dest = os.path.join(dest_dir, new_name)
        if not os.path.exists(dest):
            return dest
        counter += 1


def copy_file(src: str, dest: str) -> None:
    """Copy *src* to *dest*, preserving metadata."""
    shutil.copy2(src, dest)


def find_images(source_dir: str) -> list:
    """Recursively walk *source_dir* and return all supported image paths."""
    images = []
    for root, _dirs, files in os.walk(source_dir):
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(os.path.join(root, fname))
    return images


# ---------------------------------------------------------------------------
# Core processing function (runs in a worker thread)
# ---------------------------------------------------------------------------

def process_images(
    source_dir: str,
    output_dir: str,
    progress_callback,   # callable(processed: int, total: int, counts: dict)
    done_callback,       # callable(counts: dict, log_path: str)
):
    """Find, classify, copy images and write a CSV log.

    Designed to run inside a background thread so the GUI stays responsive.
    """
    image_paths = find_images(source_dir)
    total = len(image_paths)

    counts = {
        "total": total,
        "processed": 0,
        FOLDER_BEFORE: 0,
        FOLDER_AFTER: 0,
        FOLDER_UNCERTAIN: 0,
        "error": 0,
    }

    ensure_output_dirs(output_dir)

    log_path = os.path.join(output_dir, LOG_FILENAME)
    log_rows = []

    for filepath in image_paths:
        filename = os.path.basename(filepath)
        classification, score, error = classify_image(filepath)

        if classification == "Error":
            counts["error"] += 1
            dest_path = ""
        else:
            dest_dir = os.path.join(output_dir, classification)
            dest_path = safe_destination(dest_dir, filename)
            try:
                copy_file(filepath, dest_path)
                counts[classification] += 1
            except (shutil.Error, OSError, PermissionError) as exc:
                classification = "Error"
                error = str(exc)
                counts["error"] += 1
                dest_path = ""

        log_rows.append(
            {
                "original_path": filepath,
                "filename": filename,
                "classification": classification,
                "score": f"{score:.4f}",
                "error": error,
            }
        )

        counts["processed"] += 1
        progress_callback(counts["processed"], total, counts)

    # Write CSV log
    try:
        with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["original_path", "filename", "classification", "score", "error"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)
    except (OSError, PermissionError, csv.Error) as exc:
        log_path = f"(log write failed: {exc})"

    done_callback(counts, log_path)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("Gray Color App — Image Classifier")
        self.resizable(False, False)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # ── Folder selection ────────────────────────────────────────────
        frame_folders = ttk.LabelFrame(self, text="Folders", padding=8)
        frame_folders.grid(row=0, column=0, sticky="ew", **pad)

        ttk.Label(frame_folders, text="Source folder:").grid(
            row=0, column=0, sticky="w"
        )
        self.source_var = tk.StringVar()
        ttk.Entry(frame_folders, textvariable=self.source_var, width=50).grid(
            row=0, column=1, padx=5
        )
        ttk.Button(frame_folders, text="Browse…", command=self._browse_source).grid(
            row=0, column=2
        )

        ttk.Label(frame_folders, text="Output folder:").grid(
            row=1, column=0, sticky="w"
        )
        self.output_var = tk.StringVar()
        ttk.Entry(frame_folders, textvariable=self.output_var, width=50).grid(
            row=1, column=1, padx=5
        )
        ttk.Button(frame_folders, text="Browse…", command=self._browse_output).grid(
            row=1, column=2
        )

        # ── Progress ────────────────────────────────────────────────────
        frame_progress = ttk.LabelFrame(self, text="Progress", padding=8)
        frame_progress.grid(row=1, column=0, sticky="ew", **pad)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            frame_progress,
            variable=self.progress_var,
            maximum=100,
            length=480,
        )
        self.progress_bar.grid(row=0, column=0, columnspan=3, pady=4)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frame_progress, textvariable=self.status_var).grid(
            row=1, column=0, columnspan=3
        )

        # ── Statistics ──────────────────────────────────────────────────
        frame_stats = ttk.LabelFrame(self, text="Statistics", padding=8)
        frame_stats.grid(row=2, column=0, sticky="ew", **pad)

        stat_labels = [
            ("Total files found:", "total"),
            ("Processed:", "processed"),
            ("Before (grayscale):", FOLDER_BEFORE),
            ("After (colored):", FOLDER_AFTER),
            ("Uncertain:", FOLDER_UNCERTAIN),
            ("Errors:", "error"),
        ]

        self._stat_vars = {}
        for idx, (label_text, key) in enumerate(stat_labels):
            ttk.Label(frame_stats, text=label_text).grid(
                row=idx, column=0, sticky="w"
            )
            var = tk.StringVar(value="—")
            ttk.Label(frame_stats, textvariable=var, width=10, anchor="e").grid(
                row=idx, column=1, sticky="e"
            )
            self._stat_vars[key] = var

        # ── Action buttons ──────────────────────────────────────────────
        frame_buttons = ttk.Frame(self, padding=8)
        frame_buttons.grid(row=3, column=0, sticky="ew", **pad)

        self.start_btn = ttk.Button(
            frame_buttons, text="▶  Start Classification", command=self._start
        )
        self.start_btn.pack(side="left", padx=4)

        ttk.Button(frame_buttons, text="✕  Quit", command=self.destroy).pack(
            side="right", padx=4
        )

    # ------------------------------------------------------------------
    # Folder browsing
    # ------------------------------------------------------------------

    def _browse_source(self):
        folder = filedialog.askdirectory(title="Select source folder")
        if folder:
            self.source_var.set(folder)

    def _browse_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_var.set(folder)

    # ------------------------------------------------------------------
    # Start / stop classification
    # ------------------------------------------------------------------

    def _start(self):
        source = self.source_var.get().strip()
        output = self.output_var.get().strip()

        if not source:
            messagebox.showwarning("Missing input", "Please select a source folder.")
            return
        if not os.path.isdir(source):
            messagebox.showerror("Invalid folder", f"Source folder does not exist:\n{source}")
            return
        if not output:
            messagebox.showwarning("Missing input", "Please select an output folder.")
            return

        # Reset stats display
        for var in self._stat_vars.values():
            var.set("—")
        self.progress_var.set(0)
        self.status_var.set("Scanning…")
        self.start_btn.config(state="disabled")

        # Run in background thread
        thread = threading.Thread(
            target=process_images,
            args=(source, output, self._on_progress, self._on_done),
            daemon=True,
        )
        thread.start()

    # ------------------------------------------------------------------
    # Thread callbacks — must schedule GUI updates via after()
    # ------------------------------------------------------------------

    def _on_progress(self, processed: int, total: int, counts: dict):
        """Called from the worker thread after each image is processed."""
        self.after(0, self._update_progress, processed, total, counts)

    def _update_progress(self, processed: int, total: int, counts: dict):
        """Update progress bar and stats (runs on main thread)."""
        pct = (processed / total * 100) if total > 0 else 0
        self.progress_var.set(pct)
        self.status_var.set(f"Processing… {processed} / {total}")

        self._stat_vars["total"].set(str(total))
        self._stat_vars["processed"].set(str(processed))
        self._stat_vars[FOLDER_BEFORE].set(str(counts[FOLDER_BEFORE]))
        self._stat_vars[FOLDER_AFTER].set(str(counts[FOLDER_AFTER]))
        self._stat_vars[FOLDER_UNCERTAIN].set(str(counts[FOLDER_UNCERTAIN]))
        self._stat_vars["error"].set(str(counts["error"]))

    def _on_done(self, counts: dict, log_path: str):
        """Called from the worker thread when processing finishes."""
        self.after(0, self._update_done, counts, log_path)

    def _update_done(self, counts: dict, log_path: str):
        """Show final summary (runs on main thread)."""
        self.progress_var.set(100)
        self.status_var.set("Done ✓")
        self.start_btn.config(state="normal")

        # Final stat update
        self._stat_vars["total"].set(str(counts["total"]))
        self._stat_vars["processed"].set(str(counts["processed"]))
        self._stat_vars[FOLDER_BEFORE].set(str(counts[FOLDER_BEFORE]))
        self._stat_vars[FOLDER_AFTER].set(str(counts[FOLDER_AFTER]))
        self._stat_vars[FOLDER_UNCERTAIN].set(str(counts[FOLDER_UNCERTAIN]))
        self._stat_vars["error"].set(str(counts["error"]))

        summary = (
            f"Classification complete!\n\n"
            f"  Total files found : {counts['total']}\n"
            f"  Before (grayscale): {counts[FOLDER_BEFORE]}\n"
            f"  After  (colored)  : {counts[FOLDER_AFTER]}\n"
            f"  Uncertain         : {counts[FOLDER_UNCERTAIN]}\n"
            f"  Errors            : {counts['error']}\n\n"
            f"CSV log saved to:\n{log_path}"
        )
        messagebox.showinfo("Done", summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
