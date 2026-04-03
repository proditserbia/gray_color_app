# Gray Color App — Image Classifier

A Python desktop tool that automatically classifies images into **Before**
(grayscale) and **After** (colored) categories, with a third **Uncertain**
bucket for borderline cases.

---

## What the tool does

You point it at a folder that contains a mix of grayscale and color photos.
The app scans every supported image, runs a pixel-level RGB channel analysis,
and copies each file into one of three output sub-folders:

| Output sub-folder | Meaning |
|-------------------|---------|
| `output/Before`   | Grayscale images |
| `output/After`    | Color images |
| `output/Uncertain`| Images where the result is borderline |

A CSV log (`classification_log.csv`) is also written to the output folder with
the original path, filename, classification, confidence score, and any error
messages.

---

## Supported image formats

`.jpg` · `.jpeg` · `.png` · `.bmp` · `.webp`

---

## Installation

### Prerequisites

* Python 3.8 or newer
* `pip`

### Install dependencies

```bash
pip install -r requirements.txt
```

Tkinter is part of the Python standard library. On some Linux distributions
you may need to install it separately:

```bash
# Debian / Ubuntu
sudo apt-get install python3-tk
```

---

## How to run

```bash
python main.py
```

The GUI will open.  Use the **Browse…** buttons to choose:

1. **Source folder** — the folder containing your mixed images (sub-folders
   are scanned recursively).
2. **Output folder** — where the three sub-folders and the CSV log will be
   created.

Then click **▶ Start Classification**.  A progress bar shows how many images
have been processed.  When finished, a summary dialog appears.

---

## How classification works

The tool uses **pixel-level RGB channel analysis** rather than relying on
image-mode metadata (e.g. `PIL.Image.mode == 'L'`), which can be unreliable
for images saved with an RGB header but containing only gray pixels.

**Steps:**

1. Load the image with OpenCV as a 3-channel BGR image (`cv2.IMREAD_COLOR`).
2. Compute the per-pixel maximum absolute difference across the R, G, and B
   channels.
3. Take the **mean** of those per-pixel differences — this becomes the
   **color score** (range 0–255).
4. Apply thresholds:

   | Score range | Classification |
   |-------------|----------------|
   | ≤ 5         | **Before** (grayscale) |
   | 5 – 15      | **Uncertain** |
   | ≥ 15        | **After** (colored) |

   The thresholds (`GRAY_THRESHOLD` and `UNCERTAIN_THRESHOLD`) are defined as
   constants at the top of `main.py` and can be tuned without changing any
   other code.

A score near 0 means all three channels are essentially identical (gray); a
high score means the channels diverge significantly (color).

---

## Project structure

```
gray_color_app/
├── main.py           # Application code (GUI + classification logic)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```