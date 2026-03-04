# Polish ID Photo Generator -- AI-Assisted MVP

This repository contains a small experimental MVP that converts a normal
photo (e.g. a selfie) into a **JPG compliant with Polish ID / passport
technical requirements**.

The goal of the project was to test how quickly a working product
prototype can be built using **AI-assisted development tools (Cursor)**
combined with classical computer vision techniques.

The MVP focuses on **technical compliance**, not beautification or
generative image modification.

------------------------------------------------------------------------

# Problem

Online applications for identity documents require photos that satisfy
strict technical constraints:

-   specific face geometry
-   fixed 35×45 aspect ratio
-   minimum resolution
-   neutral lighting
-   plain white background

Many users attempting to upload phone photos fail these constraints and
have their applications rejected.

The insight behind this MVP was simple:

> Identity systems do not care how the photo was taken --- only whether
> it satisfies deterministic technical parameters.

------------------------------------------------------------------------

# What the MVP does

**Input**\
A normal photo (selfie or camera photo).

**Output**\
A final **JPG file compliant with Polish ID photo requirements**,
including:

-   35×45 aspect ratio\
-   correct face positioning and scale\
-   white background\
-   neutral lighting\
-   minimum resolution\
-   maximum file size ≤ 2.5 MB

The system intentionally avoids:

-   generative AI
-   facial retouching
-   altering facial features

The goal is **technical compliance**, not aesthetic improvement.

------------------------------------------------------------------------

# User flow

1.  User uploads a photo (JPG / PNG)
2.  Backend processes the image using a deterministic pipeline
3.  User sees a protected preview
4.  Payment step (stub in MVP)
5.  Final JPG becomes available for download

The product intentionally uses a **single-page flow** with minimal
friction.

------------------------------------------------------------------------

# System architecture

**Frontend**\
Next.js application with a minimal upload interface and preview.

**Backend**\
FastAPI service performing image processing and validation.

**Image processing stack:**

-   OpenCV
-   MediaPipe Face Mesh
-   Pillow
-   rembg (background removal)

The backend exposes a single endpoint:

`POST /process-image`

which returns the final JPG.

------------------------------------------------------------------------

# Image processing pipeline

The core of the system is a deterministic computer‑vision pipeline:

1.  **Image normalization**\
    EXIF rotation and RGB conversion.

2.  **Face detection and landmarks**\
    MediaPipe FaceMesh extracts facial landmarks.

3.  **Geometry crop**\
    Face is positioned so that:

    -   eyes sit near the center of the frame\
    -   face occupies \~70--80% of image height\
    -   final ratio matches 35×45

4.  **Background removal**\
    Subject is separated from the background and composited onto a white
    background.

5.  **Lighting normalization**\
    Global luminance adjustment in LAB color space.

6.  **Minimal denoise & sharpening**

7.  **JPEG export**\
    Image is resized and compressed to satisfy resolution and file size
    constraints.

------------------------------------------------------------------------

# AI‑assisted development

The project was developed using **Cursor**, an AI‑assisted coding
environment.

AI was used to:

-   design the system architecture
-   scaffold project structure
-   implement parts of the FastAPI backend
-   iterate on the image‑processing pipeline
-   debug and refine validation logic

The goal of this experiment was to explore how AI tools can accelerate
**end‑to‑end MVP development**.

------------------------------------------------------------------------

# Tech stack

**Frontend**\
Next.js

**Backend**\
FastAPI (Python)

**Computer vision**\
OpenCV\
MediaPipe\
Pillow\
rembg

**Development environment**\
Cursor (AI‑assisted coding)

------------------------------------------------------------------------

# Demo

The repository includes a short demo showing the full flow:

upload → processing → preview → download

------------------------------------------------------------------------

# Running locally

### Requirements

-   Python 3.11+
-   Node.js 18+

### Backend

    cd backend
    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt
    uvicorn app.main:app --reload

### Frontend

    cd frontend
    npm install
    npm run dev

Application:

http://localhost:3000
