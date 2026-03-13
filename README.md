# 🔬 SurgiVision OS v2.0 — Intraoperative Intelligence Platform

A browser-native surgical AI perception platform built to simulate what **Intuitive Surgical's Intraoperative Intelligence Group** is building for the da Vinci robot. Runs entirely in your browser — no server, no build step.

> **Drop a real endoscopic frame → Claude Vision API → live surgical scene understanding**

![Version](https://img.shields.io/badge/version-2.0.0-00d4ff?style=flat-square)
![Model](https://img.shields.io/badge/model-claude--sonnet--4-00ff88?style=flat-square)
![Input](https://img.shields.io/badge/input-image%20%2B%20video-a78bfa?style=flat-square)
![Runtime](https://img.shields.io/badge/runtime-browser%20native-ffaa00?style=flat-square)

---

## ✨ What's New in v2 (vs v1)

| Feature | v1 | v2 |
|---|---|---|
| AI Engine | DETR (COCO, generic) | **Claude Sonnet Vision API** (surgical-aware) |
| Input | Images only | **Images + Video** with scrubber |
| Output | Bounding boxes | **Segmentation + phase + safety warnings** |
| Scene context | None | **Surgical phase detection** (Dissection, Clipping, etc.) |
| Safety | None | **Critical structure proximity warnings** |
| HUD | Basic | **Full intraoperative HUD** with live overlays |
| Data | Synthetic demo only | **Real dataset links** (CholecSeg8k, Endoscapes, etc.) |

---

## 🎯 Capabilities

### 1. Surgical Scene Perception
Claude Vision analyzes each frame and identifies:
- **Anatomical structures** — liver, gallbladder, cystic duct, hepatocystic triangle
- **Surgical instruments** — grasper, clipper, hook, scissors, irrigator
- **Tissue types** — fat, connective tissue, blood vessels, bile
- **Fluids** — blood, bile, irrigation fluid

### 2. Surgical Phase Detection
Identifies the current phase of surgery in real time:
- Trocar Placement → Preparation → Calot Triangle Dissection
- Clipping & Cutting → Gallbladder Dissection → Extraction → Closure

### 3. Safety Assessment
Flags critical situations:
- ⚠ Instrument proximity to major vessel
- ⚠ Unclear anatomy / poor visibility
- ⚠ Active bleeding detected
- ⚠ Critical View of Safety (CVS) not achieved

### 4. Video Frame Analysis
- Load any surgical video (MP4, MOV, WEBM)
- Scrub to any frame with the timeline
- Analyze individual frames on demand
- Temporal navigation through procedure

---

## 🚀 Getting Started

### 1. Get an API Key
Go to [console.anthropic.com](https://console.anthropic.com) and create an API key.

### 2. Open the App
```bash
# Option A: direct open
open index.html

# Option B: local server (recommended for video)
python3 -m http.server 8080
# visit http://localhost:8080
```

### 3. Enter Your API Key
Paste your `sk-ant-api...` key in the topbar input and click **SAVE**.

### 4. Load Surgical Footage
Drop an image or video, or use one of the linked free datasets.

---

## 📁 Free Surgical Datasets

| Dataset | Type | Structures | Access |
|---|---|---|---|
| **[CholecSeg8k](https://www.kaggle.com/datasets/newslab/cholecseg8k)** | Images (annotated) | 13 surgical classes | Free on Kaggle |
| **[Endoscapes2023](https://github.com/CAMMA-public/Endoscapes)** | Video frames | Anatomy + CVS | Free on GitHub |
| **[CholecT45](https://github.com/CAMMA-public/cholect50)** | Videos | Instrument+verb+target | Free on GitHub |
| **[SurgiSR4K](https://arxiv.org/abs/2507.00209)** | 4K da Vinci video | Full surgical scene | Free on HuggingFace |
| **[Cholec80](https://camma.unistra.fr/datasets/)** | 80 full videos | Phase + tool | Request form |

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| AI Vision | Anthropic Claude Sonnet (claude-sonnet-4-20250514) |
| Rendering | HTML5 Canvas API (2D overlay) |
| Video | HTML5 Video Element + frame capture |
| Fonts | Orbitron + IBM Plex Mono (Google Fonts) |
| Framework | Vanilla JS — zero dependencies |
| Hosting | Any static host (GitHub Pages, Netlify, etc.) |

---

## 📐 Architecture

```
User drops image/video
        ↓
Frame extracted to Canvas
        ↓
Canvas → base64 JPEG
        ↓
POST /v1/messages (Claude Vision API)
  - System: surgical perception prompt
  - Image: base64 frame
        ↓
JSON response parsed:
  { phase, structures[], warnings[], summary }
        ↓
Canvas overlay rendered (bounding boxes + labels)
Right panel updated (detections, phase, warnings)
HUD overlays updated
Safety alert triggered if critical
```

---

## 🔮 Roadmap

- [ ] Auto-analyze every N seconds during video playback
- [ ] Export analysis as COCO JSON annotation format
- [ ] Side-by-side frame comparison (before/after)
- [ ] Temporal tracking across frames
- [ ] Custom fine-tuned surgical model via Replicate/HuggingFace
- [ ] Multi-frame procedure summary report

---

## ⚠️ Disclaimer

For demonstration and research purposes only. **Not a medical device.** Do not use for clinical decision-making.

---

## 📄 License

MIT
