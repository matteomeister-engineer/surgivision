# SurgiVision OS

**Intraoperative AI Perception System — Browser-native surgical scene understanding**

---

## Overview

SurgiVision OS is a real-time surgical scene perception application that runs entirely in the browser. It demonstrates the kind of intraoperative intelligence pipeline that powers next-generation robotic surgery systems — combining a large vision-language model for semantic understanding with a domain-specific segmentation backbone trained on endoscopic imagery.

The system processes laparoscopic and endoscopic video frames and produces:
- Pixel-level anatomical segmentation masks
- Surgical phase recognition
- Action triplet detection (`instrument → verb → target`)
- Safety warnings and critical event detection
- Real-time inference metrics

---

## Architecture

```
Endoscopic Frame
       │
       ├──► Groq · Llama 4 Scout Vision ──► Surgical Phase
       │         (LLM scene understanding)    Action Triplets
       │                                      Safety Warnings
       │
       └──► EndoViT + DPT ──────────────► Pixel-level Segmentation Masks
                (domain-pretrained ViT)       13-class anatomical labels
```

### Dual-Pipeline Design

**Pipeline 1 — LLM Scene Understanding (Groq · Llama 4 Scout)**
- Identifies surgical phase (Dissection, Clipping, Hemostasis, etc.)
- Detects action triplets: `<instrument, verb, target>`
- Generates safety warnings for critical structures
- ~1.5s inference via Groq free API

**Pipeline 2 — Surgical Segmentation (EndoViT + DPT)**
- EndoViT: Vision Transformer pre-trained on Endo700k (743,724 endoscopic images)
- Fine-tuned on CholecSeg8k (8,080 annotated laparoscopic frames, 13 classes)
- DPT decoder head for pixel-precise segmentation
- MPS-accelerated inference on Apple Silicon
- ~2-4s per frame

Both pipelines run in parallel — the LLM provides semantic context, EndoViT provides spatial precision.

---

## Segmentation Classes (CholecSeg8k · 13 classes)

| Class | Color | Description |
|---|---|---|
| Liver | `#c43020` | Hepatic parenchyma |
| Gallbladder | `#40a040` | Cholecystic structure |
| Grasper | `#c0c8d0` | Laparoscopic grasper |
| L-Hook Electrocautery | `#e0e0ff` | Cautery instrument |
| Fat | `#d0b040` | Adipose tissue |
| Cystic Duct | `#e03050` | Critical bile duct |
| Hepatic Vein | `#8040c0` | Vascular structure |
| Blood | `#ff2040` | Active bleeding |
| Abdominal Wall | `#c08060` | Peritoneal wall |
| Connective Tissue | `#b08080` | Fascial tissue |
| Gastrointestinal | `#a06040` | GI tract |
| Liver Ligament | `#804020` | Hepatic ligaments |

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Vanilla JS · HTML5 Canvas API · CSS Grid Bento layout |
| LLM Inference | Groq API · Llama 4 Scout 17B Vision |
| Segmentation Backbone | EndoViT (MAE ViT-Base, Endo700k pretrained) |
| Segmentation Head | DPT (Dense Prediction Transformer) |
| Training | PyTorch 2.0 · CUDA/MPS · AdamW · CosineAnnealingLR |
| Loss | CrossEntropy + Dice |
| Backend | Python · Flask · CORS |
| Acceleration | Apple MPS (inference) · Kaggle P100 (training) |

---

## Setup

### 1. Segmentation server (EndoViT)

```bash
cd /path/to/EndoViT
conda activate endovit
python surgivision_server.py
# Runs on http://localhost:5050
```

### 2. Web app

```bash
cd /path/to/SurgiVision
python3 -m http.server 8080
# Open http://localhost:8080
```

### 3. API key

Enter your [Groq API key](https://console.groq.com) (`gsk_...`) in the topbar. Free tier, no credit card required.

---

## Training

Fine-tuning EndoViT on CholecSeg8k (few-shot · 4 videos · Kaggle P100):

```bash
python datasets/CholecSeg8k/utils/preprocess_CholecSeg8k_multi_process.py \
    --data_dir /path/to/archive \
    --output_dir ./datasets/CholecSeg8k/data_preprocessed \
    --cpu_count 4
# Then run EndoViT_SurgiVision_Training.ipynb on Kaggle
```

Config: backbone LR `1e-5` · head LR `1e-4` · 50 epochs · batch 12 · ~65% mIoU (4-video few-shot)

---

## Highlights

- **Real-time surgical perception**: dual-pipeline inference optimised for low latency on-device
- **Domain-specific model development**: EndoViT fine-tuned on cholecystectomy data, not generic ImageNet features
- **Safety-critical design**: explicit detection of cystic duct and hepatic vein with alert escalation
- **On-device inference**: MPS-optimised for Apple Silicon, adaptable to embedded surgical robotics constraints
- **Synthetic data readiness**: architecture designed to integrate with generative endoscopic data pipelines

---

## References

- [EndoViT](https://link.springer.com/article/10.1007/s11548-024-03091-5) — Batić et al., 2024
- [CholecSeg8k](https://arxiv.org/abs/2012.12453) — Hong et al., 2020
- [DPT](https://arxiv.org/abs/2103.13413) — Ranftl et al., 2021
- [MAE](https://arxiv.org/abs/2111.06377) — He et al., 2021

---

## License

MIT — research and portfolio demonstration.
Dataset usage subject to CholecSeg8k (CC BY-NC-SA 4.0) and respective Endo700k dataset licenses.
