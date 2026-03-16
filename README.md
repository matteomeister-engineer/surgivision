# SurgiVision OS

**Intraoperative AI Perception System — Browser-native surgical scene understanding**

🔗 **[Try it live → localhost:8080](http://localhost:8080)** · [Get a free Groq API key](https://console.groq.com)

---

## What it does

Drop any endoscopic or laparoscopic frame and get back in ~2s:
- Pixel-level anatomical segmentation masks
- Surgical phase recognition
- Action triplets (`instrument → verb → target`)
- Safety warnings for critical structures
- Real-time inference metrics

---

## Architecture

```
Endoscopic Frame
       │
       ├──► Groq · Llama 4 Scout Vision ──► Phase · Triplets · Warnings
       │
       └──► EndoViT + DPT ──────────────► Pixel-level segmentation masks
```

Two pipelines run in parallel — the LLM handles semantic understanding, EndoViT handles spatial precision.

**Pipeline 1 — Groq · Llama 4 Scout Vision** (~1.5s)
Surgical phase, action triplets `<instrument, verb, target>`, safety warnings.

**Pipeline 2 — EndoViT + DPT** (~2-4s)
ViT-Base pretrained on Endo700k (743K endoscopic frames), fine-tuned on CholecSeg8k (8,080 annotated cholecystectomy frames, 13 classes) via a DPT decoder head. MPS-accelerated on Apple Silicon.

---

## Quick start

```bash
# 1. Start the segmentation server
cd /path/to/EndoViT && conda activate endovit
python surgivision_server.py        # → http://localhost:5050

# 2. Serve the app
cd /path/to/SurgiVision
python3 -m http.server 8080         # → http://localhost:8080

# 3. Enter your Groq API key in the topbar (free, no credit card)
```

---

## Tech stack

| | |
|---|---|
| Frontend | Vanilla JS · HTML5 Canvas · CSS Grid |
| LLM | Groq · Llama 4 Scout 17B Vision |
| Segmentation | EndoViT (MAE ViT-Base) + DPT head |
| Training | PyTorch · AdamW · CosineAnnealingLR · CrossEntropy + Dice |
| Backend | Python · Flask |
| Hardware | Apple MPS (inference) · Kaggle P100 (training) |

---

## Training

```bash
# Preprocess CholecSeg8k
python datasets/CholecSeg8k/utils/preprocess_CholecSeg8k_multi_process.py \
    --data_dir /path/to/archive \
    --output_dir ./datasets/CholecSeg8k/data_preprocessed

# Then run EndoViT_SurgiVision_Training.ipynb on Kaggle
```

16 videos · 50 epochs · batch 12 · backbone LR `5e-6` · head LR `5e-5` · ~65% mIoU

---

## Highlights

- Dual-pipeline design — LLM semantic context + domain ViT spatial precision running in parallel
- EndoViT fine-tuned on real cholecystectomy data, not generic ImageNet features
- Explicit safety escalation for critical structures (cystic duct, hepatic vein, active bleeding)
- Fully browser-native — no build step, no framework, drop-in deployable

---

## References

- [EndoViT](https://link.springer.com/article/10.1007/s11548-024-03091-5) — Batić et al., 2024
- [CholecSeg8k](https://arxiv.org/abs/2012.12453) — Hong et al., 2020
- [DPT](https://arxiv.org/abs/2103.13413) — Ranftl et al., 2021
- [MAE](https://arxiv.org/abs/2111.06377) — He et al., 2021

---

## License

MIT — research and portfolio use.
Dataset usage subject to CholecSeg8k (CC BY-NC-SA 4.0) and respective Endo700k dataset licenses.
