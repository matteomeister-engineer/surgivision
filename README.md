# SurgiVision OS

**Intraoperative AI Perception System — Browser-native surgical scene understanding**

🔗 **[Try it live →](https://matteomeister-engineer.github.io/surgivision)** · [Get a free Groq API key](https://console.groq.com)

---

## What it does

Drop any endoscopic or laparoscopic frame or video and get back in ~2s:
- Surgical phase recognition (Dissection, Clipping, Hemostasis, etc.)
- Action triplets (`instrument → verb → target`)
- Anatomical structure detection with color-coded overlays
- Safety warnings for critical structures
- Real-time inference metrics

Works entirely in the browser — no installation, no backend, no local server.

---

## Quick start

1. Open **[surgivision](https://matteomeister-engineer.github.io/surgivision)**
2. Enter your [Groq API key](https://console.groq.com) in the topbar (`gsk_...`) — free, no credit card
3. Drop an endoscopic image or video

---

## Architecture

```
Endoscopic Frame / Video
        │
        └──► Groq · Llama 4 Scout Vision (17B) ──► Surgical Phase
                                                    Action Triplets
                                                    Structure Detection
                                                    Safety Warnings
```

Single LLM pipeline — Groq handles all semantic understanding with ~1.5s inference latency.
Overlay masks rendered client-side via HTML5 Canvas flood-fill on detected bounding boxes.

---

## Tech stack

| | |
|---|---|
| Frontend | Vanilla JS · HTML5 Canvas API · CSS Grid |
| Vision LLM | Groq · Llama 4 Scout 17B Vision |
| Deployment | GitHub Pages (static, no server) |

---

## Highlights

- **Zero setup** — open URL, add API key, drop frame
- **Video support** — auto-plays and analyzes every 2s with smooth overlay fade
- **Groq free tier** — no cost, no credit card required
- **Fully browser-native** — single HTML file, no build step, no framework

---

## References

- [EndoViT](https://link.springer.com/article/10.1007/s11548-024-03091-5) — Batić et al., 2024
- [CholecSeg8k](https://arxiv.org/abs/2012.12453) — Hong et al., 2020

---

## License

MIT — research and portfolio use.
