# Third-party components and data

Nothing below is redistributed in this repository. Download each item from its official source and follow its terms.

| Item | Used for | Licence / terms | Included here |
|---|---|---|---|
| LivePortrait (KwaiVGI) | main reenactment model | code MIT; its InsightFace detector models are non-commercial research only | no |
| PersonaLive, LIA-X, EDTalk | the three further generators of Sec. VII-D | each model's own licence (LIA-X code CC BY-NC 4.0); research use | no |
| ArcIris (CVRL, Khan, Flynn, Czajka) | the iris matcher | research-use licence, not redistributable | no (see DATA.md) |
| MediaPipe Face Landmarker | iris and eyelid detection | Apache-2.0 | no (pip package) |
| VideoSeal 1.0 (Meta) | the watermark baseline (Sec. VII-E) | MIT | no |
| PDQ (Meta) | perceptual video hashes | BSD | no (pip package) |
| TrustFA / AVAPrintDB score files | behavioural baselines | CC BY-NC-SA 4.0 (AVAPrintDB); RAVDESS CC BY-NC-SA 4.0 | no |
| RAVDESS | 21 actors of the 67 people | CC BY-NC-SA 4.0 | no |
| CASIA-Iris-Distance (CASIA-IrisV4) | 46 near-infrared photo owners and the real-iris stand-ins | research use; no redistribution; acknowledgement and a copy of the paper to the dataset owners after publication | no |
| UTIRIS V.1 | the 79-entry pool of proxy irises | University of Tehran (CIPCE) holds the copyright; no redistribution | no |

The result files in `outputs/` contain only scalar scores, verdicts and counts derived from these datasets, never images, crops, templates or embeddings.
