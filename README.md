# Bushfire Burnt-Area Mapping

Semantic segmentation of burnt areas in Australian bushfires from Sentinel-2 satellite imagery, using a U-Net convolutional neural network. Includes a Flask web app that pulls live Sentinel-2 tiles and overlays the model's predicted burn boundary.

> **Context:** This was our first introduction to and built deep learning project, built for a university subject (UTS, Deep Learning & CNNs, 2025) where we were introduced to deep learning for the first time properly. Instead of using a simple existing dataset/problem for the sake of learning we wanted to take ona mroe ambitious project. We chose to attempt to build a deep learning model for accurately mapping the burn scars of bushfires in Australia. We quickly realised that there was no existing public machine learning ready dataset for this task so the biggest obstacle to our project was sourcing and effectively labelling enough imagery of bushfires before trying to train a model at all. This biggest effort of this project became data prepration and preprocessing and all the issues that came with it. We had about ~7-8 weeks from start to finish.

What we built was a complete end to end pipeline (from hand-labelling our own dataset through to a working demo) and it also has real, documented limitations. I've kept it honest rather than polishing over the rough edges; the [What I'd do differently](#what-id-do-differently) section is part of the point. We are proud of what were able to build from scratch within the time we had and the skills we learnt, particularly with data preparation and processing.

This project was selected and showcased at the UTS Tech Festival in 2025. You can find the project report and the presentation here:

Report: docs/report.pdf
Presentation: https://www.figma.com/deck/3dS7tllxgHeMzR4ugr1EmB

---

## What it does

1. Takes a geographic bounding box and a date.
2. Fetches a Sentinel-2 false-colour image (SWIR/NIR bands `B12`, `B11`, `B04`, which make burn scars stand out) from the Sentinel Hub API.
3. Runs a U-Net to produce a per-pixel burnt/not-burnt mask.
4. Cleans the mask with morphological opening/closing and overlays it in red on the original image.

## Results

| Metric | Value |
|---|---|
| IoU (Jaccard) | **0.67** |
| Dice coefficient | **0.67** |
| Train loss | 0.14 |
| Validation loss | 0.13 |

Trained for 25 epochs (early-stopped from a max of 50, patience 5). Validation loss tracked training loss closely, so no significant overfitting. The model delineates clear burn boundaries well; it struggles with smoke/cloud interference and with telling active fires apart from old burn scars.

<!-- Add a sample prediction image here — it's the single most persuasive thing in the README:
![Sample prediction](docs/figures/sample_prediction.png)
-->

## The dataset (built from scratch)

No suitable labelled dataset existed, so we made one:

- Used NSW fire-boundary geodatabases and the Copernicus EO Browser to locate historic large Australian bushfires.
- Downloaded ~1 km ground-footprint patches from the Sentinel Hub API around those fires.
- Hand-annotated burn masks in **CVAT**, labelling **~4,800 images** balanced between fire and no-fire scenes.
- Exported in a Pascal VOC-style layout (`JPEGImages/`, `SegmentationClass/`, split into `train/` and `val/`).

The annotated dataset and trained weights are not stored in this repo (too large for Git) — see [Data](#data) and [Model weights](#model-weights) for download links.

## Model

A standard U-Net (encoder / bottleneck / decoder with skip connections), 3-channel input → 1-channel mask.

- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** Adam, lr `1e-4`
- **Batch size:** 4
- **Input size:** 512 × 512
- **Augmentation:** random rotation (±15°), horizontal/vertical flip, contrast jitter

Full architecture is in [`src/model.py`](src/model.py); training and evaluation are in [`notebooks/fire_segmentation.ipynb`](notebooks/fire_segmentation.ipynb).

## Repository structure

```
├── notebooks/fire_segmentation.ipynb   # data prep, training, evaluation
├── src/
│   ├── model.py                         # U-Net
│   ├── dataset.py                       # SegmentationDataset
│   ├── create_missing_masks.py
│   └── inference.py                     # predict + mask post-processing
├── app/
│   ├── app.py                           # Flask demo (Sentinel Hub + inference)
│   └── templates/form.html
├── data/README.md                       # dataset download + expected layout
├── models/README.md                     # weight download links
└── docs/report.pdf                      # full project report
```

## Setup

```bash
git clone https://github.com/<you>/bushfire-burnt-area-mapping.git
cd bushfire-burnt-area-mapping
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the web app

The app needs Sentinel Hub OAuth credentials (free tier available at https://www.sentinel-hub.com/). Copy the example env file and fill in your own:

```bash
cp .env.example .env
# edit .env with your SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET
```

Download the trained weights (see [Model weights](#model-weights)) into `models/`, then:

```bash
python app/app.py
# open http://localhost:8070
```

## Data

The ~4,800-image annotated dataset isn't committed (89 MB of zips). Download link and the expected VOC-style folder layout are in [`data/README.md`](data/README.md).

## Model weights

`best_model.pth` (~119 MB) is hosted outside Git. See [`models/README.md`](models/README.md) for the download link (via GitHub Releases or `gdown`).

## What I'd do differently

Being honest about the limitations, since this was a learning project:

- **Dataset consistency** — annotating burn boundaries by hand across three people introduced label noise; a shared annotation guideline and inter-annotator checks would have helped a lot.
- **Smoke and cloud** — the model conflates smoke/cloud with burn scars. Adding a cloud mask (e.g. Sentinel-2 SCL band) as input or a filtering step would reduce false positives.
- **Old vs. active burns** — false-colour alone can't always separate a fresh burn from an ancient scar; multi-temporal input (before/after pairs) would give the model the change signal it's missing.
- **Metrics** — IoU/Dice around 0.67 is a reasonable first result but leaves clear headroom; a Dice or combined BCE+Dice loss, and more/cleaner data, would be the obvious next levers. The poor accuracy comes down primarily to preprocessing mistakes we made during the project.

Future directions we noted in the report: transfer learning from a pretrained encoder, and incorporating aerial linescan imagery alongside the satellite data.

## Team

Built by Sukush Kunwar, **Oliver Revis**, and Francis Clive Fernandes (UTS "Fire Safety Team", 2025).

## License

MIT — see [LICENSE](LICENSE).
