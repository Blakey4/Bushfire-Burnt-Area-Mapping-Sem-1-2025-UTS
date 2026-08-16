# Bushfire Burnt-Area Mapping

Semantic segmentation of burnt areas in Australian bushfires from Sentinel-2 satellite imagery, using a U-Net convolutional neural network. Includes a Flask web app that pulls live Sentinel-2 tiles and overlays the model's predicted burn boundary.

<img width="1993" height="1240" alt="image" src="https://github.com/user-attachments/assets/dcff0573-8673-4e79-a0b0-304f2a431f6a" />



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

The results of the model were poor. This was large part due to issues found during the building of the dataset, silly preproecssing mistakes we made and the limited time we had left to build and experiemnt with the mdoel after creating the dataset. We are happy with what we are able to build and learn given this was our first introduction to deep learning and the time constraints we had.

<img width="1431" height="624" alt="image" src="https://github.com/user-attachments/assets/f1bb60db-8809-419c-a252-b531c559774e" />

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

### The Dataset - Identifying past bushfires with Imagery
To create our dataset the first step was to compile a list of fires and dates across the country that had Sentinel 2 imagery coverage. In our early research at this stage we found the FY 2019-202 Bushfire Boundaries Dataset and the NSW Operational Bushfire Boundaries Geodatabase. These mapped the final extent / burn scar of previous fires in Australia. As these were final extent maps that did not line up with the exact date/time of sentinel 2 imagery of each of the fires so could not be used to help label the data however we did use them as a spatial reference to pinpoint the location of previous large fires. We then manually viewed Sentinel 2 Imagery of each of the respective areas in Copernicus EO Browser to identify a sample of fires and dates from various states and territories across the east coast that we could use to build our dataset.

<img width="1464" height="875" alt="image" src="https://github.com/user-attachments/assets/a9f1f592-a92d-4988-8071-ed25014dc6f2" />

The image above shows the geodatabase used as reference (purple) and the selected regions
(orange)

Most of the imagery we decided to use construct our dataset was from the 2019-2020 Bushfire Season (with the exception of the 2024/2025 Grampians Fire) as it provided the greatest geographic spread of significant fires since Sentinel 2 Imagery became available. The list of fires regions we chose are listed below. These correspond to the orange boxes in the previous image.

| Fire | State | Dates |
|---|---|---|
| Yarrarubula South | New South Wales | 3/01/2020, 13/01/2020 |
| Woodenbong | Queensland | 16/11/2019 |
| Torrington | NSW / Queensland | 16/11/2019 |
| Tambo Complex | Queensland | 29/11/2019 |
| Northern NSW | New South Wales | 16/11/2019 |
| Gospers | New South Wales | 21/12/2019, 21/11/2019, 11/11/2019, 1/11/2019, 27/10/2019 |
| Lower Blue Mountains | New South Wales | 21/12/2019 |
| Kempsey | New South Wales | 16/11/2019 |
| Kangaroo Island | South Australia | 31/12/2019, 21/12/2019 |
| Hanging Rock | New South Wales | 21/12/2019 |
| Green Valley | NSW / Victoria | 8/01/2020, 3/01/2020 |
| Grampians | Victoria | 2/01/2025, 25/12/2024, 20/12/2024, 18/12/2024 |
| Georges Yard | New South Wales | 21/12/2019 |
| Currowan | New South Wales | 21/12/2019, 1/12/2019 |
| Bombay Fire | New South Wales | 1/12/2019 |

<!-- Insert Table of selected bushfires-->

### The Dataset - Downloading Patches from Sentinel Hub's API

<img width="920" height="869" alt="image" src="https://github.com/user-attachments/assets/2b47ae12-90f0-4e95-8e55-5e8dde6662df" />

<img width="1174" height="920" alt="image" src="https://github.com/user-attachments/assets/32cb4992-cea1-41a7-baef-54692cecb553" />

### The Dataset - Annotation / Labelling

<img width="1839" height="717" alt="image" src="https://github.com/user-attachments/assets/2f3902fd-db86-4eaf-80a5-393296772971" />


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
