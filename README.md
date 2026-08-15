# VisionFortress

An image classification model that predicts a Team Fortress 2 character class (Scout, Heavy, Spy, etc.) from screenshots. Built while learning PyTorch through [learnpytorch.io](https://www.learnpytorch.io/), turned into a project with a custom dataset and a custom model architecture.

## About the Project

I couldn't find a Team Fortress 2 classification dataset, so I used a [TF2 object detection dataset from Roboflow Universe](https://universe.roboflow.com/tf2-xw2rt/team-fortress-2-classes-nuo4m). That dataset had characters labeled with bounding boxes (YOLO format). I processed it with a script that:

- Crops out each bounding box individually (if an image has multiple characters, each one is extracted separately),
- Adds a bit of padding around the edges,
- Resizes to a square while preserving aspect ratio (letterboxing),

which produced a classic `ImageFolder`-style **multi-class classification** dataset (`train/<class>/`, `valid/<class>/`, `test/<class>/`). That conversion script isn't part of this repo.

## Model

`VisionFortressV1` is a small CNN with:

- 3 convolutional blocks (each: 2x Conv2d + BatchNorm + ReLU + MaxPool), channel count doubles each block (20 → 40 → 80)
- `AdaptiveAvgPool2d` + `Dropout(0.3)` + `Linear` output layer
- ~113,700 trainable parameters in total

Input size is **128x128**. Data augmentation used during training: horizontal flip, ±10° rotation, and color jitter. Optimizer: `AdamW`, loss: `CrossEntropyLoss`, trained 150 epochs. (more can give better results)

**Test results:** ~62% accuracy, 1.22 loss (across 9 classes, where random guessing would be ~11%).

Classes: `demoman, engineer, heavy, medic, pyro, scout, sniper, soldier, spy`

## Repo Structure

```
visionfortress/
├── visionfortress_V1.ipynb   # Data loading, model, training, evaluation
├── helper_functions.py       # train_step / test_step / eval_model functions
├── predict.py                 # Single-image inference (e.g. testing on a game screenshot)
├── models/
│   └── model_0.pth           # Trained model weights
├── cropped_dataset/           # train/valid/test folders (cropped images per class)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training / evaluation

Open `visionfortress_V1.ipynb` and run it top to bottom. The dataset is already prepared under `cropped_dataset/`; if you want to train on your own data, you'll need to build the same folder structure (`train/valid/test` → one folder per class).

### Predicting on a single image (works best with one character in frame)

```bash
python predict.py
```

Just update these two lines at the top of the script with your own paths:

```python
MODEL_PATH = "models/model_0.pth"
IMAGE_PATH = "screenshot.png"
```

Running it prints the most likely class along with the probability for every class.

## Notes / Limitations

- I made this to just learn pytorch and CNNs
- The model was trained on **already-cropped** images where the character fills most of the frame. On a raw screenshot where the character is small or there are multiple characters, you'd need a detection step first — this repo only covers the classification part.
- Trained on 1,829 training images with a small CNN, so ~64% test accuracy is in the expected range; a larger/more balanced dataset or transfer learning could improve it further.
- This project is for educational/research purposes only, not for commercial use, and is not affiliated with or endorsed by Valve Corporation.

## License

MIT — see the `LICENSE` file for details.