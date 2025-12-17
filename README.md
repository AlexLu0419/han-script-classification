# Han Script Classification (SC / TC / JP / KR)

The dataset is generated using **Unicode encoding (CJK related blocks)** combined with the **Google Noto Font Family**:

- **Noto Serif**
- **Noto Sans**

Targeting fonts for the following four languages:

- **Simplified Chinese (SC)**
- **Traditional Chinese (TC)**
- **Japanese (JP)**
- **Korean (KR)**

Each Unicode character supported by each font is rendered into a **128×128 glyph PNG image**.

Each language includes multiple font weights (Regular / Bold / Medium / Black…), and can be widely used for tasks such as font analysis, font recognition, style transfer, and OCR synthetic data generation.

## 📁 1. Glyph Generation Process

### Font Source (`fonts/`)

**Please download the font first in .ttf format under the `fonts/` directory.**

- [Noto Sans](https://fonts.google.com/noto/specimen/Noto+Sans)
- [Noto Serif](https://fonts.google.com/noto/specimen/Noto+Serif)

1. Read the **cmap** of each `.ttf` file.
2. Filter CJK Unicode blocks.
3. Render each character as `U+XXXX.png`.
4. Output by Language → Font → Character Code.

## 📦 2. Generated Data Structure (`data/`)

```
data/
  Noto_Sans_JP/
    NotoSerifSC-Regular/
      U+4E00.png
      ...
    NotoSansJP-Bold/

    NotoSansJP-ExtraBold/

    NotoSansJP-Light/
    
    NotoSansJP-Medium/
    ...
  Noto_Sans_TC/

  Noto_Sans_SC/

  Noto_Sans_KR/

  Noto_Serif_JP/

  Noto_Serif_TC/

  Noto_Serif_SC/

  Noto_Serif_KR/

```

## ⚠️ 3. Notes

- Different Noto sub-fonts do not support exactly the same Unicode characters.
- JP/KR include Kana or Hangul characters.
- If full Unicode support is needed, the script can be modified.

## 4. Models and Experiments

### Best Model (`aml_final.py` & `aml_eval.py`)
This is the final model used to generate the results in the report.
- **Architecture**: `SimpleCNN`
  - 3 Convolutional Blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool)
  - 2 Fully Connected Layers
- **Data Split**: Random split (80% Training, 20% Testing)
- **Preprocessing**: Resize to 64x64, Grayscale, Normalize (mean=0.5, std=0.5)
- **Training**: 50 Epochs, Adam Optimizer (lr=1e-5), Batch Size 128

### Other Experiments

#### `cnn_mid_v2_1.ipynb`
- **Architecture**: `MediumCNNPlus` (Deeper network with 4 Conv blocks and Global Average Pooling)
- **Data Split**: Random split (80% Train, 10% Val, 10% Test)
- **Preprocessing**: Resize to 64x64, No Normalization
- **Training**: 15 Epochs, Adam Optimizer (lr=1e-3)

#### `cnn_mid_v2_nonrd_1.ipynb`
- **Architecture**: `MediumCNNPlus` (Same as above)
- **Data Split**: **Stratified Split by Font Weight**
  - Uses `split_by_weight_folders` to ensure balanced sampling from each font style/weight.
  - Selects fixed number of samples (Train: 10000, Val: 1500, Test: 1500) per leaf folder.
- **Purpose**: To test model performance when data is explicitly balanced across different font weights, avoiding potential bias from over-represented styles in a random split.
