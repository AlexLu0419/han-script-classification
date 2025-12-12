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
