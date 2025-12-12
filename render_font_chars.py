import os
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

# ====== CONFIG ======

# Root directories (relative to where you run the script)
FONTS_ROOT = Path("fonts")   # /fonts
DATA_ROOT = Path("data")     # /data

IMG_SIZE = (128, 128)
FONT_SIZE = 110   # a bit larger than before
BG_COLOR = 255   # white (grayscale)
FG_COLOR = 0     # black (grayscale)

# Enable / disable unicode range filtering
FILTER_TO_CJK_AND_RELATED = True


# ====== UNICODE RANGE FILTERING ======

def is_interesting_codepoint(cp: int) -> bool:
    """
    Decide whether we care about this codepoint.
    You can tweak these ranges as needed.

    Currently includes:
    - CJK Unified Ideographs + Extension A
    - CJK Compatibility Ideographs
    - Hiragana, Katakana
    - Hangul syllables + Jamo
    - CJK symbols & punctuation, fullwidth forms
    """
    ranges = [
        (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0xAC00, 0xD7A3),  # Hangul Syllables
        (0x1100, 0x11FF),  # Hangul Jamo
        (0x3000, 0x303F),  # CJK Symbols and Punctuation
        (0xFF00, 0xFFEF),  # Halfwidth and Fullwidth Forms
    ]
    for start, end in ranges:
        if start <= cp <= end:
            return True
    return False


# ====== CORE LOGIC ======

def get_supported_chars_from_ttf(ttf_path: Path):
    """
    Use fontTools to read the TTF's best cmap and return a sorted list
    of supported Unicode characters (optionally filtered to CJK-related).
    """
    tt = TTFont(str(ttf_path))
    cmap = tt.getBestCmap()  # dict: {codepoint: glyph_name}

    codepoints = []
    for cp in cmap.keys():
        if FILTER_TO_CJK_AND_RELATED:
            if not is_interesting_codepoint(cp):
                continue
        codepoints.append(cp)

    codepoints = sorted(set(codepoints))
    chars = [chr(cp) for cp in codepoints]
    return codepoints, chars


def render_char_image(char: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """
    Render a single character into a grayscale PIL Image, centered tightly.
    """
    img = Image.new("L", IMG_SIZE, color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Get full bounding box including offsets
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        # bbox = (left, top, right, bottom)
        left, top, right, bottom = bbox
        w = right - left
        h = bottom - top

        # Center the bbox inside the image
        x = (IMG_SIZE[0] - w) / 2 - left
        y = (IMG_SIZE[1] - h) / 2 - top
    except AttributeError:
        # Fallback for very old Pillow: best-effort centering
        w, h = draw.textsize(char, font=font)
        x = (IMG_SIZE[0] - w) / 2
        y = (IMG_SIZE[1] - h) / 2

    draw.text((x, y), char, font=font, fill=FG_COLOR)
    return img


def process_single_ttf(ttf_path: Path):
    """
    For one TTF file:
      - find supported characters
      - render each as image
      - save to /data/<font_group>/<ttf_basename>/U+XXXX.png
    """
    font_group = ttf_path.parent.name             # e.g. 'korean', 'japanese', 'tc', 'sc'
    font_basename = ttf_path.stem                 # file name without extension

    out_dir = DATA_ROOT / font_group / font_basename
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Processing: {ttf_path} -> {out_dir}")

    codepoints, chars = get_supported_chars_from_ttf(ttf_path)

    # Load font once
    try:
        pil_font = ImageFont.truetype(str(ttf_path), FONT_SIZE)
    except Exception as e:
        print(f"[ERROR] Failed to load font {ttf_path}: {e}")
        return

    for cp, ch in zip(codepoints, chars):
        filename = f"U+{cp:04X}.png"
        out_path = out_dir / filename

        # Skip if already exists
        if out_path.exists():
            continue

        try:
            img = render_char_image(ch, pil_font)
            img.save(out_path)
        except Exception as e:
            print(f"[WARN] Failed to render U+{cp:04X} ({ch}) in {ttf_path.name}: {e}")

    print(f"[DONE] {ttf_path} -> {len(chars)} glyphs rendered.")


def main():
    if not FONTS_ROOT.exists():
        print(f"[ERROR] Fonts root not found: {FONTS_ROOT}")
        return
    
    wanted_fonts = ['Bold', 'Light', 'Medium', 'Regular']

    all_ttf_files = list(FONTS_ROOT.rglob("*.ttf"))
    if not all_ttf_files:
        print(f"[WARN] No .ttf files found under {FONTS_ROOT}")
        return
    
    ttf_files = []
    for ttf_file in all_ttf_files:
        for fonts in wanted_fonts:
            if fonts in ttf_file.name:
                ttf_files.append(ttf_file)
                continue

    print(f"[INFO] Found {len(ttf_files)} .ttf files.")

    DATA_ROOT.mkdir(exist_ok=True)

    for ttf_path in ttf_files:
        process_single_ttf(ttf_path)


if __name__ == "__main__":
    main()