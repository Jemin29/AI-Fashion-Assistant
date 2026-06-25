# Prompt Engineering Guide — Week 2

## Overview

SDXL is extremely prompt-sensitive. This guide covers best practices for
generating high-quality fashion images with the `PromptBuilder` system.

---

## Prompt Structure

Every generated prompt follows this structure (order matters):

```
[SUBJECT]  →  [STYLE TAGS]  →  [GENDER MOD]  →  [SEASON MOD]  →
[FASHION TECH]  →  [QUALITY BOOSTERS]  →  [STYLE QUALITY]
```

### Example Assembled Prompt
```
A woman wearing an elegant red silk evening gown,
luxury fashion, high-end designer, premium materials, haute couture,
female model, womenswear, summer fashion, lightweight fabric,
fashion editorial, fashion photography, lookbook style,
photorealistic, 8k resolution, ultra detailed, sharp focus,
luxury editorial, dramatic lighting, 8k
```

---

## Style Presets

Use `style_presets.py` to apply curated tag bundles:

| Style | Best For | Guidance Scale |
|-------|----------|---------------|
| `streetwear` | Hoodies, sneakers, graphic tees | 7.0 |
| `formal` | Suits, blazers, evening wear | 8.0 |
| `casual` | T-shirts, jeans, everyday looks | 7.0 |
| `athleisure` | Sports, gym, activewear | 7.5 |
| `bohemian` | Dresses, flowy fabrics, earthy tones | 7.0 |
| `luxury` | Designer, couture, premium materials | 8.5 |
| `minimalist` | Clean lines, neutral palette | 7.5 |
| `avant_garde` | Conceptual, sculptural, editorial | 9.0 |
| `ethnic` | Traditional wear, embroidery | 7.5 |

---

## CLIP Token Limit

SDXL uses **two CLIP text encoders**, each with a **77-token limit**.

The `PromptBuilder` estimates token count (1 word ≈ 1.3 tokens) and
automatically truncates if needed.

**Tips to stay within limits:**
- Use concise, specific adjectives rather than long sentences
- Let `PromptBuilder` append quality tags — don't duplicate them
- A 50-word prompt ≈ 65 tokens — safely within budget

---

## Negative Prompt Best Practices

Always use a negative prompt with SDXL. The `NegativePrompts` module
provides pre-built bundles:

```python
from week2.prompts.negative_prompts import get_fashion_negative, format_negative

neg = format_negative(get_fashion_negative())
```

### Critical Negatives for Fashion

```
blurry, out of focus, low quality, deformed, bad anatomy,
watermark, text, signature, cropped, bad clothing, wrinkled fabric,
dirty, stained, torn fabric, ill-fitting
```

---

## Scheduler Recommendations

| Scheduler | Style | Speed | Quality |
|-----------|-------|-------|---------|
| `euler` | General purpose | Fast | Good |
| `euler_a` | Artistic/diverse | Fast | High variance |
| `dpm++` | Photorealistic | Medium | Excellent |
| `ddim` | Reproducible | Fast | Good |
| `unipc` | High quality | Slow | Excellent |

---

## Seeds and Reproducibility

```python
# Reproducible generation
result = generator.generate(prompt="...", seed=42)

# Random seed (different each run)
result = generator.generate(prompt="...", seed=-1)

# Find the seed used
print(result.seed)   # Always stored, even for random runs
```

---

## Category Templates

Use `PromptBuilder.build_from_metadata()` with Week 1 dataset records.
Templates are defined in `configs/prompt_config.yaml`:

```yaml
category_templates:
  dresses: "a {gender} wearing a {style} {color} dress, {description}"
  hoodies: "a {gender} wearing a {style} {color} hoodie, {description}"
```

---

## Common Issues and Fixes

| Problem | Fix |
|---------|-----|
| Grey/washed out images | Use `sdxl-vae-fp16-fix` VAE |
| Blurry faces | Add `detailed face, sharp focus` to prompt |
| Wrong clothing style | Be more specific: "oversized graphic tee" not just "shirt" |
| Background distracts | Add `clean white studio background` |
| Too dark/moody | Add `bright lighting, well-lit, soft light` |
| Body anatomy issues | Add `correct anatomy, natural pose` |
