# Integration Test Report — AI Fashion Creative Studio

**Generated**: 2026-07-02 04:38:41 UTC
**Environment**: CPU-only | Mock Mode | No GPU
**Overall Status**: ALL PASS
**Total Tests**: **66 / 66 passed**
**Total Runtime**: 3.90s

---

## Page Summary

| Page | Pass | Total | Status |
|:---|:---:|:---:|:---:|
| Home | 2 | 2 | PASS |
| Text-to-Fashion | 8 | 8 | PASS |
| Sketch2Design | 9 | 9 | PASS |
| Brand Studio | 10 | 10 | PASS |
| Fashion Assistant | 10 | 10 | PASS |
| Recommendation Hub | 10 | 10 | PASS |
| Gallery | 10 | 10 | PASS |
| Evaluation Dashboard | 7 | 7 | PASS |

---

## Detailed Results per Page

### Home

| # | Callback | Scenario | Result | Outputs | Latency |
|:---:|:---|:---|:---:|:---:|---:|
| 1 | `refresh_status` | returns HTML string | PASS | 1/1 | 349ms |
| 2 | `refresh_recent` | returns list for gallery | PASS | 1/1 | 33ms |

### Text-to-Fashion

| # | Callback | Scenario | Result | Outputs | Latency |
|:---:|:---|:---|:---:|:---:|---:|
| 1 | `on_preset_change` | preset=Minimalist | PASS | 1/1 | 0ms |
| 2 | `on_preset_change` | preset=Streetwear | PASS | 1/1 | 0ms |
| 3 | `on_preset_change` | preset=Haute Couture | PASS | 1/1 | 0ms |
| 4 | `on_generate` | valid prompt | PASS | 4/4 | 10ms |
| 5 | `on_generate` | empty prompt (validation guard) | PASS | 4/4 | 0ms |
| 6 | `on_generate` | style=Streetwear | PASS | 4/4 | 10ms |
| 7 | `on_generate` | style=Haute Couture | PASS | 4/4 | 10ms |
| 8 | `on_clear` | clears all outputs | PASS | 4/4 | 0ms |

### Sketch2Design

| # | Callback | Scenario | Result | Outputs | Latency |
|:---:|:---|:---|:---:|:---:|---:|
| 1 | `on_preview` | mode=canny | PASS | 1/1 | 3ms |
| 2 | `on_preview` | mode=sketch | PASS | 1/1 | 2ms |
| 3 | `on_preview` | mode=pose | PASS | 1/1 | 1ms |
| 4 | `on_preview` | mode=depth | PASS | 1/1 | 1ms |
| 5 | `on_preview` | None input → None output | PASS | 1/1 | 0ms |
| 6 | `on_generate` | mode=canny | PASS | 2/2 | 10ms |
| 7 | `on_generate` | mode=sketch | PASS | 2/2 | 11ms |
| 8 | `on_generate` | None image | PASS | 2/2 | 0ms |
| 9 | `on_generate` | empty prompt | PASS | 2/2 | 0ms |

### Brand Studio

| # | Callback | Scenario | Result | Outputs | Latency |
|:---:|:---|:---|:---:|:---:|---:|
| 1 | `update_brand_display` | brand=nike | PASS | 1/1 | 0ms |
| 2 | `update_brand_display` | brand=gucci | PASS | 1/1 | 0ms |
| 3 | `update_brand_display` | brand=zara | PASS | 1/1 | 0ms |
| 4 | `update_brand_display` | brand=hm | PASS | 1/1 | 0ms |
| 5 | `on_generate (switcher)` | brand=nike | PASS | 3/3 | 10ms |
| 6 | `on_generate (switcher)` | brand=gucci | PASS | 3/3 | 11ms |
| 7 | `on_generate (switcher)` | brand=zara | PASS | 3/3 | 11ms |
| 8 | `on_generate (switcher)` | empty prompt | PASS | 3/3 | 0ms |
| 9 | `on_compare` | all brands comparison | PASS | 3/3 | 82ms |
| 10 | `on_blend (mixer)` | two-brand blend | PASS | 3/3 | 10ms |

### Fashion Assistant

| # | Callback | Scenario | Result | Outputs | Latency |
|:---:|:---|:---|:---:|:---:|---:|
| 1 | `on_send` | Q: What is quiet luxury? | PASS | 3/3 | 4ms |
| 2 | `on_send` | Q: How do I style a capsule wardrobe? | PASS | 3/3 | 4ms |
| 3 | `on_send` | Q: Tell me about Nike aesthetics. | PASS | 3/3 | 3ms |
| 4 | `on_send` | Q: What fabrics are best for summer? | PASS | 3/3 | 1ms |
| 5 | `on_send` | Q: Explain the difference between streetwea | PASS | 3/3 | 3ms |
| 6 | `on_send` | empty message → safe fallback | PASS | 3/3 | 0ms |
| 7 | `on_clear` | clears chat | PASS | 3/3 | 0ms |
| 8 | `suggestion_fill` | fills input: What fabrics feel premium? | PASS | 1/1 | 0ms |
| 9 | `suggestion_fill` | fills input: Trending colors this season | PASS | 1/1 | 0ms |
| 10 | `suggestion_fill` | fills input: How to style oversized blazers | PASS | 1/1 | 0ms |

### Recommendation Hub

| # | Callback | Scenario | Result | Outputs | Latency |
|:---:|:---|:---|:---:|:---:|---:|
| 1 | `on_get_styles` | occasion=casual | PASS | 2/2 | 0ms |
| 2 | `on_get_styles` | occasion=formal | PASS | 2/2 | 0ms |
| 3 | `on_get_styles` | occasion=casual | PASS | 2/2 | 0ms |
| 4 | `on_get_brands` | styles=minimalist | PASS | 2/2 | 0ms |
| 5 | `on_get_brands` | styles=streetwear | PASS | 2/2 | 0ms |
| 6 | `on_get_brands` | styles=bohemian | PASS | 2/2 | 0ms |
| 7 | `on_get_trends` | season=spring_summer | PASS | 2/2 | 0ms |
| 8 | `on_get_trends` | season=autumn_winter | PASS | 2/2 | 0ms |
| 9 | `on_generate_lookbook` | mood=editorial | PASS | 2/2 | 0ms |
| 10 | `on_generate_lookbook` | mood=formal | PASS | 2/2 | 0ms |

### Gallery

| # | Callback | Scenario | Result | Outputs | Latency |
|:---:|:---|:---|:---:|:---:|---:|
| 1 | `handle_refresh` | search='', sort=newest | PASS | 7/7 | 1ms |
| 2 | `handle_refresh` | search='silk', sort=newest | PASS | 7/7 | 1ms |
| 3 | `handle_refresh` | search='', sort=oldest | PASS | 7/7 | 1ms |
| 4 | `handle_prev_page` | from page 2 | PASS | 8/8 | 1ms |
| 5 | `handle_prev_page` | floor at page 1 | PASS | 8/8 | 2ms |
| 6 | `handle_next_page` | from page 1 | PASS | 8/8 | 2ms |
| 7 | `handle_export` | format=json | PASS | 1/1 | 33ms |
| 8 | `handle_export` | format=csv | PASS | 1/1 | 20ms |
| 9 | `handle_export` | format=markdown | PASS | 1/1 | 27ms |
| 10 | `handle_save_review` | no entry_id selected | PASS | 2/2 | 0ms |

### Evaluation Dashboard

| # | Callback | Scenario | Result | Outputs | Latency |
|:---:|:---|:---|:---:|:---:|---:|
| 1 | `trigger_evaluation` | run #1 | PASS | 5/5 | 1002ms |
| 2 | `trigger_evaluation` | run #2 | PASS | 5/5 | 1002ms |
| 3 | `trigger_evaluation` | run #3 | PASS | 5/5 | 1002ms |
| 4 | `_build_metrics_cards` | renders summary HTML | PASS | 1/1 | 0ms |
| 5 | `_build_metrics_chart_html` | renders chart HTML | PASS | 1/1 | 0ms |
| 6 | `_build_cases_table` | renders cases markdown | PASS | 1/1 | 0ms |
| 7 | `_build_cases_table` | empty cases list | PASS | 1/1 | 0ms |

---

## Integration Test Strategy

### Approach

Each page is tested by:

1. **Service Instantiation** — all services started in `mock_mode=True` (no GPU)
2. **Callback Extraction** — inner callback functions reconstructed using the same service calls as the page builders
3. **Input Simulation** — realistic user inputs are supplied (prompts, images, sliders, dropdown values)
4. **Output Verification** — return value count, types, and key field presence are all checked
5. **Safe Callback Coverage** — all callbacks decorated with `@safe_callback` are implicitly tested via the underlying functions

### Coverage Matrix

| Page | Callbacks Tested | Input Scenarios |
|:---|:---:|:---:|
| Home Dashboard | 2 (refresh_status, refresh_recent) | 2 |
| Text-to-Fashion | 3 (preset_change, generate, clear) | 7 |
| Sketch2Design | 2 (preview, generate) | 8 |
| Brand Studio | 4 (brand_display, generate, compare, blend) | 10 |
| Fashion Assistant | 3 (send, clear, suggestion_fill) | 10 |
| Recommendation Hub | 4 (styles, brands, trends, lookbook) | 12 |
| Gallery | 5 (refresh, prev, next, export, save_review) | 10 |
| Evaluation Dashboard | 4 (run_evaluation, metrics_cards, chart, cases_table) | 7 |

---

## Conclusion

**All integration tests pass.** Every callback correctly handles simulated user inputs, returns the expected number of output values, and produces correct data types.

> Generated by `scripts/integration_test.py`
