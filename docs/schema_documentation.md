
# UnifiedFashionItem Schema Documentation
Version: 1.0.0 | Date: 2026-06-03

## Overview
The UnifiedFashionItem is the canonical, dataset-agnostic schema for all
fashion records in the AI Fashion Design Assistant pipeline.

It unifies records from two source datasets:
  • FashionGen   — HDF5-based, ~293K items, text descriptions, no landmarks
  • DeepFashion  — TXT annotation-based, ~800K items, landmarks, no descriptions

## Field Reference

### Required Fields (must be non-empty)
| Field          | Type            | Description                              |
|----------------|-----------------|------------------------------------------|
| image_id       | str             | Unique ID: "FG_0000042" or "DF_img_..."  |
| image_path     | str             | Relative POSIX path from project root    |
| source_dataset | DatasetSource   | "fashiongen" or "deepfashion"            |
| category       | CategoryEnum    | One of 11 taxonomy keys (see below)      |

### Taxonomy Fields
| Field          | Type            | Values                                   |
|----------------|-----------------|------------------------------------------|
| category       | CategoryEnum    | t_shirts, shirts, hoodies, jackets,      |
|                |                 | pants, jeans, shorts, dresses,           |
|                |                 | ethnic_wear, footwear, accessories       |
| subcategory    | str | None      | Sub-level label, e.g. "graphic_tee"      |
| gender         | GenderEnum|None | men, women, unisex (None if unknown)     |

### Appearance Attribute Fields
| Field          | Type            | Description                              |
|----------------|-----------------|------------------------------------------|
| color          | List[str]       | Normalised color names, e.g. ["Navy"]    |
| fabric         | List[str]       | Normalised fabric names, e.g. ["Cotton"] |
| pattern        | List[str]       | Pattern types, e.g. ["stripes"]          |
| fit            | FitEnum | None  | slim_fit, regular_fit, relaxed_fit,      |
|                |                 | oversized, cropped, skinny, straight,    |
|                |                 | athletic_fit                             |

### Contextual Attribute Fields
| Field          | Type            | Description                              |
|----------------|-----------------|------------------------------------------|
| style          | StyleEnum|None  | streetwear, luxury, formal,              |
|                |                 | business_casual, techwear, minimalist,   |
|                |                 | vintage, athleisure                      |
| season         | SeasonEnum      | spring, summer, autumn, winter,          |
|                |                 | all_season (default)                     |
| occasion       | List[Occasion]  | casual, business_casual, formal, party,  |
|                |                 | sport, outdoor, beach, wedding_festive,  |
|                |                 | lounge                                   |
| description    | str | None      | Human-written text (FashionGen only)     |

### Raw Attribute List
| Field          | Type            | Description                              |
|----------------|-----------------|------------------------------------------|
| attributes     | List[str]       | Free-form attribute strings.             |
|                |                 | FashionGen: from dict flatten.           |
|                |                 | DeepFashion: decoded from 1000-dim vec.  |

### Spatial Data Fields (DeepFashion-specific)
| Field          | Type                 | Description                         |
|----------------|----------------------|-------------------------------------|
| landmarks      | List[LandmarkPoint]  | 0 or 6 landmark points              |
| bounding_box   | BoundingBox | None   | Pixel + normalised bbox             |

LandmarkPoint: name (str), x (float [0,1]), y (float [0,1]), visible (bool)
Landmark names: left_collar, right_collar, left_sleeve, right_sleeve,
                left_hem, right_hem

BoundingBox: x1, y1, x2, y2 (int pixels), nx1, ny1, nx2, ny2 (float [0,1])

### Pipeline Provenance Fields (auto-populated)
| Field          | Type            | Description                              |
|----------------|-----------------|------------------------------------------|
| is_valid       | bool            | True if no hard errors exist             |
| errors         | List[str]       | Hard validation errors                   |
| warnings       | List[str]       | Soft validation warnings                 |
| processed_at   | str             | ISO-8601 UTC timestamp                   |
| schema_version | str             | "1.0.0"                         |

## Dataset Field Coverage Matrix
| Field          | FashionGen | DeepFashion |
|----------------|------------|-------------|
| image_id       | ✅          | ✅           |
| image_path     | ✅          | ✅           |
| category       | ✅          | ✅           |
| subcategory    | ✅          | ⚠️ (raw name)|
| gender         | ✅          | ❌ None      |
| color          | ✅          | ⚠️ extracted |
| fabric         | ✅          | ⚠️ extracted |
| pattern        | ✅          | ⚠️ extracted |
| fit            | ✅          | ⚠️ extracted |
| style          | ✅ inferred | ❌ None      |
| season         | ✅ inferred | ⚠️ all_season|
| occasion       | ✅ inferred | ❌ []        |
| description    | ✅          | ❌ None      |
| attributes     | ⚠️ flattened| ✅           |
| landmarks      | ❌ []       | ✅ 6 points  |
| bounding_box   | ❌ None     | ✅           |

Legend: ✅ = populated  ⚠️ = partial/derived  ❌ = always empty/None

## Validation Rules
1. image_id must be non-empty and contain no whitespace.
2. image_path must use forward slashes only.
3. category must be one of the 11 CategoryEnum values.
4. gender, style, fit, season must be valid Enum values or None.
5. LandmarkPoint x, y must be in [0.0, 1.0].
6. BoundingBox x2 > x1 and y2 > y1 (non-degenerate).
7. Cross-field: dresses + men → warning.
8. Cross-field: accessories/footwear + fit → warning.
9. DeepFashion: landmarks must be 0 or 6 entries.
10. FashionGen: landmarks must be [] (warning if non-empty).
