# Week 6 — Creative Studio API Reference

## ⚙️ Configuration System

### `get_config() -> AppConfig`
Loads and validates application settings from `configs/app_config.yaml` and environment variables. Returns a validated Pydantic model configuration.

### `get_services_config() -> ServicesConfig`
Loads model weights, database paths, and service configurations from `configs/services_config.yaml`.

---

## 📝 Logging Framework

### `setup_logging(log_level: str = None, log_dir: Path = None, force: bool = False)`
Configures the Loguru logger with rotation on file sinks, access logs, and console levels.

### `log_access(action: str, details: str = "")`
Logs general user interactions into `logs/access.log`.

### `log_generation(prompt: str, model: str, latency_ms: float, success: bool = True)`
Writes detailed records of image generation runs to `logs/studio.log`.

---

## 📦 Service Adapters

### `GenerationService`
- **`generate(prompt, negative_prompt, steps, cfg, width, height, seed, style_label) -> (Image, dict)`**  
  Generates a fashion design image. Falls back to generating styled gradients in mock mode.
- **`get_style_presets() -> list[str]`**  
  Returns a list of pre-configured fashion style names.

### `ControlNetService`
- **`generate_conditioned(prompt, control_image, mode, scale, steps, cfg) -> (Image, dict)`**  
  Creates a guided image structure based on sketch, depth, or pose maps.
- **`preprocess_image(image, mode) -> Image`**  
  Converts image formats to appropriate edge guidance outlines.

### `LoRAService`
- **`generate_with_brand(prompt, brand, scale, steps, cfg) -> (Image, dict)`**  
  Generates a personalized design with brand fine-tuned weights (Nike, Gucci, Zara, H&M).
- **`mix_styles(prompt, brand_weights) -> (Image, dict)`**  
  Blends styling details of multiple brands together.

### `RAGService`
- **`answer_question(question) -> dict`**  
  Retrieves knowledge answers with references and sources from ChromaDB.
- **`semantic_search(query, n_results) -> list[dict]`**  
  Queries the vector index for raw search matches.

### `RecommendationService`
- **`recommend_styles(gender, style, occasion, fit, n) -> list[dict]`**  
  Computes style matches based on user preferences.
- **`recommend_brands(styles, aesthetic, n) -> list[dict]`**  
  Returns brand aesthetics that overlap with user interests.

### `TrendService`
- **`get_all_trends() -> list[dict]`**  
  Fetches fashion trends sorted by growth velocity.
- **`forecast_season(season) -> list[dict]`**  
  Predicts upcoming season highlights.

### `EvaluationService`
- **`get_last_report() -> dict`**  
  Loads the previous RAG benchmarking run.
- **`run_evaluation() -> dict`**  
  Executes the RAGEvaluator suite and updates output reports.
