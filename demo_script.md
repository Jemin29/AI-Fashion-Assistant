# AI Fashion Assistant - Live Demo Walkthrough Script

This script guides a presenter through showing all verified functional capabilities of the **AI Fashion Assistant** platform (Weeks 1–7).

---

## 📋 Prerequisites & Setup
1. **API Server Running**: Ensure the FastAPI backend is running:
   ```bash
   uvicorn week7.backend.main:app --reload --port 8000
   ```
2. **UI App Running**: Ensure the Gradio interface is launched:
   ```bash
   python week6/gradio_app/main.py
   ```
3. **Environment**: Both apps should run in mock mode if GPU weights are not mounted locally.

---

## 🎭 Step-by-Step Demo Flow

### 🎬 Phase 1: Gateway Diagnostics & Health Checks
* **Action**: Open a terminal or tool like Postman, or point the browser to `http://127.0.0.1:8000/api/v1/health`.
* **Presenter Script**:
  > *"Welcome to the AI Fashion Assistant demo. We begin at the **FastAPI Gateway** layer, which serves as our enterprise-grade access point. By querying the `/api/v1/health` endpoint, we see a JSON diagnostic status confirming that all backend engines—including our vector database, Redis cache state, and Stable Diffusion models—are healthy and operational."*
* **Visual Check**: Show the JSON output containing `"status": "ok"` or `"mock"`.

### 🎬 Phase 2: Text-to-Fashion Design Studio
* **Action**: Open the Gradio Web UI (`http://127.0.0.1:7860`). Navigate to the **Text-to-Fashion** tab.
* **Input Prompt**: `Avant-garde sculptural blazer, off-white`
* **Settings**: Select the `Minimalist Studio` preset, set CFG scale to `8.0`, steps to `30`, and seed to `42`. Click **Generate Design**.
* **Presenter Script**:
  > *"Next, let's look at the **Fashion Generation Studio**. Here, a designer can type a text prompt to render high-fidelity concepts. Behind the scenes, the Gradio client communicates with our FastAPI `/generate` endpoint, drawing from custom SDXL style presets. Within seconds, we get a highly stylized product rendering complete with generation metadata."*
* **Visual Check**: Show the generated off-white blazer design image and the sidecar metadata cards.

### 🎬 Phase 3: Sketch2Design (ControlNet Shape-Conditioning)
* **Action**: Navigate to the **Sketch2Design** tab. Upload a silhouette sketch (or use a simple drawing hanger outline).
* **Input Prompt**: `Linen summer dress, light sky blue, resort wear`
* **Settings**: Set Conditioning Scale to `0.85` and click **Preprocess Image** then **Conditioned Generation**.
* **Presenter Script**:
  > *"A critical challenge in fashion is maintaining structural control. In the **Sketch2Design** tab, a designer can upload a hand-drawn sketch. First, our preprocessor extracts edge outlines. Then, the ControlNet engine conditions the Stable Diffusion pipeline using these boundaries. The result is a design that perfectly adheres to the physical contours of the sketch while incorporating the textures and colors of the prompt."*
* **Visual Check**: Display the edge map and generated light blue linen dress side-by-side in the comparison view.

### 🎬 Phase 4: Brand LoRA Studio & Style Mixing
* **Action**: Navigate to the **Brand Studio** tab. 
  1. Select `nike` and click **Generate** with prompt `Athletic windbreaker`.
  2. Toggle **Style Mixer**, check both `nike` and `gucci` boxes, set weights to `nike: 0.6` and `gucci: 0.4`, then click **Generate Mixed Style**.
* **Presenter Script**:
  > *"To ensure alignment with specific brand aesthetics, we've integrated a **LoRA Brand Registry**. Watch as we apply a single adapter, such as 'Nike', to inject sporty, high-performance styling cues. Furthermore, using our **Style Mixer**, we can blend multiple brand weights. Here, we're generating a design that is 60% Nike athletic techwear and 40% Gucci luxury haute-couture—creating a brand-new hybrid garment concept."*
* **Visual Check**: Show the Nike windbreaker design, followed by the mixed-brand luxury sportswear concept showing design elements of both brands.

### 🎬 Phase 5: Grounded Q&A (Fashion RAG Assistant)
* **Action**: Navigate to the **RAG Assistant** tab.
* **Input Question**: `How do I style a beige trench coat?` Click **Submit**.
* **Presenter Script**:
  > *"Designers also need access to fabric knowledge and trend libraries. The **Fashion RAG Assistant** provides a conversational interface grounded in our verified database. When we submit a question, the assistant uses a hybrid retriever—performing keyword search and FAISS/ChromaDB vector lookup—to extract relevant passages. The response is grounded in facts, preventing AI hallucination, and displays clickable citations pointing directly back to our knowledge base documents."*
* **Visual Check**: Show the chat bubble response with citations listed at the bottom.

### 🎬 Phase 6: Personalized Recommendations
* **Action**: Navigate to the **Recommendations** tab. Enter preferences: `Gender: unisex`, `Aesthetic: streetwear`, `Occasion: casual`, `Fit: oversized`. Click **Get Recommendations**.
* **Presenter Script**:
  > *"Finally, we connect designs to users. The **Personalization Service** tracks user search histories and profiles to suggest relevant styles and brand pairings. Based on our inputs, the recommender runs a similarity query to return a curated list of style coordinates and brand aesthetics tailored to this specific user profile."*
* **Visual Check**: Show the card displaying recommended style coordinates and matching brands.
