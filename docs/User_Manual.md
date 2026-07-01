# User Manual - Gradio Creative Studio UI

Welcome to the **AI Fashion Creative Studio**! This guide describes how to navigate, input parameters, and review outputs across the 5 primary workspace tabs.

---

## 🎨 1. Concept Fashion Generation (Text-to-Fashion)

Convert descriptive text design prompts into high-fidelity apparel concepts.

### Action Steps
1. **Enter Prompt**: Type a description of the garment (e.g., *Oversized street hoodie with bold graphics*).
2. **Select Style Preset**: Choose a style from the dropdown (e.g., *Streetwear Editorial*, *Luxury Fashion Week*, *Minimalist Studio*).
3. **Configure Parameters**:
   * **Inference Steps**: Adjust between `1` and `150` (higher steps improve details but take longer).
   * **Guidance Scale (CFG)**: Adjust between `1.0` and `30.0` (higher scale makes the output adhere closer to the prompt).
   * **Resolution**: Select from standard resolution mappings (e.g. *Square 512x512*, *Portrait 512x768*).
4. **Generate**: Click **Generate Design**.
5. **Output**: Review the generated design card along with generation latency telemetry.

---

## 📐 2. Sketch-to-Design (ControlNet Shape Guidance)

Use hand-drawn drawings or boundary templates to force strict shapes on generations.

### Action Steps
1. **Upload Sketch**: Drag and drop an outline image or drawing into the input canvas.
2. **Select Conditioning Mode**: Choose **Canny** (edge outlines) or other guide parameters.
3. **Enter Text Prompt**: Type details of fabric texture and color choices (e.g., *Linen summer trousers in pure white*).
4. **Generate**: Click **Generate from Sketch**.
5. **Output**: Review the preprocessed outline guide displayed side-by-side with the final layout design.

---

## 🏷️ 3. Brand LoRA Styling Studio

Swap brand design aesthetics or mix brand DNA proportions to create custom hybrids.

### Action Steps
1. **Choose Single Adapter**: Select a brand (e.g., *Nike*) and slide the adapter scale to dictate brand intensity.
2. **Enable Multi-Style Mix**: Switch to the mixer panel to combine two brands.
3. **Select Weights**: Define adapter ratios (e.g., *50% Nike Sportswear* and *50% Gucci Haute-Couture*).
4. **Generate**: Click **Generate Styled Concept**.
5. **Output**: Review the mixed design with applied transparent copyright watermarks.

---

## 💬 4. Grounded Fashion Chat Assistant (RAG Chat)

Converse with an assistant to get advice on fabric choices, design aesthetics, or active color trends.

### Action Steps
1. **Type Question**: Ask questions like: *Why is linen popular in Spring/Summer? Explain its weave.*
2. **Submit**: Click **Send**.
3. **Output**: The response is displayed with citation references mapping answers directly back to knowledge base records (e.g. `[KB-024]`).

---

## 👗 5. Personalization Recommendations

Receive style recommendations matching user profiles.

### Action Steps
1. **Define Profile**: Input demographic information (gender, fit choices like relaxed/oversized, and target occasions like casual/formal).
2. **Recommend**: Click **Fetch Recommendations**.
3. **Output**: Review recommended style names and descriptions blended from database listings.
