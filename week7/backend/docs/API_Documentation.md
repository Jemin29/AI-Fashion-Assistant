# AI Fashion Assistant - Production API Documentation

This document serves as the official production API specification for the **AI Fashion Assistant REST Web Backend**. The backend is built using FastAPI and integrates Redis caching/task-tracking, Celery background workers, JWT-based security layers, Falconsai NSFW content filtering, Pillow transparent watermarking, and slowapi rate limiting.

---

## 1. Documentation & Interactive Playgrounds

FastAPI automatically parses Pydantic schemas and route annotations to compile standard documentation interfaces.

### OpenAPI Specification
- **Endpoint**: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)
- **Description**: Dynamically generated OpenAPI 3.0.2 JSON schema outlining all routes, parameters, request body models, and response wrappers.

### Swagger UI
- **Endpoint**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Description**: Interactive API playground allowing developer teams to review endpoints, input payloads, authorize calls, and test live HTTP request-response cycles directly inside the browser.

### ReDoc
- **Endpoint**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Description**: Sleek, clean, production-grade rendering of the OpenAPI spec optimized for reference reading and API catalog documentation.

---

## 2. Authentication & Authorization Guide

The system uses standard stateless **JSON Web Tokens (JWT)** to secure routes. Authenticated users are granted access and refresh tokens.

### Token Lifecycle
1. **Login**: Client submits credentials to `/auth/token` using standard `application/x-www-form-urlencoded` fields.
2. **Authorization**: If credentials are valid, the server returns an `access_token` (valid for 30 minutes) and a `refresh_token` (valid for 7 days).
3. **Protected Access**: For all secured routes, the client must include the JWT in the HTTP headers:
   ```http
   Authorization: Bearer <YOUR_ACCESS_TOKEN>
   ```
4. **Token Refresh**: When the access token expires, the client sends a `POST` request to `/auth/refresh` containing the refresh token. The server validates the token signature and returns a brand-new access/refresh token pair.

---

## 3. Global Schema Standards

### Unified Success Envelope
All success responses (except raw standard forms or oauth responses) are returned in the following unified structure:
```json
{
  "success": true,
  "data": { ... }
}
```

### Unified Error Envelope
All API exceptions, validation failures, rate limits, and server crashes return standardized JSON structures:
```json
{
  "success": false,
  "error": {
    "code": "<ERROR_CODE>",
    "message": "<User-friendly explanation of what went wrong.>"
  }
}
```

#### Common Error Codes:
- `VALIDATION_ERROR` (`422 Unprocessable Entity`): Input structural validation failed.
- `HTTP_ERROR_401` (`401 Unauthorized`): Bearer token is missing, expired, or signature is invalid.
- `UNSAFE_CONTENT_DETECTED` (`400 Bad Request`): Image generation flagged by the NSFW filter pipeline.
- `RATE_LIMIT_EXCEEDED` (`429 Too Many Requests`): Request speed crossed configured slowapi limits.
- `INTERNAL_SERVER_ERROR` (`500 Internal Server Error`): Uncaught exception handled by ErrorHandlerMiddleware.

---

## 4. REST API Endpoint Reference

### 4.1 Authentication Router (`/auth`)

#### `POST /auth/register` (Register Account)
- **Description**: Registers a new user with a unique username and a hashed password (direct bcrypt).
- **Rate Limit**: 15 requests/minute.
- **Request Payload**:
  ```json
  {
    "username": "fashion_designer",
    "password": "SuperSecretPassword123"
  }
  ```
- **Response (`201 Created`)**:
  ```json
  {
    "message": "User registered successfully",
    "username": "fashion_designer"
  }
  ```

#### `POST /auth/token` (Obtain Access & Refresh Tokens)
- **Description**: Authenticate credentials and get JWT token pairs.
- **Request Payload** (`application/x-www-form-urlencoded`):
  - `username`: `fashion_designer`
  - `password`: `SuperSecretPassword123`
- **Response (`200 OK`)**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1Ni...",
    "refresh_token": "eyJhbGciOiJIUzI1Ni...",
    "token_type": "bearer"
  }
  ```

#### `POST /auth/refresh` (Refresh Access Token)
- **Description**: Exchange a valid refresh token for a brand-new access token pair.
- **Request Payload**:
  ```json
  {
    "refresh_token": "eyJhbGciOiJIUzI1Ni..."
  }
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1Ni_new...",
    "refresh_token": "eyJhbGciOiJIUzI1Ni_new...",
    "token_type": "bearer"
  }
  ```

#### `GET /auth/me` (Get Active Profile)
- **Description**: Retrieves the profile details of the authenticated user. Secured by JWT validation.
- **Headers Required**: `Authorization: Bearer <access_token>`
- **Response (`200 OK`)**:
  ```json
  {
    "username": "fashion_designer",
    "role": "designer",
    "status": "active"
  }
  ```

---

### 4.2 Text-to-Fashion Generation Router (`/api/v1/generation`)

#### `GET /api/v1/generation/presets` (Retrieve Style Presets)
- **Description**: Lists pre-configured style tags (e.g. Streetwear, Haute Couture) used to inject trigger descriptors.
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "data": [
      {
        "name": "Streetwear Editorial",
        "trigger": "streetwear, urban editorial, oversized silhouette, bold typography"
      }
    ]
  }
  ```

#### `POST /api/v1/generation/generate` (Generate Image)
- **Description**: Renders a fashion design matching prompt parameters. Image returned is base64 encoded and contains the transparent drop-shadow watermark signature in the bottom-right corner.
- **Rate Limit**: 10 requests/minute.
- **Request Payload**:
  ```json
  {
    "prompt": "A modern black silk blazer, studio runway model",
    "negative_prompt": "blurry, low quality",
    "style_preset": "Minimalist Studio",
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "cfg": 7.5,
    "seed": 42,
    "session_id": "session_designer_1"
  }
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "data": {
      "image": "iVBORw0KGgoAAAANSUhEUgAA...",
      "meta": {
        "prompt": "A modern black silk blazer, studio runway model, minimalist fashion, clean white studio...",
        "negative_prompt": "blurry, low quality",
        "seed": 42,
        "width": 1024,
        "height": 1024,
        "guidance_scale": 7.5,
        "device_used": "cpu",
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "run_mode": "mock",
        "generation_time_s": 0.12
      }
    }
  }
  ```

---

### 4.3 ControlNet Studio Router (`/api/v1/controlnet`)

#### `POST /api/v1/controlnet/preprocess` (Preview Sketch/Edge Map)
- **Description**: Processes an uploaded layout design file to extract its edge maps or conditioning guides (Canny, Depth, Pose) for layout consistency.
- **Rate Limit**: 10 requests/minute.
- **Request Payload** (`multipart/form-data`):
  - `mode`: `canny` (Form String)
  - `file`: `sketch.png` (Multipart Binary File)
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "data": {
      "preprocessed_image_path": "C:\\Users\\HP\\Desktop\\AI Fashion Agent\\fashion-ai-assistant\\week7\\backend\\sketches\\preprocess_abc123.png",
      "meta": {
        "mode": "canny",
        "width": 512,
        "height": 512,
        "processing_time_s": 0.05
      }
    }
  }
  ```

#### `POST /api/v1/controlnet/generate` (Generate Conditioned Design)
- **Description**: Generates design concepts using the layout structures extracted from the uploaded conditioning file.
- **Rate Limit**: 10 requests/minute.
- **Request Payload** (`multipart/form-data`):
  - `prompt`: `A loose linen summer shirt` (Form String)
  - `negative_prompt`: `blurry` (Form String)
  - `mode`: `canny` (Form String)
  - `conditioning_scale`: `0.8` (Form Float)
  - `steps`: `25` (Form Int)
  - `cfg`: `7.5` (Form Float)
  - `seed`: `42` (Form Int, Optional)
  - `session_id`: `session_designer_1` (Form String)
  - `file`: `sketch_contour.png` (Multipart Binary File)
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "data": {
      "image_path": "C:\\Users\\HP\\Desktop\\...\\outputs\\generated\\controlnet_canny_def456.png",
      "image": "iVBORw0KGgoAAAANSUhEUgAA...",
      "metadata": {
        "prompt": "A loose linen summer shirt",
        "controlnet_type": "canny",
        "scale": 0.8,
        "seed": 42
      }
    }
  }
  ```

---

### 4.4 Brand Studio Router (LoRA) (`/api/v1/lora`)

#### `POST /api/v1/lora/generate` (Single Brand Generation)
- **Description**: Generates design concepts styled to emulate a specific brand's adapter weights (e.g. Nike, Gucci, Zara).
- **Rate Limit**: 10 requests/minute.
- **Request Payload**:
  ```json
  {
    "prompt": "Oversized hoodie with technical stitching",
    "brand": "nike",
    "lora_scale": 0.85,
    "steps": 25,
    "cfg": 7.5,
    "seed": 1001,
    "session_id": "designer_session"
  }
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "data": {
      "saved_path": "C:\\Users\\HP\\Desktop\\...\\outputs\\generated\\lora_nike_789.png",
      "metadata": {
        "prompt": "Oversized hoodie with technical stitching, swoosh logo, sportswear aesthetic",
        "brand": "nike",
        "scale": 0.85,
        "seed": 1001
      }
    }
  }
  ```

#### `POST /api/v1/lora/mix` (LoRA Adapter Style Blending)
- **Description**: Blends multiple brand adapters dynamically by interpolating weight ratios to output hybrid brand aesthetics.
- **Rate Limit**: 10 requests/minute.
- **Request Payload**:
  ```json
  {
    "prompt": "Avant-garde streetwear tracksuit",
    "brand_weights": {
      "nike": 0.6,
      "gucci": 0.4
    },
    "steps": 25,
    "cfg": 7.5,
    "seed": 2026,
    "session_id": "designer_session"
  }
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "data": {
      "saved_path": "C:\\Users\\HP\\Desktop\\...\\outputs\\generated\\lora_mixed_321.png",
      "metadata": {
        "prompt": "Avant-garde streetwear tracksuit",
        "mixed_weights": {
          "nike": 0.6,
          "gucci": 0.4
        },
        "seed": 2026
      }
    }
  }
  ```

---

### 4.5 RAG & Assistant Router (`/api/v1/rag`)

#### `POST /api/v1/rag/chat` (Conversational Assistant Chat)
- **Description**: Submit dialogue queries to the RAG fashion assistant. Queries vector-stores for fashion-expert context.
- **Rate Limit**: 20 requests/minute.
- **Request Payload**:
  ```json
  {
    "message": "Which fabrics are most suitable for summer tailored suits?",
    "session_id": "user_session_99"
  }
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "data": {
      "answer": "For summer tailored suits, breathable and lightweight fabrics are highly recommended. Linen is the premier choice due to its loose weave and excellent thermal dissipation, though it creases easily. Cotton and tropical wool blends offer clean structures with high air permeability.",
      "recommendations": ["Linen", "Seersucker", "Tropical Wool Blend"],
      "confidence_score": 0.94,
      "citations": [
        {
          "source": "fashion_domain_research.py",
          "text": "Linen properties: breathable, loose-weave, summer luxury."
        }
      ]
    }
  }
  ```

#### `POST /api/v1/rag/search` (Semantic Vector Space Search)
- **Description**: Executes direct semantic vector similarity search across the vector collection database (ChromaDB).
- **Rate Limit**: 20 requests/minute.
- **Request Payload**:
  ```json
  {
    "query": "breathable weave structures",
    "limit": 3
  }
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "data": {
      "citations": [
        {
          "document_id": "doc_102",
          "content": "Open weave structures such as linen, seersucker, and canvas weaves maximize air ventilation...",
          "score": 0.82
        }
      ]
    }
  }
  ```

---

### 4.6 Recommendations Router (`/api/v1/recommendations`)

#### `POST /api/v1/recommendations/styles` (Style Recommendation Engine)
- **Description**: Get customized style suggestions based on target fit, occasion, and preferences.
- **Rate Limit**: 20 requests/minute.
- **Request Payload**:
  ```json
  {
    "gender": "women",
    "style": "minimalist",
    "occasion": "formal",
    "fit": "tailored",
    "limit": 2,
    "color": "camel"
  }
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "data": [
      {
        "id": "rec_001",
        "description": "Camel tailored trousers paired with matching silk blouse",
        "fit": "tailored",
        "style": "minimalist",
        "occasion": "formal"
      }
    ]
  }
  ```

---

### 4.7 Asynchronous Task Manager Router (`/task`)

#### `POST /task/start` (Queue Asynchronous Task)
- **Description**: Submits an intensive modeling task to be queued and executed by background Celery workers. Avoids locking request cycles.
- **Request Payload**:
  ```json
  {
    "task_type": "generation",
    "payload": {
      "prompt": "An industrial techwear parka",
      "seed": 42,
      "resolution": "1024x1024"
    }
  }
  ```
- **Response (`201 Created`)**:
  ```json
  {
    "task_id": "59cb3a52-d17e-4b68-b80c-78d10b06bcf4",
    "task_type": "generation",
    "status": "PENDING"
  }
  ```

#### `GET /task/{id}` (Poll Task Status)
- **Description**: Query the Redis metadata store to track status, execution metrics, and pull completed task results.
- **Response (`200 OK`)**:
  ```json
  {
    "task_id": "59cb3a52-d17e-4b68-b80c-78d10b06bcf4",
    "task_type": "generation",
    "status": "SUCCESS",
    "progress": 100,
    "execution_time_s": 1.45,
    "result": {
      "image": "iVBORw0KGgoAAAANSUhEUgAA...",
      "metadata": {
        "prompt": "An industrial techwear parka",
        "seed": 42,
        "width": 1024,
        "height": 1024
      }
    }
  }
  ```

#### `DELETE /task/{id}` (Revoke Task)
- **Description**: Cancels and terminates an active or queued Celery task, purging its state entry from Redis.
- **Response (`200 OK`)**:
  ```json
  {
    "message": "Task 59cb3a52-d17e-4b68-b80c-78d10b06bcf4 cancelled successfully",
    "task_id": "59cb3a52-d17e-4b68-b80c-78d10b06bcf4"
  }
  ```

---

## 5. Security & Safety Pipeline Responses

### 5.1 NSFW Safety Violation (`400 Bad Request`)
If a prompt triggers generating inappropriate elements or the image safety filter model classifies the output bitmap as unsafe:
- **HTTP Status**: `400 Bad Request`
- **Response Envelope**:
  ```json
  {
    "success": false,
    "error": {
      "code": "UNSAFE_CONTENT_DETECTED",
      "message": "Generative output contains contents flagged as unsafe by NSFW filters."
    }
  }
  ```

### 5.2 Rate Limit Triggered (`429 Too Many Requests`)
If requests are made at a rate higher than the limits configured for an endpoint group:
- **HTTP Status**: `429 Too Many Requests`
- **Response Envelope**:
  ```json
  {
    "success": false,
    "error": {
      "code": "RATE_LIMIT_EXCEEDED",
      "message": "Rate limit exceeded: 10 per 1 minute. Please slow down."
    }
  }
  ```
