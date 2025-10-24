# DeepSeek-OCR Technical Documentation

## 1. Detailed Technical Architecture

### 1.1 Multimodal Backbone
DeepSeek-OCR is registered as a custom multimodal model for vLLM through `DeepseekOCRForCausalLM`, which composes a vision stack and a language model that share Hugging Face checkpoints.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L257-L329】 The model instantiates a SAM-based encoder, a CLIP-L backbone, and an MLP projector to turn global and tiled image features into the 1280-dimensional embedding space expected by the language model.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L288-L309】【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L400-L445】 The projector also tracks special separator embeddings to delimit tiled image views before the features are merged with the text stream.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L303-L438】

### 1.2 Image Token Budgeting
The `DeepseekOCRProcessingInfo` helper estimates the number of image tokens required by combining base (global) views and tiled (local) crops, adapting the token count to the input aspect ratio and crop mode.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L50-L107】 This ensures prompts expand `<image>` placeholders into the correct number of embeddings, and exposes sizing information for dummy inputs when profiling or scheduling workloads.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L118-L147】【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L190-L229】

### 1.3 Vision Feature Extraction Pipeline
The processor first normalizes images, optionally crops them into tiles, and generates token IDs aligned with `<image>` markers using `DeepseekOCRProcessor` and `dynamic_preprocess`.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py†L45-L102】【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py†L330-L435】 During inference, `_pixel_values_to_embedding` pushes global and local patches through SAM and CLIP, stitches them with learned separators, and returns a flattened sequence for merging with text tokens.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L364-L465】 The runtime then injects these multimodal embeddings into the language model wherever `<image>` tokens occur, so subsequent decoding steps work on a unified context window.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L498-L553】

### 1.4 Weight Loading and Compatibility
Weights downloaded from Hugging Face are remapped so that shared checkpoints can initialize both the vision modules and the language backbone. Components tied to the vision stack keep their original prefixes, while language layers are rewritten under `language.` before being loaded with `AutoWeightsLoader` for vLLM compatibility.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L564-L582】

## 2. Design Overview

### 2.1 Configurable Inference Profiles
Runtime behavior is controlled via `config.py`, which defines presets for image and base resolutions, crop mode, concurrency, prompt templates, and tokenizer initialization. These settings allow operators to trade accuracy for memory footprint (e.g., "Tiny" through "Gundam" profiles) and to toggle prompt variants suited to OCR, layout parsing, or general description tasks.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py†L1-L37】【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py†L40-L42】

### 2.2 Processor Responsibilities
`DeepseekOCRProcessor` handles padding, normalization, tokenizer setup, and `<image>` token bookkeeping. It enforces left padding for batch alignment, injects pad tokens if they are missing, and produces structured outputs that include `input_ids`, `pixel_values`, tiled crops, and spatial metadata for downstream consumption.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py†L111-L188】【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py†L241-L329】 The processor also masks image token positions so loss computations can ignore them while retaining alignment between text and vision streams.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py†L424-L499】

### 2.3 Multimodal Integration in vLLM
The custom multimodal processor registered with vLLM translates prompts and image payloads into batched tensors, configures fields exposed to the engine (pixel values, crop indices), and replaces `<image>` markers with the computed token spans. This integration supports caching for common cases and ensures the engine requests the correct number of embeddings during speculative decoding.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L150-L255】

## 3. Core Business Logic, API Surface, and Request Flow

### 3.1 Single-Image Streaming Workflow
`run_dpsk_ocr_image.py` orchestrates end-to-end inference: it registers the model, loads and EXIF-corrects an input image, and tokenizes it via `DeepseekOCRProcessor` according to the configured prompt and crop mode.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py†L1-L57】【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py†L146-L219】 An asynchronous vLLM engine is instantiated with the DeepSeek-OCR architecture override, deterministic sampling, and an n-gram repeat suppression processor tailored to table tags.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py†L147-L199】 The script streams tokens as they are generated, captures the final markdown, and optionally saves both raw and post-processed outputs (including cropped figures and annotated bounding boxes) into the output directory.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py†L204-L303】

### 3.2 Batched and Document Workloads
The PDF and batch runners follow the same pattern: they register the custom model, construct an `LLM` instance with concurrency and memory limits from the shared config, and reuse `DeepseekOCRProcessor` plus the n-gram logits processor to prepare inputs. PDF inputs are rasterized page by page before batching, and results include optional figure crops and overlays for downstream consumption.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py†L1-L83】【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py†L84-L170】

### 3.3 API Interaction Surface
Clients interact with the system by supplying prompts and `multi_modal_data` payloads that contain processor-formatted image tensors. The engine expects `<image>` placeholders to appear in the prompt when images are present; otherwise it runs in text-only mode. Sampling parameters expose temperature, max token budget, and custom logits processors so downstream services can tailor latency and determinism requirements.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py†L177-L199】 The configuration module exposes the knobs (paths, prompt templates, crop strategies) that upstream services adjust to match deployment scenarios.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py†L8-L37】

