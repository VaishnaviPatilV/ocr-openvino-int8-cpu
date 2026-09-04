# ============================================================
# glm_cpu.py
#
# Generic Document / Image OCR API
# GLM-OCR OpenVINO INT8 - CPU only - Windows
#
# Pipeline:
# Image / Base64 / File Upload
#          |
#          v
#       PIL Image
#          |
#          v
# Conservative preprocessing
#          |
#          v
# GLM-OCR OpenVINO INT8
#          |
#          v
# Generic OCR text
#          |
#          v
# Save .txt using original filename
#          |
#          v
# FastAPI JSON response
#
# No YOLO
# No object detection
# No container detection
# No ISO validation
# No CUDA inference
# ============================================================

# ============================================================
# 1. CPU ENVIRONMENT
# ============================================================

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"


# ============================================================
# 2. IMPORTS
# ============================================================

import base64
import gc
import json
import logging
import threading
import time
import uuid

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from PIL import Image

import openvino as ov

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
)

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from transformers import AutoProcessor
from optimum.intel import OVModelForVisualCausalLM


# ============================================================
# 3. CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# GLM-OCR OpenVINO INT8 model
# ------------------------------------------------------------

GLM_OV_PATH = r"D:\GLM_OCR\GLM_OCR_INT8"

GLM_PROCESSOR_PATH = GLM_OV_PATH


# ------------------------------------------------------------
# OCR settings
# ------------------------------------------------------------

MAX_NEW_TOKENS = 512

MAX_CONCURRENT_INFERENCE = 1


# ------------------------------------------------------------
# Image limits
# ------------------------------------------------------------

MIN_IMAGE_DIM = 32
MAX_IMAGE_DIM = 4096


# ------------------------------------------------------------
# Supported image extensions
# ------------------------------------------------------------

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------

HOST = "0.0.0.0"
PORT = 8082


# ============================================================
# 4. RESULT DIRECTORIES
# ============================================================

BASE_DIR = "ocr_results"

RECEIVED_DIR = os.path.join(
    BASE_DIR,
    "received",
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
)

LOG_DIR = os.path.join(
    BASE_DIR,
    "logs",
)


for directory in [
    RECEIVED_DIR,
    RESULTS_DIR,
    LOG_DIR,
]:
    os.makedirs(
        directory,
        exist_ok=True,
    )


# ============================================================
# 5. LOGGING
# ============================================================

LOG_FILE = os.path.join(
    LOG_DIR,
    f"ocr_{datetime.now().strftime('%Y%m%d')}.log",
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("glm_ocr")

_log_lock = threading.Lock()


def log_event(
    request_id: str,
    step: str,
    data: dict,
):
    """
    Structured logging.
    Never logs image bytes.
    """

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]

    log_obj = {
        "time": timestamp,
        "request_id": request_id,
        "step": step,
        **data,
    }

    message = json.dumps(
        log_obj,
        default=str,
        ensure_ascii=False,
    )

    with _log_lock:

        with open(
            LOG_FILE,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                message + "\n"
            )

    logger.info(message)


# ============================================================
# 6. STARTUP
# ============================================================

print("\n" + "=" * 70)

print(
    "[GLM-OCR-OV] Loading INT8 model..."
)

print(
    f"[GLM-OCR-OV] Model: {GLM_OV_PATH}"
)

print(
    f"[GLM-OCR-OV] Processor: {GLM_PROCESSOR_PATH}"
)

print(
    "[GLM-OCR-OV] CPU-only mode: ENABLED"
)


# ============================================================
# 7. MODEL EXISTENCE CHECK
# ============================================================

if not os.path.isdir(GLM_OV_PATH):

    raise FileNotFoundError(
        f"GLM INT8 directory not found: {GLM_OV_PATH}"
    )


# ============================================================
# 8. LOAD PROCESSOR
# ============================================================

glm_processor = None
glm_model = None


try:

    glm_processor = AutoProcessor.from_pretrained(
        GLM_PROCESSOR_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    print(
        "[GLM-OCR-OV] Processor loaded."
    )

except Exception as e:

    print(
        f"[GLM-OCR-OV] Failed to load processor: {e}"
    )

    raise


# ============================================================
# 9. LOAD OPENVINO MODEL
# ============================================================

try:

    glm_model = (
        OVModelForVisualCausalLM.from_pretrained(
            GLM_OV_PATH,
            device="CPU",
            trust_remote_code=True,
            local_files_only=True,
        )
    )

    print(
        "[GLM-OCR-OV] Complete INT8 model loaded on CPU."
    )

except Exception as e:

    print(
        f"[GLM-OCR-OV] Failed to load model: {e}"
    )

    raise


# ============================================================
# 10. OPENVINO DEVICE CHECK
# ============================================================

try:

    ov_core = ov.Core()

    available_devices = list(
        ov_core.available_devices
    )

    print(
        "[GLM-OCR-OV] Available devices:",
        available_devices,
    )

    if "CPU" not in available_devices:

        raise RuntimeError(
            "OpenVINO CPU device is not available."
        )

except Exception as e:

    print(
        "[OPENVINO] Device check failed:",
        e,
    )

    available_devices = []


print(
    "[OCR] Generic image/document OCR mode enabled."
)

print("=" * 70 + "\n")


# ============================================================
# 11. OCR SEMAPHORE
# ============================================================

glm_semaphore = threading.Semaphore(
    MAX_CONCURRENT_INFERENCE
)


# ============================================================
# 12. REQUEST MODELS
# ============================================================

class OcrRequest(BaseModel):

    image_base64: Optional[str] = None

    filename: Optional[str] = None


class OcrBatchRequest(BaseModel):

    images: List[str] = []

    filenames: Optional[List[str]] = None


# ============================================================
# 13. UTILITY FUNCTIONS
# ============================================================

def today_folder(base: str) -> str:

    path = os.path.join(
        base,
        datetime.now().strftime(
            "%Y-%m-%d"
        ),
    )

    os.makedirs(
        path,
        exist_ok=True,
    )

    return path


def generate_request_id() -> str:

    return (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S_"
        )
        + uuid.uuid4().hex[:8]
    )


# ============================================================
# 14. BASE64
# ============================================================

def strip_data_uri(
    b64: str,
) -> str:

    if b64.startswith("data:"):

        return b64.split(
            ",",
            1,
        )[1]

    return b64


def decode_base64_image(
    b64: str,
) -> Image.Image:

    raw = strip_data_uri(
        b64
    ).strip()

    try:

        binary = base64.b64decode(
            raw,
            validate=True,
        )

    except Exception:

        raise ValueError(
            "Invalid Base64 payload"
        )

    if not binary:

        raise ValueError(
            "Empty image payload"
        )

    try:

        image = Image.open(
            BytesIO(binary)
        )

        image.load()

    except Exception:

        raise ValueError(
            "Corrupted or unsupported image data"
        )

    return image


# ============================================================
# 15. EXTENSION VALIDATION
# ============================================================

def validate_extension(
    filename: Optional[str],
) -> Optional[str]:

    if not filename:

        return None

    _, ext = os.path.splitext(
        filename
    )

    ext = ext.lower()

    if ext not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: "
                f"{sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    return ext


# ============================================================
# 16. SAFE TEXT FILE NAME
# ============================================================

def get_unique_txt_path(
    directory: str,
    filename: str,
) -> str:

    safe_name = Path(
        filename or "image.jpg"
    ).name

    stem = Path(
        safe_name
    ).stem or "image"

    illegal_chars = (
        '<>:"/\\|?*\x00'
    )

    for ch in illegal_chars:

        stem = stem.replace(
            ch,
            "_",
        )

    stem = stem.strip(
        ". "
    )

    if not stem:

        stem = "image"

    if len(stem) > 180:

        stem = stem[:180]

    candidate = os.path.join(
        directory,
        f"{stem}.txt",
    )

    if not os.path.exists(candidate):

        return candidate

    counter = 1

    while True:

        candidate = os.path.join(
            directory,
            f"{stem}_{counter}.txt",
        )

        if not os.path.exists(candidate):

            return candidate

        counter += 1


# ============================================================
# 17. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    pil: Image.Image,
) -> Image.Image:

    try:

        # ----------------------------------------------------
        # Normalize image mode
        # ----------------------------------------------------

        if pil.mode == "P":

            pil = pil.convert(
                "RGBA"
            )

        if pil.mode in (
            "RGBA",
            "LA",
        ):

            background = Image.new(
                "RGBA",
                pil.size,
                (
                    255,
                    255,
                    255,
                    255,
                ),
            )

            pil = Image.alpha_composite(
                background,
                pil.convert("RGBA"),
            ).convert("RGB")

        elif pil.mode == "L":

            pil = pil.convert(
                "RGB"
            )

        elif pil.mode != "RGB":

            pil = pil.convert(
                "RGB"
            )

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        width, height = pil.size

        if width < 1 or height < 1:

            raise ValueError(
                "Image has zero dimension"
            )

        # ----------------------------------------------------
        # Minimum size
        # ----------------------------------------------------

        if min(
            width,
            height,
        ) < MIN_IMAGE_DIM:

            scale = (
                MIN_IMAGE_DIM
                / float(
                    min(
                        width,
                        height,
                    )
                )
            )

            new_width = max(
                1,
                int(
                    width * scale
                ),
            )

            new_height = max(
                1,
                int(
                    height * scale
                ),
            )

            pil = pil.resize(
                (
                    new_width,
                    new_height,
                ),
                Image.BICUBIC,
            )

        # ----------------------------------------------------
        # Maximum size
        # ----------------------------------------------------

        width, height = pil.size

        if max(
            width,
            height,
        ) > MAX_IMAGE_DIM:

            scale = (
                MAX_IMAGE_DIM
                / float(
                    max(
                        width,
                        height,
                    )
                )
            )

            new_width = max(
                1,
                int(
                    width * scale
                ),
            )

            new_height = max(
                1,
                int(
                    height * scale
                ),
            )

            pil = pil.resize(
                (
                    new_width,
                    new_height,
                ),
                Image.LANCZOS,
            )

        return pil

    except Exception as e:

        raise ValueError(
            f"Image preprocessing failed: {e}"
        )


# ============================================================
# 18. SAVE IMAGE
# ============================================================

def save_received_image(
    pil: Image.Image,
    request_id: str,
) -> str:

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    path = os.path.join(
        today_folder(
            RECEIVED_DIR
        ),
        f"{request_id}_{timestamp}.jpg",
    )

    try:

        pil.convert(
            "RGB"
        ).save(
            path,
            "JPEG",
            quality=90,
        )

    except Exception as e:

        logger.warning(
            f"[SAVE_IMAGE] Failed: {e}"
        )

    return path


# ============================================================
# 19. SAVE OCR TEXT
# ============================================================

def save_result_text(
    text: str,
    filename: str,
) -> str:

    directory = today_folder(
        RESULTS_DIR
    )

    path = get_unique_txt_path(
        directory,
        filename,
    )

    try:

        with open(
            path,
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            f.write(
                text
            )

    except Exception as e:

        logger.error(
            f"[SAVE_TEXT] Failed: {e}"
        )

        return ""

    logger.info(
        f"[SAVE_TEXT] Saved OCR text: {path}"
    )

    return path


# ============================================================
# 20. GLM-OCR PROMPT
# ============================================================

# Official GLM-OCR document parsing uses the
# "Text Recognition:" task prompt.

GLM_OCR_PROMPT = (
    "Text Recognition:"
)


# ============================================================
# 21. GLM-OCR INFERENCE
# ============================================================

def glm_ocr(
    pil: Image.Image,
) -> str:

    if glm_processor is None:

        raise RuntimeError(
            "GLM-OCR processor is not loaded"
        )

    if glm_model is None:

        raise RuntimeError(
            "GLM-OCR model is not loaded"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use the multimodal chat template directly.
    # This is the critical difference from the old code.
    # --------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil,
                },
                {
                    "type": "text",
                    "text": GLM_OCR_PROMPT,
                },
            ],
        }
    ]

    try:

        logger.info(
            "[GLM-OCR] Preparing multimodal inputs..."
        )

        # ----------------------------------------------------
        # Build multimodal tensors directly
        # ----------------------------------------------------

        inputs = (
            glm_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        # ----------------------------------------------------
        # Remove token_type_ids if generated.
        # GLM-OCR official Transformers examples do not
        # require this field for generation.
        # ----------------------------------------------------

        if "token_type_ids" in inputs:

            inputs.pop(
                "token_type_ids"
            )

        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        logger.info(
            "[GLM-OCR] Input tensors:"
        )

        for key, value in inputs.items():

            logger.info(
                f"  {key}: "
                f"shape={getattr(value, 'shape', None)} "
                f"dtype={getattr(value, 'dtype', None)}"
            )

        # ----------------------------------------------------
        # CPU OpenVINO inference
        # ----------------------------------------------------

        with glm_semaphore:

            start = time.perf_counter()

            output = glm_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

        logger.info(
            f"[GLM-OCR-OV] Inference time: "
            f"{round(elapsed, 3)} sec"
        )

        # ----------------------------------------------------
        # Get generated IDs
        # ----------------------------------------------------

        if hasattr(
            output,
            "sequences",
        ):

            output_ids = (
                output.sequences
            )

        else:

            output_ids = output

        logger.info(
            "[GLM-OCR] Output shape: "
            f"{getattr(output_ids, 'shape', None)}"
        )

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        try:

            if (
                "input_ids"
                in inputs
            ):

                input_ids = inputs[
                    "input_ids"
                ]

                input_shape = getattr(
                    input_ids,
                    "shape",
                    None,
                )

                logger.info(
                    "[GLM-OCR] input_ids shape: "
                    f"{input_shape}"
                )

                # Usually:
                # [batch, sequence]
                #
                # Use the final dimension for
                # sequence length.

                input_length = (
                    input_shape[-1]
                )

            else:

                input_length = 0

            # ------------------------------------------------
            # Normal generation shape:
            # [batch, sequence]
            # ------------------------------------------------

            if (
                hasattr(
                    output_ids,
                    "ndim",
                )
                and output_ids.ndim == 2
            ):

                generated_ids = (
                    output_ids[
                        :,
                        input_length:,
                    ]
                )

            # ------------------------------------------------
            # Single vector:
            # [sequence]
            # ------------------------------------------------

            elif (
                hasattr(
                    output_ids,
                    "ndim",
                )
                and output_ids.ndim == 1
            ):

                generated_ids = (
                    output_ids[
                        input_length:
                    ]
                )

            else:

                generated_ids = (
                    output_ids
                )

            # ------------------------------------------------
            # Decode
            # ------------------------------------------------

            full_text = (
                glm_processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            )

        except Exception as decode_error:

            logger.warning(
                "[GLM-OCR] Decode fallback: "
                f"{decode_error}"
            )

            full_text = (
                glm_processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            )

        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------

        full_text = (
            full_text or ""
        ).strip()

        logger.info(
            f"[GLM-OCR] OCR text length: "
            f"{len(full_text)}"
        )

        logger.info(
            f"[GLM-OCR] OCR result:\n{full_text}"
        )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        del inputs
        del output

        if "generated_ids" in locals():

            del generated_ids

        gc.collect()

        return full_text

    except Exception as e:

        logger.exception(
            "[GLM-OCR] Inference failed"
        )

        raise RuntimeError(
            f"OCR inference failed: {e}"
        )


# ============================================================
# 22. COMPLETE OCR PIPELINE
# ============================================================

def run_ocr_pipeline(
    pil: Image.Image,
    request_id: str,
    filename: Optional[str] = None,
) -> dict:

    pipeline_start = time.perf_counter()

    fname = (
        filename
        or "image.jpg"
    )

    txt_path = ""

    try:

        # ----------------------------------------------------
        # 1. Save original
        # ----------------------------------------------------

        save_received_image(
            pil,
            request_id,
        )

        # ----------------------------------------------------
        # 2. Preprocess
        # ----------------------------------------------------

        pil_clean = preprocess_image(
            pil
        )

        # ----------------------------------------------------
        # 3. OCR
        # ----------------------------------------------------

        text = glm_ocr(
            pil_clean
        )

        # ----------------------------------------------------
        # 4. Save text
        # ----------------------------------------------------

        txt_path = save_result_text(
            text,
            fname,
        )

        # ----------------------------------------------------
        # 5. Timing
        # ----------------------------------------------------

        total_time = (
            time.perf_counter()
            - pipeline_start
        )

        # ----------------------------------------------------
        # 6. Log
        # ----------------------------------------------------

        log_event(
            request_id,
            "OCR_COMPLETE",
            {
                "filename": fname,
                "text_length": len(text),
                "text_file": txt_path,
                "processing_time_sec":
                    round(
                        total_time,
                        3,
                    ),
                "status":
                    "success"
                    if text
                    else "empty",
            },
        )

        # ----------------------------------------------------
        # 7. Response
        # ----------------------------------------------------

        text_file_response = (
            txt_path.replace(
                "\\",
                "/",
            )
            if txt_path
            else None
        )

        return {
            "success": bool(text),
            "filename": fname,
            "text": text,
            "text_file":
                text_file_response,
            "processing_time":
                round(
                    total_time,
                    3,
                ),
        }

    except Exception as e:

        total_time = (
            time.perf_counter()
            - pipeline_start
        )

        log_event(
            request_id,
            "OCR_ERROR",
            {
                "filename": fname,
                "error": str(e),
                "processing_time_sec":
                    round(
                        total_time,
                        3,
                    ),
            },
        )

        return {
            "success": False,
            "filename": fname,
            "text": "",
            "text_file": None,
            "processing_time":
                round(
                    total_time,
                    3,
                ),
            "error": str(e),
        }

    finally:

        try:

            del pil

            if "pil_clean" in locals():

                del pil_clean

        except Exception:

            pass

        gc.collect()


# ============================================================
# 23. FASTAPI
# ============================================================

app = FastAPI(
    title=(
        "GLM-OCR Generic "
        "Document/Image OCR API"
    ),
    version="2.0",
)


# ============================================================
# 24. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",

        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],

    allow_credentials=False,

    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],

    allow_headers=[
        "*",
    ],
)


# ============================================================
# 25. HEALTH
# ============================================================

@app.get(
    "/api/health"
)
def health():

    core_devices = []

    try:

        core = ov.Core()

        core_devices = list(
            core.available_devices
        )

    except Exception:

        pass

    return {
        "status": "running",

        "device": "CPU",

        "openvino_devices":
            core_devices,

        "glm_ocr_int8_loaded":
            glm_model is not None,

        "max_new_tokens":
            MAX_NEW_TOKENS,

        "max_concurrent_inference":
            MAX_CONCURRENT_INFERENCE,
    }


# ============================================================
# 26. BASE64 OCR
# ============================================================

@app.post(
    "/api/ocr"
)
def ocr_base64(
    req: OcrRequest,
):

    if not req.image_base64:

        raise HTTPException(
            status_code=400,
            detail=(
                "image_base64 is required"
            ),
        )

    request_id = (
        generate_request_id()
    )

    filename = (
        req.filename
        or "image.jpg"
    )

    try:

        pil = decode_base64_image(
            req.image_base64
        )

    except ValueError as e:

        log_event(
            request_id,
            "DECODE_ERROR",
            {
                "filename": filename,
                "error": str(e),
            },
        )

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    return run_ocr_pipeline(
        pil,
        request_id,
        filename,
    )


# ============================================================
# 27. FILE OCR
# ============================================================

@app.post(
    "/api/ocr/file"
)
async def ocr_file(
    file: UploadFile = File(...),
):

    filename = (
        file.filename
        or "upload.jpg"
    )

    validate_extension(
        filename
    )

    request_id = (
        generate_request_id()
    )

    try:

        raw = await file.read()

    except Exception:

        raise HTTPException(
            status_code=400,
            detail=(
                "Failed to read uploaded file"
            ),
        )

    if not raw:

        raise HTTPException(
            status_code=400,
            detail="Empty file payload",
        )

    try:

        pil = Image.open(
            BytesIO(raw)
        )

        pil.load()

    except Exception:

        raise HTTPException(
            status_code=400,
            detail=(
                "Corrupted or unsupported image"
            ),
        )

    return run_ocr_pipeline(
        pil,
        request_id,
        filename,
    )


# ============================================================
# 28. BATCH OCR
# ============================================================

@app.post(
    "/api/ocr/batch"
)
def ocr_batch(
    req: OcrBatchRequest,
):

    if not req.images:

        raise HTTPException(
            status_code=400,
            detail=(
                "images list is required"
            ),
        )

    filenames = (
        req.filenames
        or []
    )

    results = []

    batch_start = (
        time.perf_counter()
    )

    for idx, b64 in enumerate(
        req.images
    ):

        request_id = (
            generate_request_id()
        )

        filename = (
            filenames[idx]
            if idx < len(filenames)
            else f"image_{idx + 1}.jpg"
        )

        try:

            pil = decode_base64_image(
                b64
            )

        except ValueError as e:

            log_event(
                request_id,
                "DECODE_ERROR",
                {
                    "filename": filename,
                    "error": str(e),
                },
            )

            results.append(
                {
                    "success": False,
                    "filename": filename,
                    "text": "",
                    "text_file": None,
                    "processing_time": 0.0,
                    "error": str(e),
                }
            )

            continue

        result = run_ocr_pipeline(
            pil,
            request_id,
            filename,
        )

        results.append(
            result
        )

    total_time = (
        time.perf_counter()
        - batch_start
    )

    return {
        "success": True,
        "results": results,
        "total_processing_time":
            round(
                total_time,
                3,
            ),
    }


# ============================================================
# 29. RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
    )