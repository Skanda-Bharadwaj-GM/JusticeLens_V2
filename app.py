import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image


BASE_MODEL_ID = "caidas/swin2SR-classical-sr-x2-64"
DEFAULT_CHECKPOINT = "models/lens_cloud_latest_checkpoint.pth"
HF_WEIGHTS_REPO = "skandab17/justice-lens-weights"
HF_WEIGHTS_FILENAME = "lens_cloud_latest_checkpoint.pth"
OUTPUT_DIR = Path("results/webapp_runs")
UPLOAD_DIR = OUTPUT_DIR / "uploads"
RESTORED_DIR = OUTPUT_DIR / "restored"
# Longest image side after resize. Lower on CUDA to reduce VRAM (Swin2SR attention is memory-heavy).
MAX_IMAGE_SIZE = 512
MAX_IMAGE_SIZE_CUDA = 384
WIN_TESSERACT_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def configure_tesseract(custom_cmd: Optional[str]) -> Optional[str]:
    """Point pytesseract at the Tesseract binary. Returns an error message or None."""
    import pytesseract

    cmd = (custom_cmd or "").strip() or os.environ.get("TESSERACT_CMD", "").strip()
    if cmd:
        if not os.path.isfile(cmd):
            return f"Tesseract not found at `{cmd}`."
        pytesseract.pytesseract.tesseract_cmd = cmd
        return None
    if os.name == "nt" and os.path.isfile(WIN_TESSERACT_DEFAULT):
        pytesseract.pytesseract.tesseract_cmd = WIN_TESSERACT_DEFAULT
    return None


def ocr_image(image: Image.Image, lang: str) -> tuple[str, Optional[str]]:
    try:
        import pytesseract
    except ImportError:
        return "", "Python package `pytesseract` is not installed. Add it to your environment."

    try:
        text = pytesseract.image_to_string(image.convert("RGB"), lang=lang.strip() or "eng")
        return text.strip(), None
    except Exception as e:
        return "", str(e)


def save_ocr_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if content else "", encoding="utf-8")


def get_device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_effective_max_image_side(device_name: str) -> int:
    """Max input side length; override with env JUSTICE_LENS_MAX_IMAGE_SIZE (positive int)."""
    raw = os.environ.get("JUSTICE_LENS_MAX_IMAGE_SIZE", "").strip()
    if raw.isdigit():
        return max(64, min(int(raw), 2048))
    if device_name == "cuda":
        return MAX_IMAGE_SIZE_CUDA
    return MAX_IMAGE_SIZE


def _checkpoint_mtime(path: str) -> float:
    return os.path.getmtime(path) if path and os.path.isfile(path) else -1.0


def ensure_checkpoint_from_hub(
    checkpoint_path: str,
    use_custom_weights: bool,
    download_if_missing: bool,
    hf_repo_id: str,
    hf_filename: str,
    hf_token: Optional[str],
) -> tuple[str, str]:
    """
    If custom weights are enabled and the local file is missing, download from Hugging Face.
    Returns (path_for_loading, status_note).
    """
    if not use_custom_weights or not checkpoint_path:
        return checkpoint_path, ""

    if os.path.isfile(checkpoint_path):
        return checkpoint_path, ""

    if not download_if_missing:
        return checkpoint_path, ""

    from huggingface_hub import hf_hub_download

    local_dir = str(Path(checkpoint_path).parent)
    os.makedirs(local_dir, exist_ok=True)
    try:
        saved_path = hf_hub_download(
            repo_id=hf_repo_id.strip(),
            filename=hf_filename.strip(),
            local_dir=local_dir,
            token=(hf_token.strip() if hf_token else None) or os.environ.get("HF_TOKEN"),
        )
    except Exception as e:
        return checkpoint_path, f"Hugging Face download failed: {e}"

    if saved_path and os.path.isfile(saved_path):
        return saved_path, f"Downloaded weights from `{hf_repo_id}` → `{saved_path}`."
    if os.path.isfile(checkpoint_path):
        return checkpoint_path, f"Downloaded weights from `{hf_repo_id}` → `{checkpoint_path}`."
    expected = str(Path(local_dir) / Path(hf_filename).name)
    if os.path.isfile(expected):
        return expected, f"Downloaded weights from `{hf_repo_id}` → `{expected}`."
    return checkpoint_path, "Download reported success but file not found at expected path."


@st.cache_resource(show_spinner=False)
def load_pipeline(
    checkpoint_path: str,
    use_custom_weights: bool,
    device_name: str,
    checkpoint_mtime: float,
):
    import torch
    from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

    processor = Swin2SRImageProcessor.from_pretrained(BASE_MODEL_ID)
    model = Swin2SRForImageSuperResolution.from_pretrained(BASE_MODEL_ID)

    status_msg = "Running with base pretrained weights."
    if use_custom_weights and checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device_name)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        status_msg = (
            f"Loaded custom weights: `{checkpoint_path}` "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    elif use_custom_weights:
        status_msg = f"Checkpoint not found at `{checkpoint_path}`. Using base weights."

    model.to(device_name)
    model.eval()
    return processor, model, status_msg


def prepare_image(image: Image.Image, max_side: Optional[int] = None) -> Image.Image:
    image = image.convert("RGB")
    cap = max_side if max_side is not None else MAX_IMAGE_SIZE
    if max(image.size) <= cap:
        return image

    # Keep aspect ratio and cap dimensions to avoid out-of-memory issues.
    image.thumbnail((cap, cap))
    return image


def _forward_deblur(
    resized_image: Image.Image,
    processor,
    model,
    device_name: str,
) -> Image.Image:
    import torch

    inputs = processor(images=resized_image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device_name)

    with torch.inference_mode():
        if device_name == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(pixel_values=pixel_values)
        else:
            outputs = model(pixel_values=pixel_values)

    output_tensor = outputs.reconstruction.squeeze(0).detach().float().cpu().clamp(0, 1)
    output_array = (output_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(output_array)


def deblur_image(
    image: Image.Image,
    processor,
    model,
    device_name: str,
) -> Image.Image:
    import torch

    base = get_effective_max_image_side(device_name)
    fallbacks = [384, 320, 256, 224, 192]
    ordered = [base] + [s for s in fallbacks if s < base]
    seen: set[int] = set()
    sizes: list[int] = []
    for s in ordered:
        if s not in seen:
            seen.add(s)
            sizes.append(s)

    last_err: Optional[BaseException] = None
    for max_side in sizes:
        resized_image = prepare_image(image, max_side=max_side)
        if device_name == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            return _forward_deblur(resized_image, processor, model, device_name)
        except torch.OutOfMemoryError as e:
            last_err = e
            if device_name == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    if last_err is not None:
        raise RuntimeError(
            "CUDA ran out of memory even after shrinking the input. "
            "Close other GPU apps, set JUSTICE_LENS_MAX_IMAGE_SIZE to a smaller value (e.g. 256), "
            "or run on CPU by setting CUDA_VISIBLE_DEVICES empty before starting Streamlit."
        ) from last_err
    raise RuntimeError("Deblur failed unexpectedly.")




def save_result_images(uploaded_bytes: bytes, uploaded_name: str, restored_img: Image.Image):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESTORED_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(uploaded_name).stem.replace(" ", "_")
    input_path = UPLOAD_DIR / f"{safe_name}_{timestamp}_input.jpg"
    output_path = RESTORED_DIR / f"{safe_name}_{timestamp}_restored.jpg"

    with open(input_path, "wb") as f:
        f.write(uploaded_bytes)

    restored_img.save(output_path, quality=95)
    return input_path, output_path


def infer_once(
    uploaded_file,
    processor,
    model,
    device_name: str,
) -> Optional[tuple[Image.Image, Image.Image, Path, Path]]:
    if not uploaded_file:
        return None

    file_bytes = uploaded_file.getvalue()
    original_img = Image.open(uploaded_file).convert("RGB")
    restored_img = deblur_image(original_img, processor, model, device_name)
    input_path, output_path = save_result_images(file_bytes, uploaded_file.name, restored_img)
    return original_img, restored_img, input_path, output_path


def render_header():
    st.set_page_config(page_title="Justice Lens - Evidence Deblur", layout="wide")
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2.1rem;
                font-weight: 700;
                margin-bottom: 0.2rem;
                color: #d2e3ff;
            }
            .subtitle {
                font-size: 1rem;
                color: #9db2ce;
                margin-bottom: 1.2rem;
            }
            .panel {
                border: 1px solid #2f3f57;
                border-radius: 12px;
                padding: 14px;
                background: #111a28;
            }
            .note {
                color: #b6c8e6;
                font-size: 0.92rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='main-title'>Justice Lens Evidence Deblur Console</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Upload blurred legal/document evidence and restore clarity for visual review.</div>",
        unsafe_allow_html=True,
    )


def main():
    render_header()
    device_name = get_device_name()

    with st.sidebar:
        st.header("Inference Setup")
        st.write(f"Device: `{device_name}`")
        if device_name == "cuda":
            st.caption(
                f"Input is resized to max side **{get_effective_max_image_side(device_name)}** px on GPU "
                f"(set env `JUSTICE_LENS_MAX_IMAGE_SIZE` to change). OOM retries use smaller sizes."
            )
        use_custom_weights = st.toggle("Use custom checkpoint", value=True)
        checkpoint_path = st.text_input("Local checkpoint path", value=DEFAULT_CHECKPOINT)
        st.caption("Matches `download_weights.py`: saved under `models/`.")
        download_if_missing = st.toggle(
            "Download from Hugging Face if file missing",
            value=True,
        )
        hf_repo_id = st.text_input("Hugging Face repo", value=HF_WEIGHTS_REPO)
        hf_filename = st.text_input("Weights file on repo", value=HF_WEIGHTS_FILENAME)
        hf_token = st.text_input(
            "HF token (optional; or set env HF_TOKEN)",
            type="password",
            help="Needed only if the repo is private or gated.",
        )
        st.caption("If local file is missing, weights are fetched from the repo above.")
        st.caption("Model loads after you click Deblur.")

        st.divider()
        st.subheader("Text extraction (OCR)")
        run_ocr = st.toggle("Extract text after deblur", value=True)
        ocr_lang = st.text_input("Tesseract language(s)", value="eng", help="e.g. eng, or hin+eng")
        tesseract_cmd = st.text_input(
            "Tesseract executable (optional)",
            value="",
            help="Windows example: C:\\Program Files\\Tesseract-OCR\\tesseract.exe. Or set env TESSERACT_CMD.",
        )
        st.caption("Requires the Tesseract OCR engine installed (see tesseract-ocr/tesseract on GitHub).")

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload blurred evidence image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=False,
    )
    run_button = st.button("Deblur Evidence", type="primary", disabled=uploaded_file is None)
    st.markdown("</div>", unsafe_allow_html=True)

    if run_button and uploaded_file is not None:
        resolved_ckpt, hub_note = ensure_checkpoint_from_hub(
            checkpoint_path=checkpoint_path,
            use_custom_weights=use_custom_weights,
            download_if_missing=download_if_missing,
            hf_repo_id=hf_repo_id,
            hf_filename=hf_filename,
            hf_token=hf_token or None,
        )
        if hub_note:
            if hub_note.startswith("Hugging Face download failed"):
                st.warning(hub_note)
            elif "Downloaded" in hub_note:
                st.success(hub_note)
            else:
                st.info(hub_note)

        ckpt_mtime = _checkpoint_mtime(resolved_ckpt)
        with st.spinner("Loading model... this can take time on first run."):
            processor, model, model_status = load_pipeline(
                checkpoint_path=resolved_ckpt,
                use_custom_weights=use_custom_weights,
                device_name=device_name,
                checkpoint_mtime=ckpt_mtime,
            )
        st.info(model_status)

        with st.spinner("Reconstructing document..."):
            result = infer_once(uploaded_file, processor, model, device_name)

        if result is None:
            st.error("No image received.")
            return

        original_img, restored_img, input_path, output_path = result
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Evidence")
            st.image(original_img, use_container_width=True)
        with col2:
            st.subheader("Deblurred Result")
            st.image(restored_img, use_container_width=True)

        st.success("Deblurring complete. Files saved to project folder.")
        st.markdown(
            f"<div class='note'>Saved input: <code>{input_path.as_posix()}</code><br>"
            f"Saved output: <code>{output_path.as_posix()}</code></div>",
            unsafe_allow_html=True,
        )

        if run_ocr:
            st.divider()
            st.subheader("Extracted text")
            cfg_err = configure_tesseract(tesseract_cmd or None)
            if cfg_err:
                st.warning(cfg_err)

            with st.spinner("Running OCR on deblurred image..."):
                text_restored, err_r = ocr_image(restored_img, ocr_lang)
            if err_r:
                st.error(f"OCR failed: {err_r}")
            else:
                display_r = text_restored if text_restored else "(No text detected.)"
                st.markdown("**From deblurred image**")
                st.text_area(
                    "Deblurred OCR",
                    value=display_r,
                    height=220,
                    key="ocr_restored",
                    label_visibility="collapsed",
                )
                ocr_out = output_path.with_name(f"{output_path.stem}_ocr.txt")
                save_ocr_text(ocr_out, text_restored)
                st.caption(f"Transcript saved: `{ocr_out.as_posix()}`")

            with st.expander("OCR from original upload (before deblur)", expanded=False):
                with st.spinner("Running OCR on original..."):
                    text_orig, err_o = ocr_image(original_img, ocr_lang)
                if err_o:
                    st.error(err_o)
                else:
                    display_o = text_orig if text_orig else "(No text detected.)"
                    st.text_area(
                        "Original OCR",
                        value=display_o,
                        height=160,
                        key="ocr_original",
                        label_visibility="collapsed",
                    )
                    ocr_in = input_path.with_name(f"{input_path.stem}_ocr.txt")
                    save_ocr_text(ocr_in, text_orig)
                    st.caption(f"Transcript saved: `{ocr_in.as_posix()}`")


if __name__ == "__main__":
    main()
