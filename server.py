"""
FastAPI server for Coqui XTTS v2 text-to-speech service.

This server loads the XTTS model on startup and provides HTTP endpoints
for text-to-speech synthesis, including real-time streaming.
"""

# Standard library imports
import asyncio
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

# Third-party imports
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Patch torch.load to use weights_only=False for XTTS checkpoint loading
# PyTorch 2.6+ defaults to weights_only=True for security, but XTTS checkpoints
# contain custom classes that need to be loaded
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    """Patch torch.load to allow loading XTTS checkpoints."""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
xtts_model = None  # Direct XTTS model instance
xtts_config = None
tts_device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model loading and cleanup."""
    try:
        load_xtts_model()
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down XTTS server")


# Initialize FastAPI app
app = FastAPI(
    title="XTTS TTS Service",
    version="1.1.0",
    lifespan=lifespan,
)

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when allow_origins is "*"
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly list methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["X-Audio-Format", "X-Audio-Encoding", "X-Audio-Sample-Rate", "X-Audio-Channels"],
)


@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    """Handle CORS preflight requests."""
    return {}


class TTSRequest(BaseModel):
    """Request model for TTS synthesis."""
    text: str
    voice: str = "en_US-lessac-medium"  # Legacy field (not used)
    speed: float = 1.0  # Playback speed multiplier (not yet implemented)
    language: Optional[str] = None  # Language code (e.g., "en", "es", "fr")
    speaker_id: Optional[str] = None  # Built-in speaker name (e.g., "Daisy Studious", "Gracie Wise")
    stream_chunk_size: Optional[int] = 20  # Chunk size for streaming (smaller = lower latency)
    # Note: 58 built-in speakers available. See speakers_xtts.pth for full list.


def load_xtts_model():
    """Load the XTTS model on startup with auto-download support."""
    global xtts_model, xtts_config, tts_device

    model_path = os.getenv("TTS_MODEL_PATH", "")

    # Accept XTTS license automatically (CPML non-commercial license)
    os.environ["COQUI_TOS_AGREED"] = "1"

    logger.info(f"CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")
    tts_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {tts_device}")

    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.utils.manage import ModelManager

        # If model path provided and exists, load from there
        if model_path and Path(model_path).exists():
            model_dir = Path(model_path)
            logger.info(f"Loading XTTS model from local path: {model_path}")
        else:
            # Auto-download using TTS ModelManager
            logger.info("Model path not found, auto-downloading XTTS v2...")
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            manager = ModelManager(progress_bar=True)
            model_path_result, config_path, _ = manager.download_model(model_name)
            # model_path_result is the model.pth file, config_path is config.json
            model_dir = Path(config_path).parent
            logger.info(f"Model downloaded to: {model_dir}")

        # Check for required files
        required_files = ["config.json", "model.pth"]
        for file in required_files:
            if not (model_dir / file).exists():
                raise RuntimeError(f"Required model file not found: {model_dir / file}")

        # Load config
        config = XttsConfig()
        config.load_json(str(model_dir / "config.json"))

        # Initialize model from config
        model_instance = Xtts.init_from_config(config)

        # Load checkpoint
        checkpoint_dir = str(model_dir)
        checkpoint_path = str(model_dir / "model.pth")
        vocab_path = str(model_dir / "vocab.json")
        speaker_file_path = str(model_dir / "speakers_xtts.pth")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        model_instance.load_checkpoint(
            config,
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            speaker_file_path=speaker_file_path,
            use_deepspeed=False,
        )

        if tts_device == "cuda":
            model_instance.cuda()
        else:
            num_threads = int(os.getenv("TORCH_NUM_THREADS", "4"))
            torch.set_num_threads(num_threads)
            logger.info(f"CPU mode: Using {num_threads} threads for inference")
            model_instance.eval()

        xtts_model = model_instance
        xtts_config = config

        logger.info("XTTS model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load XTTS model: {e}", exc_info=True)
        raise


# Reference audio path for voice cloning
REFERENCE_AUDIO_PATH = os.getenv("REFERENCE_AUDIO_PATH", "ref.mp3")

# Cache for cloned voice latents
_cloned_voice_cache = {}


def get_cloned_voice_latents(reference_audio: str):
    """
    Get speaker conditioning latents from a reference audio file (voice cloning).

    Returns:
        tuple: (gpt_cond_latent, speaker_embedding)
    """
    global _cloned_voice_cache

    # Check cache first
    if reference_audio in _cloned_voice_cache:
        logger.info(f"Using cached latents for: {reference_audio}")
        return _cloned_voice_cache[reference_audio]

    if not Path(reference_audio).exists():
        raise ValueError(f"Reference audio not found: {reference_audio}")

    logger.info(f"Computing voice latents from: {reference_audio}")

    # Compute conditioning latents from reference audio
    gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
        audio_path=reference_audio
    )

    # Cache the result
    _cloned_voice_cache[reference_audio] = (gpt_cond_latent, speaker_embedding)

    return gpt_cond_latent, speaker_embedding


def get_speaker_latents(speaker_id: str):
    """
    Get speaker conditioning latents for streaming inference.

    Returns:
        tuple: (gpt_cond_latent, speaker_embedding)
    """
    if not hasattr(xtts_model, 'speaker_manager') or xtts_model.speaker_manager is None:
        raise ValueError("Speaker manager not available")

    if speaker_id not in xtts_model.speaker_manager.speakers:
        available = list(xtts_model.speaker_manager.speakers.keys())[:5]
        raise ValueError(f"Speaker '{speaker_id}' not found. Available: {available}...")

    speaker_data = xtts_model.speaker_manager.speakers[speaker_id]

    gpt_cond_latent = speaker_data.get("gpt_cond_latent")
    speaker_embedding = speaker_data.get("speaker_embedding")

    # Convert to tensors if needed and move to correct device
    if isinstance(gpt_cond_latent, np.ndarray):
        gpt_cond_latent = torch.from_numpy(gpt_cond_latent)
    if isinstance(speaker_embedding, np.ndarray):
        speaker_embedding = torch.from_numpy(speaker_embedding)

    # Ensure correct device
    if tts_device == "cuda":
        gpt_cond_latent = gpt_cond_latent.cuda()
        speaker_embedding = speaker_embedding.cuda()

    return gpt_cond_latent, speaker_embedding


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if xtts_model is None:
        return {"status": "unhealthy", "error": "Model not loaded"}
    return {
        "status": "healthy",
        "device": tts_device,
        "model_loaded": xtts_model is not None,
        "streaming_supported": True
    }


@app.get("/speakers")
async def list_speakers():
    """List available built-in speakers."""
    if xtts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if hasattr(xtts_model, 'speaker_manager') and xtts_model.speaker_manager is not None:
        speakers = list(xtts_model.speaker_manager.speakers.keys())
        return {"speakers": speakers, "count": len(speakers)}
    
    return {"speakers": [], "count": 0, "note": "No speaker manager available"}


@app.post("/tts")
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text (non-streaming).

    Returns audio as a complete WAV file.
    """
    if xtts_model is None or xtts_config is None:
        raise HTTPException(
            status_code=503,
            detail="TTS model not loaded. Check server logs."
        )

    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    try:
        logger.info(f"Synthesizing: '{request.text[:50]}...' (language: {request.language or 'en'})")

        speaker_id = request.speaker_id or "Ana Florence"
        logger.info(f"Using speaker: {speaker_id}")

        with torch.inference_mode():
            result = xtts_model.synthesize(
                text=request.text,
                config=xtts_config,
                speaker_wav=None,
                language=request.language or "en",
                speaker_id=speaker_id
            )

        # Handle return value
        if isinstance(result, dict):
            wav = result.get('wav', result)
            sample_rate = result.get('sample_rate', 22050)
        elif isinstance(result, tuple):
            wav = result[0]
            sample_rate = result[1] if len(result) > 1 else 22050
        else:
            wav = result
            sample_rate = 22050

        # Convert to numpy
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
        elif isinstance(wav, list):
            wav = np.array(wav)
        elif not isinstance(wav, np.ndarray):
            wav = np.array(wav)

        # Ensure 1D
        if wav.ndim > 1:
            wav = wav.flatten()

        # Ensure float32
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        # Normalize audio
        max_val = np.abs(wav).max()
        if max_val > 0:
            target_max = 0.95
            if max_val < 0.1:
                wav = wav * (target_max / max_val) * 0.5
            else:
                wav = wav * (target_max / max_val)

        wav = np.clip(wav, -1.0, 1.0)

        logger.info(f"Audio shape: {wav.shape}, sample_rate: {sample_rate}")

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, wav, sample_rate, format='WAV')
        buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="tts_output.wav"'
            }
        )

    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"TTS synthesis failed: {str(e)}"
        )


@app.post("/tts/stream")
async def synthesize_speech_stream(request: TTSRequest):
    """
    Synthesize speech with real-time streaming using cloned voice.

    Returns audio chunks as they're generated using chunked transfer encoding.
    Audio format: Raw PCM float32, mono, 24kHz

    Uses reference audio (ref.mp3) for voice cloning by default.
    Set speaker_id to use a built-in speaker instead.
    """
    if xtts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    language = request.language or "en"
    chunk_size = request.stream_chunk_size or 20

    try:
        # Use cloned voice from reference audio by default
        # If speaker_id is provided, use built-in speaker instead
        if request.speaker_id:
            gpt_cond_latent, speaker_embedding = get_speaker_latents(request.speaker_id)
            logger.info(f"Using built-in speaker: {request.speaker_id}")
        else:
            gpt_cond_latent, speaker_embedding = get_cloned_voice_latents(REFERENCE_AUDIO_PATH)
            logger.info(f"Using cloned voice from: {REFERENCE_AUDIO_PATH}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    async def audio_generator() -> AsyncGenerator[bytes, None]:
        """Generate audio chunks in real-time."""
        try:
            logger.info(f"Starting stream: '{request.text[:50]}...' speaker={request.speaker_id or 'cloned'}")
            
            with torch.inference_mode():
                # Use inference_stream for real-time chunk generation
                chunks = xtts_model.inference_stream(
                    text=request.text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    stream_chunk_size=chunk_size,
                    enable_text_splitting=True,
                    speed=request.speed
                )
                
                chunk_count = 0
                total_samples = 0
                
                for chunk in chunks:
                    # Convert tensor to numpy
                    if isinstance(chunk, torch.Tensor):
                        chunk_np = chunk.detach().cpu().numpy()
                    else:
                        chunk_np = np.array(chunk)
                    
                    # Ensure float32
                    if chunk_np.dtype != np.float32:
                        chunk_np = chunk_np.astype(np.float32)
                    
                    # Flatten if needed
                    if chunk_np.ndim > 1:
                        chunk_np = chunk_np.flatten()
                    
                    # Normalize chunk
                    max_val = np.abs(chunk_np).max()
                    if max_val > 1.0:
                        chunk_np = chunk_np / max_val * 0.95
                    
                    chunk_np = np.clip(chunk_np, -1.0, 1.0)
                    
                    chunk_count += 1
                    total_samples += len(chunk_np)
                    
                    # Yield raw PCM bytes
                    yield chunk_np.tobytes()
                    
                    # Allow other async tasks to run
                    await asyncio.sleep(0)
                
                logger.info(f"Stream complete: {chunk_count} chunks, {total_samples} samples")
                
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            raise

    return StreamingResponse(
        audio_generator(),
        media_type="application/octet-stream",
        headers={
            "Content-Type": "application/octet-stream",
            "X-Audio-Format": "pcm",
            "X-Audio-Encoding": "float32",
            "X-Audio-Sample-Rate": "24000",
            "X-Audio-Channels": "1",
            "Transfer-Encoding": "chunked",
            "Cache-Control": "no-cache"
        }
    )


@app.post("/tts/stream/wav")
async def synthesize_speech_stream_wav(request: TTSRequest):
    """
    Synthesize speech with streaming, but return as WAV with proper header.
    
    Note: This buffers audio internally but streams the final WAV.
    For true low-latency streaming, use /tts/stream with raw PCM.
    
    This endpoint is useful for clients that can't handle raw PCM
    but want streaming transfer encoding.
    """
    if xtts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    speaker_id = request.speaker_id or "Ana Florence"
    language = request.language or "en"
    chunk_size = request.stream_chunk_size or 20

    try:
        gpt_cond_latent, speaker_embedding = get_speaker_latents(speaker_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    async def wav_generator() -> AsyncGenerator[bytes, None]:
        """Generate WAV file with streamed chunks."""
        try:
            logger.info(f"Starting WAV stream: '{request.text[:50]}...'")
            
            all_chunks = []
            
            with torch.inference_mode():
                chunks = xtts_model.inference_stream(
                    text=request.text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    stream_chunk_size=chunk_size,
                    enable_text_splitting=True,
                    speed=request.speed
                )
                
                for chunk in chunks:
                    if isinstance(chunk, torch.Tensor):
                        chunk_np = chunk.detach().cpu().numpy()
                    else:
                        chunk_np = np.array(chunk)
                    
                    if chunk_np.dtype != np.float32:
                        chunk_np = chunk_np.astype(np.float32)
                    
                    if chunk_np.ndim > 1:
                        chunk_np = chunk_np.flatten()
                    
                    all_chunks.append(chunk_np)
                    await asyncio.sleep(0)
            
            # Concatenate all chunks
            full_audio = np.concatenate(all_chunks)
            
            # Normalize
            max_val = np.abs(full_audio).max()
            if max_val > 0:
                full_audio = full_audio / max_val * 0.95
            full_audio = np.clip(full_audio, -1.0, 1.0)
            
            # Convert to WAV
            buffer = io.BytesIO()
            sf.write(buffer, full_audio, 24000, format='WAV')
            buffer.seek(0)
            
            # Stream the WAV in chunks
            chunk_size_bytes = 8192
            while True:
                data = buffer.read(chunk_size_bytes)
                if not data:
                    break
                yield data
            
            logger.info(f"WAV stream complete: {len(full_audio)} samples")
            
        except Exception as e:
            logger.error(f"WAV streaming error: {e}", exc_info=True)
            raise

    return StreamingResponse(
        wav_generator(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="tts_output.wav"',
            "Transfer-Encoding": "chunked"
        }
    )


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "XTTS TTS Service",
        "version": "1.1.0",
        "model_loaded": xtts_model is not None,
        "device": tts_device,
        "endpoints": {
            "health": "/health - Health check",
            "speakers": "/speakers - List available speakers",
            "tts": "/tts (POST) - Full audio synthesis (returns WAV)",
            "tts_stream": "/tts/stream (POST) - Real-time streaming (returns raw PCM)",
            "tts_stream_wav": "/tts/stream/wav (POST) - Streaming with WAV output"
        },
        "streaming": {
            "format": "Raw PCM float32",
            "sample_rate": 24000,
            "channels": 1,
            "note": "Use /tts/stream for lowest latency"
        }
    }


def main():
    """Application entry point."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5002"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
