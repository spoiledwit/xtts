"""Simple test for XTTS streaming endpoint with real-time playback."""

import argparse
import queue
import threading

import httpx
import numpy as np
import sounddevice as sd
import soundfile as sf

BASE_URL = "https://d5xln73itvuncd-5002.proxy.runpod.net"
SAMPLE_RATE = 24000


def test_stream(text: str, speed: float = 1.15):
    """Test the /tts/stream endpoint with real-time chunk playback."""
    audio_queue = queue.Queue()
    all_chunks = []
    stream_done = threading.Event()

    def audio_callback(outdata, frames, time, status):
        """Callback for sounddevice to play audio from queue."""
        try:
            data = audio_queue.get_nowait()
            if len(data) < len(outdata):
                outdata[: len(data)] = data.reshape(-1, 1)
                outdata[len(data) :] = 0
            else:
                outdata[:] = data[: len(outdata)].reshape(-1, 1)
        except queue.Empty:
            outdata[:] = 0

    print(f"Text: {text}\n")
    print("Streaming and playing in real-time...\n")

    # Start audio output stream
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        callback=audio_callback,
        blocksize=1024,
    )
    stream.start()

    total_bytes = 0

    try:
        with httpx.stream(
            "POST",
            f"{BASE_URL}/tts/stream",
            json={"text": text, "language": "en", "speed": speed},
            timeout=120.0,
        ) as response:
            print(f"Status: {response.status_code}")

            for i, chunk in enumerate(response.iter_bytes(chunk_size=4096)):
                total_bytes += len(chunk)
                print(f"Chunk {i + 1}: {len(chunk)} bytes")

                # Convert to float32 audio
                audio = np.frombuffer(chunk, dtype=np.float32)
                all_chunks.append(audio)

                # Queue for playback
                audio_queue.put(audio)

    except Exception as e:
        print(f"Error: {e}")

    # Wait for queue to empty
    while not audio_queue.empty():
        sd.sleep(100)
    sd.sleep(500)  # Extra buffer

    stream.stop()
    stream.close()

    print(f"\nTotal bytes: {total_bytes}")

    # Save complete audio
    if all_chunks:
        full_audio = np.concatenate(all_chunks)
        sf.write("test_output.wav", full_audio, SAMPLE_RATE)
        print("Saved to test_output.wav")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test XTTS streaming endpoint")
    parser.add_argument(
        "text",
        nargs="?",
        default=(
            "The quick brown fox jumps over the lazy dog. "
            "This is a comprehensive test of the text to speech streaming service. "
            "We are testing how well the system handles longer paragraphs of text, "
            "including multiple sentences with varying punctuation! "
            "Can it handle questions? What about exclamations! "
            "Let's also try some numbers like 123 and dates like January 15th, 2025. "
            "Finally, we conclude this test with a simple goodbye. Thank you for listening."
        ),
        help="Text to synthesize",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.15,
        help="Speech speed (1.0 = normal, 1.2 = faster)",
    )
    args = parser.parse_args()
    test_stream(args.text, args.speed)
