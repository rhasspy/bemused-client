import argparse
import json
import logging
import os
import sys
import threading
import time
import typing
from pathlib import Path

from . import KeywordConfig, KeywordDetector, DEFAULT_NUM_HITS_TO_DETECT

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger("bemused_client")

_DETECTOR: typing.Optional[KeywordDetector] = None

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    global _DETECTOR

    parser = argparse.ArgumentParser(prog="bemused-client")

    # Model settings
    parser.add_argument(
        "model_dir",
        help="Path to directory with streaming TFLite keyword model and config",
    )

    parser.add_argument(
        "--num-hits-to-detect",
        type=int,
        default=DEFAULT_NUM_HITS_TO_DETECT,
        help="Number of times a keyword label must be the best before a detection occurs",
    )

    parser.add_argument(
        "--refractory-seconds",
        type=float,
        default=1.0,
        help="Seconds before a another detection can occur",
    )

    parser.add_argument(
        "--stdin-audio",
        action="store_true",
        help="Read 16Khz, 16-bit PCM audio from stdin instead of microphone",
    )

    # HTTP settings
    parser.add_argument(
        "--http-host",
        type=str,
        help="Host for web server (default: 0.0.0.0)",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        help="Port for web server (default: 8000)",
        default=8000,
    )

    # MQTT settings
    parser.add_argument(
        "--mqtt-host",
        type=str,
        help="Host for MQTT broker (default: 127.0.0.1)",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        help="Port for MQTT broker (default: 1883)",
        default=1883,
    )

    parser.add_argument(
        "--log-detector",
        action="store_true",
        help="Log fine-grained details about keyword detector (implies --debug)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug or args.log_detector:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Convert to paths
    args.model_dir = Path(args.model_dir)

    # ------------------
    # Load configuration
    # ------------------

    model_path = args.model_dir / "model.tflite"
    assert model_path.is_file(), "Expected tflite model at {model_path}"
    model_name = args.model_dir.name

    config_path = args.model_dir / "config.json"
    _LOGGER.debug("Loading configuration from %s", config_path)

    # Label index to keyword name
    keyword_names: typing.Dict[int, str] = {}

    with open(config_path, "r") as config_file:
        json_config = json.load(config_file)

        # Convert string labels to integers
        all_labels = json_config.get("all_labels", [])
        keyword_str_labels = json_config.get("keyword_labels", [])
        keyword_labels: typing.Set[int] = set()

        for keyword_str in keyword_str_labels:
            if isinstance(keyword_str, str):
                # Look up in list of all labels
                label_idx = all_labels.index(keyword_str)
            else:
                # Allow integer labels
                label_idx = int(keyword_str)

            keyword_labels.add(label_idx)
            keyword_names[label_idx] = str(keyword_str)

        assert keyword_labels, "No keyword labels"

        kw_config = KeywordConfig(
            model_path=model_path,
            keyword_labels=keyword_labels,
            sample_rate=int(json_config.get("sample_rate", 16000)),
            num_channels=int(json_config.get("num_channels", 1)),
            block_ms=int(json_config.get("block_ms", 20)),
            process_ms=int(json_config.get("process_ms", 100)),
            streaming=bool(json_config.get("streaming", False)),
            num_hits_to_detect=args.num_hits_to_detect,
            refractory_seconds=args.refractory_seconds,
            log_detector=args.log_detector,
        )

        _LOGGER.debug(kw_config)

    # -------------------------------
    # Start streaming from microphone
    # -------------------------------

    _LOGGER.debug("Starting keyword detector...")
    _DETECTOR = KeywordDetector(config=kw_config)
    _DETECTOR.start()

    # Start detector printing thread
    threading.Thread(
        target=detector_proc, args=(model_name, keyword_names), daemon=True
    ).start()

    if args.stdin_audio:
        if os.isatty(sys.stdin.fileno()):
            print("Reading 16khz 16-bit mono PCM from stdin...", file=sys.stdin)

        try:
            chunk_size = _DETECTOR.block_samples * kw_config.sample_bytes
            chunk_number = 0

            while True:
                chunk = sys.stdin.buffer.read(chunk_size)
                if not chunk:
                    break

                audio_sec = chunk_number * (kw_config.block_ms / 1000)
                _DETECTOR.process_chunk(chunk, tag=audio_sec)
                chunk_number += 1
        except KeyboardInterrupt:
            # Exit immediately
            sys.exit(0)

        # Process remainder of queue
        while not _DETECTOR.chunk_queue.empty():
            time.sleep(0.01)

        sys.exit(0)

    # Delay import
    _LOGGER.debug("Starting microphone...")
    import sounddevice as sd

    _LOGGER.info("Ready")

    with sd.InputStream(
        channels=kw_config.num_channels,
        samplerate=kw_config.sample_rate,
        blocksize=_DETECTOR.block_samples,
        callback=sd_callback,
    ):
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            _LOGGER.info("Stopping detector")
            _DETECTOR.stop()


# -----------------------------------------------------------------------------


def detector_proc(model_name: str, keyword_names: typing.Dict[str, str]):
    """Prints detections to stdout"""
    assert _DETECTOR

    start_time = time.perf_counter()
    detect_tick = 0

    while _DETECTOR.running:
        detection = _DETECTOR.detect_queue.get()

        # Similar command-line output to Raven
        json.dump(
            {
                "keyword": keyword_names.get(detection.label, str(detection.label)),
                "label": detection.label,
                "detect_seconds": detection.detect_time - start_time,
                "detect_timestamp": time.time(),
                "detect_tick": detect_tick,
                "model": model_name,
                "tag": detection.tag,
            },
            sys.stdout,
            ensure_ascii=False,
        )
        print("", flush=True)

        detect_tick += 1


def sd_callback(rec, frames, _time, status):
    """sounddevice callback function"""

    # Notify if errors
    if status:
        _LOGGER.error(status)
        return

    assert _DETECTOR
    _DETECTOR.process_chunk(rec)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
