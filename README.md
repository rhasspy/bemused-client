# Bemused Client

Streaming TFLite-based keyword detector.

New models can be trained with https://github.com/google-research/google-research/tree/master/kws_streaming

## Dependencies

* Python 3.7 or higher
* Tensorflow 2.4
* sounddevice

## Model Format

Models should be placed in a directory with:

* model.tflite - the streaming TFLite model file
* config.json - JSON configuration for bemused

### Model Configuration

Below is a sample configuration for a "raspberry" keyword:

```json
{
    "sample_rate": 16000,
    "num_channels": 1,
    "block_ms": 20,
    "keyword_labels": [
        "raspberry"
    ],
    "all_labels": [
        "_silence_",
        "_unknown_",
        "raspberry",
        "_not_kw_"
    ]
}

```

## Usage

```sh
$ python3 -m bemused_client /path/to/model/directory
```

See `bemused_client --help` for more options
