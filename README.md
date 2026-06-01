# Tricode
Tricode is a QR-like code system with triangular cells, adaptive ECC, optional HMAC signing, and image-based decoding.

## What It Does
- Encodes text into a custom visual code
- Supports adaptive compression
- Supports signed payloads with HMAC-SHA256
- Decodes from generated images
- Provides visualization and anchor detection utilities

## Requirements
- Python 3.11+
- `numpy`
- `Pillow`
- Optional: `opencv-python` for the full `detect` pipeline
- Optional: `cryptography` for stronger key storage

## Quick Start
Encode a payload:

```bash
python3 tricode.py encode "hello world" out.png
```

Decode it:

```bash
python3 tricode.py decode out.png
```

Enroll a signer:

```bash
python3 tricode.py enroll alice password123
```

Encode a signed payload:

```bash
python3 tricode.py encode "signed message" out.png --sign
```

## Commands
- `encode <text> <out.png> [--sign]`
- `decode <image.png> [--photo] [--verify-pw PW]`
- `detect <image.png> [--out x.png] [--thresh 0.55] [--photo] [--verify-pw PW]`
- `enroll <name> <password>`
- `templates`

## Files
- `tricode.py`: CLI entrypoint
- `tricode_final.py`: command dispatcher
- `tricode_encode.py`: payload layout and image rendering
- `tricode_decode.py`: payload recovery
- `tricode_common.py`: shared constants and ECC helpers
- `tricode_payload.py`: packing, compression, and parsing
- `tricode_render.py`: anchor rendering and templates
- `tricode_detect.py`: anchor detection and geometry
- `tricode_security.py`: key enrollment and HMAC helpers

## Notes
- This workspace uses the `Tricode` name only.
- Legacy `triqr_*` compatibility code has been removed.
- `detect` and some photo workflows need OpenCV.

