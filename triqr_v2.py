"""Load the bundled legacy TriQR decoder from TriCode.zip.

This keeps `import triqr_v2` working even when the source file is not
checked out next to the current workspace, while preserving the original
implementation shipped in the archive.
"""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile


def _load_legacy_module() -> None:
    zip_path = Path(__file__).with_name("TriCode.zip")
    if not zip_path.exists():
        raise ImportError(f"missing bundled archive: {zip_path}")

    with ZipFile(zip_path) as zf:
        try:
            source = zf.read("triqr_v2.py")
        except KeyError as exc:
            raise ImportError("triqr_v2.py not found inside TriCode.zip") from exc

    code = compile(source, f"{zip_path}!/triqr_v2.py", "exec")
    exec(code, globals(), globals())


_load_legacy_module()
del _load_legacy_module
