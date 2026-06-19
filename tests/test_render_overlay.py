"""Smoke test per il render overlay headless (richiede il PDF Bryston)."""
from __future__ import annotations

from pathlib import Path

import pytest

_PDF = Path("test_input/bryston_schematic.pdf")


@pytest.mark.skipif(not _PDF.exists(), reason="PDF Bryston non presente")
def test_build_overlay_smoke() -> None:
    from src.ui.render import build_overlay

    res = build_overlay(str(_PDF), dpi=120)
    assert res.image.width > 0 and res.image.height > 0
    assert res.components > 0
    assert 0 <= res.isolated <= res.components


@pytest.mark.skipif(not _PDF.exists(), reason="PDF Bryston non presente")
def test_link_dist_changes_clustering() -> None:
    from src.ui.render import build_overlay

    a = build_overlay(str(_PDF), dpi=100, link_dist=6.0)
    b = build_overlay(str(_PDF), dpi=100, link_dist=14.0)
    # link_dist maggiore accorpa: non puo' produrre piu' componenti.
    assert b.components <= a.components
