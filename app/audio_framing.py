"""Format des blocs audio envoyés par le navigateur.

- « legacy » (format historique, un seul micro) : PCM 16 bits brut.
- « tagged » (plusieurs micros) : 2 octets d'en-tête, puis le PCM 16 bits :
      octet 0 = version du format (1)
      octet 1 = numéro du micro (0, 1, 2…)
Le format est annoncé par l'interface dans le message « start » : il n'y a donc
jamais d'ambiguïté entre les deux.
"""
from __future__ import annotations

FRAME_VERSION = 1


def make_frame(source_index: int, pcm: bytes) -> bytes:
    return bytes([FRAME_VERSION, source_index]) + pcm


def parse_frame(data: bytes, tagged: bool) -> tuple[int | None, bytes]:
    """Retourne (numéro du micro ou None pour le format historique, PCM)."""
    if not tagged:
        return None, data
    if len(data) < 2 or data[0] != FRAME_VERSION:
        raise ValueError("bloc audio mal formé")
    return data[1], data[2:]
