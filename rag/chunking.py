"""
chunking.py — pluggable text-splitting strategies with character offsets.

Every strategy returns Piece objects whose text is exactly text[start:end],
so the UI can highlight where a chunk sits in the original document.

Strategies:
    fixed     — word windows with word overlap (the original behaviour)
    recursive — split on paragraph → line → sentence → word boundaries
    sentence  — sentence boundaries, packed up to the target size
    markdown  — one chunk per heading section, split further only if too big
    semantic  — break where adjacent sentences stop being similar

Semantic chunking embeds sentences at chunk time, so it needs an embedding
model name and only pays off when that model is already cached.

Overlap is measured in words. Values are clamped so the splitter always
makes progress (overlap >= size used to loop forever).
"""

import re
from dataclasses import dataclass
from typing import List

_WORDS = re.compile(r"\S+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"(?m)^#{1,6} .*$")

STRATEGIES = ("fixed", "recursive", "sentence", "markdown", "semantic")

# Coarsest boundary first; the empty separator means "split between words".
_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", " ")


@dataclass
class Piece:
    text: str
    start: int  # character offset in the source text
    end: int


def word_count(text: str) -> int:
    return len(_WORDS.findall(text))


def _words(text: str) -> List[tuple]:
    return [(m.start(), m.end()) for m in _WORDS.finditer(text)]


def _fixed_pieces(text: str, size: int, overlap: int) -> List[Piece]:
    words = _words(text)
    if len(words) <= size:
        return [Piece(text, 0, len(text))] if words else []

    overlap = max(0, min(overlap, size - 1))
    pieces = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        pieces.append(Piece(text[words[start][0]:words[end - 1][1]], words[start][0], words[end - 1][1]))
        if end >= len(words):
            break
        start = end - overlap
    return pieces


def _split_on(text: str, base: int, sep: str) -> List[Piece]:
    """Split on a separator, leaving it *between* segments. Packing later
    slices the original text from the first to the last segment, so the
    separator is recovered; keeping it inside a segment would make a
    trailing separator split into one identical part and recurse forever."""
    parts, pos = [], 0
    while True:
        idx = text.find(sep, pos)
        if idx == -1:
            break
        if idx > pos:
            parts.append(Piece(text[pos:idx], base + pos, base + idx))
        pos = idx + len(sep)
    if pos < len(text):
        parts.append(Piece(text[pos:], base + pos, base + len(text)))
    return parts


def _recursive_pieces(text: str, base: int, size: int) -> List[Piece]:
    if word_count(text) <= size:
        return [Piece(text, base, base + len(text))] if text.strip() else []

    for sep in _SEPARATORS:
        if sep not in text:
            continue
        out = []
        for part in _split_on(text, base, sep):
            if word_count(part.text) > size:
                out.extend(_recursive_pieces(part.text, part.start, size))
            else:
                out.append(part)
        return out

    # No separator left: hard word split (no overlap while recursing).
    return [Piece(p.text, base + p.start, base + p.end) for p in _fixed_pieces(text, size, 0)]


def _pack(text: str, pieces: List[Piece], size: int, overlap: int) -> List[Piece]:
    """Merge adjacent pieces into chunks of at most ~size words, carrying the
    trailing `overlap` words forward onto the next chunk."""
    if not pieces:
        return []
    overlap = max(0, min(overlap, size - 1))
    chunks = []
    i = 0
    while i < len(pieces):
        j, words = i, 0
        while j < len(pieces) and (words + word_count(pieces[j].text) <= size or j == i):
            words += word_count(pieces[j].text)
            j += 1

        start, end = pieces[i].start, pieces[j - 1].end
        chunks.append(Piece(text[start:end], start, end))

        next_i, carried = j, 0
        while next_i - 1 > i and carried + word_count(pieces[next_i - 1].text) <= overlap:
            next_i -= 1
            carried += word_count(pieces[next_i].text)
        i = next_i
    return chunks


def _sentence_pieces(text: str, size: int) -> List[Piece]:
    segments, pos = [], 0
    for m in _SENTENCE_BREAK.finditer(text):
        segments.append(Piece(text[pos:m.start()], pos, m.start()))
        pos = m.end()
    if pos < len(text):
        segments.append(Piece(text[pos:], pos, len(text)))

    out = []
    for seg in segments:
        if not seg.text.strip():
            continue
        if word_count(seg.text) <= size:
            out.append(seg)
        else:
            out.extend(Piece(p.text, seg.start + p.start, seg.start + p.end)
                       for p in _fixed_pieces(seg.text, size, 0))
    return out


def _markdown_pieces(text: str, size: int, overlap: int) -> List[Piece]:
    headings = list(_HEADING.finditer(text))
    if not headings:
        return _pack(text, _recursive_pieces(text, 0, size), size, overlap)

    sections = []
    if headings[0].start() > 0:
        sections.append((0, headings[0].start()))
    for idx, m in enumerate(headings):
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
        sections.append((m.start(), end))

    out = []
    for start, end in sections:
        section_text = text[start:end]
        if not section_text.strip():
            continue
        if word_count(section_text) <= size:
            out.append(Piece(section_text, start, end))
        else:
            inner = _recursive_pieces(section_text, start, size)
            out.extend(_pack(text, inner, size, overlap))
    return out


def _semantic_pieces(text: str, size: int, overlap: int, model: str | None,
                     threshold: float) -> List[Piece]:
    """Group neighbouring sentences, breaking where their embeddings diverge.

    A break is made when the cosine distance between two adjacent sentences
    exceeds `threshold` or when the next sentence would overflow the budget.
    """
    if not model:
        raise ValueError("Semantic chunking needs an embedding model.")
    from rag.store import encode_passages  # only this strategy needs embeddings

    sentences = _sentence_pieces(text, size)
    if len(sentences) < 2:
        return sentences

    embeddings = encode_passages(model, [s.text for s in sentences])
    overlap = max(0, min(overlap, size - 1))

    groups, first = [], 0
    words, budget = word_count(sentences[0].text), size
    for i in range(1, len(sentences)):
        distance = 1.0 - float(embeddings[i - 1] @ embeddings[i])
        w = word_count(sentences[i].text)
        if distance > threshold or words + w > budget:
            groups.append((first, i))
            first, words = i, w
            budget = max(size - overlap, 1)  # the carried tail comes out of it
        else:
            words += w
    groups.append((first, len(sentences)))

    chunks, prev_first = [], 0
    for first, last in groups:
        begin, carried = first, 0
        while begin > prev_first and carried + word_count(sentences[begin - 1].text) <= overlap:
            begin -= 1
            carried += word_count(sentences[begin].text)
        start, end = sentences[begin].start, sentences[last - 1].end
        chunks.append(Piece(text[start:end], start, end))
        prev_first = first
    return chunks


def split_text(text: str, strategy: str = "fixed", size: int = 500, overlap: int = 50,
               model: str | None = None, threshold: float = 0.75) -> List[Piece]:
    """Split text with the named strategy. Raises ValueError on bad input.

    `model` and `threshold` are only used by the semantic strategy.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose from: {', '.join(STRATEGIES)}")
    size = int(size)
    overlap = int(overlap)
    if size < 20:
        raise ValueError("Chunk size must be at least 20 words.")
    if not 0 <= overlap < size:
        raise ValueError("Chunk overlap must be >= 0 and smaller than the chunk size.")
    if not text.strip():
        return []

    if strategy == "fixed":
        return _fixed_pieces(text, size, overlap)
    if strategy == "recursive":
        return _pack(text, _recursive_pieces(text, 0, size), size, overlap)
    if strategy == "sentence":
        return _pack(text, _sentence_pieces(text, size), size, overlap)
    if strategy == "semantic":
        return _semantic_pieces(text, size, overlap, model, threshold)
    return _markdown_pieces(text, size, overlap)
