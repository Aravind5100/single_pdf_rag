import inspect

from chunk_strategy import (
    CHUNK_SIZE,
    OVERLAP,
    chunk_pages,
    chunk_text,
    normalize_for_search,
)


def test_constants_are_the_defaults_used_by_chunk_text():
    params = inspect.signature(chunk_text).parameters
    assert params["chunk_size"].default == CHUNK_SIZE == 900
    assert params["overlap"].default == OVERLAP == 350


def test_overlap_math_with_default_params():
    # 2000 chars, stride = 900 - 350 = 550, so windows start at 0, 550, 1100, 1650.
    text = "0123456789" * 200
    chunks = chunk_text(text)

    assert [len(c) for c in chunks] == [900, 900, 900, 350]
    # Consecutive chunks share exactly OVERLAP characters.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev[-OVERLAP:] == nxt[:OVERLAP]


def test_small_custom_window():
    # stride 3: windows at 0, 3, 6, 9
    assert chunk_text("abcdefghij", chunk_size=4, overlap=1) == ["abcd", "defg", "ghij", "j"]


def test_short_text_is_a_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_empty_and_whitespace_text_give_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_normalize_joins_hyphenated_line_breaks_and_collapses_whitespace():
    assert normalize_for_search("inter-\nnational   students\n\nonly") == "international students only"


def test_normalize_handles_none():
    assert normalize_for_search(None) == ""


def test_chunk_pages_keeps_chunks_inside_their_page():
    pages = [
        {"page_number": 1, "text": "alpha " * 400},   # about 2400 chars, several chunks
        {"page_number": 2, "text": "bravo " * 400},
    ]
    chunks = chunk_pages(pages)

    by_page = {}
    for c in chunks:
        by_page.setdefault(c["page_number"], []).append(c)

    assert set(by_page) == {1, 2}
    assert all("bravo" not in c["text"] for c in by_page[1])
    assert all("alpha" not in c["text"] for c in by_page[2])
    # Chunk ids restart per page and are 1-based.
    assert [c["chunk_id"] for c in by_page[1]][:2] == ["1_chunk_1", "1_chunk_2"]
    assert by_page[2][0]["chunk_id"] == "2_chunk_1"


def test_chunk_pages_skips_empty_pages():
    pages = [
        {"page_number": 1, "text": "real content here"},
        {"page_number": 2, "text": ""},
        {"page_number": 3, "text": "   \n "},
        {"page_number": 4, "text": "more content"},
    ]
    chunks = chunk_pages(pages)

    assert [c["page_number"] for c in chunks] == [1, 4]


def test_chunk_pages_normalizes_text_before_chunking():
    chunks = chunk_pages([{"page_number": 7, "text": "resi-\ndence   halls"}])

    assert chunks == [{"page_number": 7, "chunk_id": "7_chunk_1", "text": "residence halls"}]
