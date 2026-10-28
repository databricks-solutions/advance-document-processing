from large_pdf_parse import chunk_page_ranges, expected_chunks, stitch_chunks


def test_chunk_page_ranges():
    assert chunk_page_ranges(0) == []
    assert chunk_page_ranges(None) == []
    assert chunk_page_ranges(500) == ["1-500"]
    assert chunk_page_ranges(750) == ["1-500", "501-750"]
    assert chunk_page_ranges(1500) == ["1-500", "501-1000", "1001-1500"]
    assert chunk_page_ranges(5, chunk_size=2) == ["1-2", "3-4", "5-5"]


def test_expected_chunks():
    assert expected_chunks(500) == 1
    assert expected_chunks(501) == 2
    assert expected_chunks(-1) == 0


def _chunk(first_page, n_pages, n_elements):
    """Shape observed live from ai_parse_document v2.0 with pageRange: page ids and
    bbox page_ids are already document-global (0-based); element ids restart at 0."""
    page_ids = list(range(first_page, first_page + n_pages))
    return {
        "document": {
            "pages": [{"id": pid, "image_uri": None} for pid in page_ids],
            "elements": [
                {"id": i, "type": "text", "content": f"p{page_ids[i % n_pages]}",
                 "bbox": [{"coord": [0, 0, 1, 1], "page_id": page_ids[i % n_pages]}]}
                for i in range(n_elements)
            ],
        },
        "metadata": {"version": "2.0"},
        "error_status": None,
    }


def test_stitch_keeps_global_page_ids_and_renumbers_element_ids():
    # chunks arrive out of order; chunk_size=3 -> chunk 1 covers 0-based pages 3..4
    out = stitch_chunks([(1, _chunk(3, 2, 2)), (0, _chunk(0, 3, 3))])
    doc = out["document"]
    assert [p["id"] for p in doc["pages"]] == [0, 1, 2, 3, 4]
    assert [e["bbox"][0]["page_id"] for e in doc["elements"]] == [0, 1, 2, 3, 4]
    assert [e["id"] for e in doc["elements"]] == [0, 1, 2, 3, 4]
    assert doc["elements"][3]["content"] == "p3"
    assert out["metadata"] == {"version": "2.0"}
    assert out["error_status"] is None


def test_stitch_keeps_chunk_errors():
    bad = _chunk(3, 2, 1)
    bad["error_status"] = [{"error_message": "boom", "page_id": 4}]
    out = stitch_chunks([(0, _chunk(0, 3, 1)), (1, bad)])
    assert out["error_status"] == [{"error_message": "boom", "page_id": 4}]


def test_stitch_does_not_mutate_input():
    c = _chunk(3, 2, 2)
    stitch_chunks([(1, c)])
    assert c["document"]["elements"][1]["id"] == 1
