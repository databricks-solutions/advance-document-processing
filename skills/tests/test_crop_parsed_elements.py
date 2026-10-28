from crop_elements import (
    PARSE_DPI,
    builtin_chart_data,
    figure_context,
    pixel_box,
    splice_figure_text,
)

# Element layout observed live on page 2 (page_id 1) of a slide deck parsed with
# ai_parse_document v2.0; coords are pixels at 200 DPI (page image 5300 x 3000).
PAGE_W, PAGE_H = 5300, 3000


def _el(i, typ, coord, content, page=1):
    return {"id": i, "type": typ, "content": content, "bbox": [{"coord": coord, "page_id": page}]}


ELEMENTS = [
    _el(2, "figure", [4558, 2718, 5035, 2823], "Meta", page=0),
    _el(3, "title", [281, 123, 3191, 276], "Advertising Revenue by User Geography"),
    _el(4, "figure", [4553, 144, 5030, 249], "Meta"),
    _el(5, "text", [281, 303, 657, 378], "In Millions"),
    _el(6, "figure", [254, 711, 4876, 2400], '{"title": "None", "values": {"US & Canada": {"Q3\'23": 14956}}}'),
    _el(7, "text", [334, 2694, 4966, 2796], "Our revenue by user geography is geographically apportioned based on " * 3),
    _el(8, "page_number", [5114, 2853, 5168, 2913], "2"),
    _el(9, "title", [281, 123, 3191, 276], "Next Slide Title", page=2),
]


def test_parse_dpi():
    assert PARSE_DPI == 200


def test_pixel_box_pads_and_clamps_to_page():
    assert pixel_box([254, 711, 4876, 2400], PAGE_W, PAGE_H, pad_px=20) == (234, 691, 4896, 2420)
    assert pixel_box([10, 5, 5290, 2995], PAGE_W, PAGE_H, pad_px=20) == (0, 0, 5300, 3000)


def test_pixel_box_rescales_to_render_dpi_and_orders_corners():
    # rendering at 100 DPI halves every coordinate (page is then 2650 x 1500)
    assert pixel_box([254, 711, 4876, 2400], 2650, 1500, pad_px=0, render_dpi=100) == (127, 355, 2438, 1200)
    assert pixel_box([4876, 2400, 254, 711], PAGE_W, PAGE_H, pad_px=0) == (254, 711, 4876, 2400)


def test_figure_context_collects_same_page_title_and_units_only():
    ctx = figure_context(ELEMENTS, ELEMENTS[4])
    assert ctx == "Advertising Revenue by User Geography\nIn Millions"


def test_figure_context_includes_caption_and_skips_long_body_text():
    els = ELEMENTS + [_el(10, "caption", [254, 2420, 2000, 2480], "Figure 3: revenue by region")]
    ctx = figure_context(els, ELEMENTS[4])
    assert ctx.splitlines() == [
        "Advertising Revenue by User Geography", "In Millions", "Figure 3: revenue by region",
    ]


def test_builtin_chart_data():
    assert builtin_chart_data(ELEMENTS[4]) == {"title": "None", "values": {"US & Canada": {"Q3'23": 14956}}}
    assert builtin_chart_data(ELEMENTS[2]) is None                      # logo text
    assert builtin_chart_data({"content": "$50,082\n$46,783"}) is None  # parsed with descriptionElementTypes=''


def test_splice_replaces_figure_content_in_reading_order():
    text = splice_figure_text(ELEMENTS[1:5], {6: "[CHART: ad revenue by region, $M]"})
    assert text.splitlines() == [
        "Advertising Revenue by User Geography", "Meta", "In Millions", "[CHART: ad revenue by region, $M]",
    ]


def test_figure_context_includes_adjacent_legend_parsed_as_footnote():
    # Observed live (page 5): the legend sits right of the chart and is typed `footnote`;
    # the built-in chart JSON then names series "Segment 1..4".
    els = [
        _el(21, "title", [286, 126, 3000, 276], "Expenses as a Percentage of Revenue", page=4),
        _el(24, "figure", [281, 774, 4240, 2361], '{"title": "None"}', page=4),
        _el(25, "footnote", [4335, 1608, 4924, 2334],
            "General &\nAdministrative\n\nMarketing\n& Sales\n\nResearch &\nDevelopment\n\nCost of\nRevenue", page=4),
        _el(31, "footnote", [276, 2574, 5003, 2670], "*Provision for income taxes includes a one-time charge " * 4, page=4),
    ]
    ctx = figure_context(els, els[1])
    assert ctx.startswith("Expenses as a Percentage of Revenue\nGeneral &\nAdministrative")
    assert "Provision for income taxes" not in ctx   # far, long footnote is not context
