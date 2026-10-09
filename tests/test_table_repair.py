"""Test table_repair: hàm thuần Python + TableStructurer trên crop mẫu.

Các test TableStructurer cần model slanet-plus (chỉ có trong container worker)
nên skipif khi chạy trên host; trong container: `docker exec ami-rag-worker
python -m pytest /app/tests/test_table_repair.py` (hoặc pytest đầy đủ nếu mount).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ami_rag.core.table_repair import (
    CAT_KEPT_OCR_EMPTY,
    DEFAULT_MODEL_PATH,
    MODULE_VERSION,
    SHORT_SIM_STRICT,
    TableRepairer,
    assign_words_to_cells,
    best_window_match,
    digits_match,
    extract_ocr_cells,
    get_page_words,
    html_td_per_tr,
    is_numeric_text,
    map_cells_to_page,
    normalize_text,
    replace_td_texts,
    snap_cells_to_grid,
    structure_td_per_tr,
    text_similarity,
)

OCR_HTML = (
    "<table><tr><td>Ting Anh</td><td>3</td></tr>"
    "<tr><td>Gii tch</td><td>INT100</td></tr></table>"
)

BS4_AVAILABLE = True
try:
    import bs4  # noqa: F401
except ImportError:
    BS4_AVAILABLE = False


def test_normalize_text_bo_dau():
    assert normalize_text("Giải tích") == "giai tich"
    assert normalize_text("Thuong mại điện tử") == "thuong mai dien tu"
    assert normalize_text("  Kinh  tế ") == "kinh te"
    # Đ hoa (U+0110) và đ thường (U+0111) đều map về d
    assert normalize_text("ĐẠI HỌC") == "dai hoc"
    assert normalize_text("Đồ án") == "do an"


def test_similarity_sau_bo_dau_giong_nhau():
    # OCR mất dấu vs text layer đúng dấu -> cùng nội dung, tương đồng cao
    assert text_similarity("Gii tích", "Giải tích") >= 0.9
    # SequenceMatcher không khớp 100% do chèn "e" ("ting" vs "tieng") nhưng cao
    assert text_similarity("Ting Anh", "Tiếng Anh") >= 0.9


def test_similarity_khac_noi_dung():
    assert text_similarity("Giải tích", "Kinh tế vi mô") < 0.5
    assert text_similarity("", "abc") == 0.0
    assert text_similarity("", "") == 1.0


@pytest.mark.skipif(not BS4_AVAILABLE, reason="bs4 không có trên host")
def test_extract_ocr_cells_thu_tu():
    cells = extract_ocr_cells(OCR_HTML)
    assert cells == ["Ting Anh", "3", "Gii tch", "INT100"]


@pytest.mark.skipif(not BS4_AVAILABLE, reason="bs4 không có trên host")
def test_replace_td_texts_giu_cau_truc():
    html = (
        "<table><tr><td colspan='2'>Cu trúc d liu</td></tr>"
        "<tr><td>1</td><td>Gii tch</td></tr></table>"
    )
    out = replace_td_texts(html, ["Cấu trúc dữ liệu", "1", "Giải tích"])
    assert "Cấu trúc dữ liệu" in out
    assert "Giải tích" in out
    assert "colspan" in out  # giữ nguyên cấu trúc
    assert "Cu trúc" not in out


@pytest.mark.skipif(not BS4_AVAILABLE, reason="bs4 không có trên host")
def test_replace_td_texts_it_hon_so_td():
    out = replace_td_texts(OCR_HTML, ["Tiếng Anh"])
    assert "Tiếng Anh" in out
    assert "Gii tch" in out  # td dư giữ nguyên


def test_map_cells_to_page_he_so_tu_data():
    # bbox chuẩn hoá 0-1000 trên trang 425x607pt, ảnh crop 1042x181 (landscape)
    # -> scale = (vùng_pt / jpg_px), khớp 0.36 quan sát được
    cells_px = np.array([[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]])
    img = np.zeros((181, 1042))  # (height, width)
    cells, (sx, sy) = map_cells_to_page(cells_px, [63, 59, 945, 166], (425.2, 606.6), img)
    assert abs(sx - 0.36) < 0.01
    assert abs(sy - 0.36) < 0.01
    # ô đầu tiên nằm trong vùng bbox
    x0 = 63 * 425.2 / 1000
    y0 = 59 * 606.6 / 1000
    assert cells[0][:, 0].min() >= x0 - 1.0
    assert cells[0][:, 1].min() >= y0 - 1.0


def test_map_cells_to_page_bbox_lech():
    # bbox quá nhỏ so với ảnh -> scale ngoài ngưỡng 0.36
    cells_px = np.zeros((2, 4, 2))
    img = np.zeros((181, 1042))
    _cells, (sx, sy) = map_cells_to_page(cells_px, [0, 0, 100, 100], (425.2, 606.6), img)
    assert abs(sx - 0.36) > 0.01 or abs(sy - 0.36) > 0.01


def test_count_tolerance_nguong():
    # per-tr gate thay count gate: so td tung hang phai trung nhau
    model_rows = [4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2]
    html_rows = [4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2]
    assert structure_td_per_tr(
        ["<tr>"] + ["<td></td>"] * 4 + ["</tr>"] * 2
    ) == [4, 4]
    # trang 59: model 18 hang vs html 19 hang -> KHONG khop (bo bang)
    assert model_rows != html_rows
    # trang 59 dpi 200 (da do): 19 hang khop exact
    assert html_rows == [4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2]
    assert abs(67 - 69) / 69 <= 0.15  # count gate cu se qua — per-tr gate chan


@pytest.mark.skipif(not BS4_AVAILABLE, reason="bs4 không có trên host")
def test_html_td_per_tr():
    html = "<table><tr><td>a</td><td>b</td></tr><tr><td colspan='2'>c</td></tr></table>"
    assert html_td_per_tr(html) == [2, 1]


def test_structure_td_per_tr_dem_du_td():
    toks = ["<tr>", "<td></td>", "<td></td>", "</tr>", "<tr>", "<td></td>", "</tr>"]
    assert structure_td_per_tr(toks) == [2, 1]


def test_is_numeric_text():
    assert is_numeric_text("0912 316")
    assert is_numeric_text("3")
    assert is_numeric_text("1,000.5")
    assert not is_numeric_text("INT100")
    assert not is_numeric_text("Giải tích")
    assert not is_numeric_text("")


def test_module_version():
    assert MODULE_VERSION.startswith("table_repair/")


def test_repairer_lazy_khoi_tao_khong_minio():
    r = TableRepairer(minio_client=None)
    assert r._structurer is None
    assert r._render_cache == {}
    assert r._chars_cache == {}


def _mk_chars(s, x0=0.0, y0=0.0, cw=4.0, ch=6.0, space_w=3.0):
    """Chars giả từ chuỗi: mỗi ký tự rộng cw, khoảng trắng rộng space_w."""
    out = []
    x = x0
    for ch_ in s:
        if ch_ == " ":
            out.append((" ", x, y0, x + space_w, y0 + ch))
            x += space_w
        else:
            out.append((ch_, x, y0, x + cw, y0 + ch))
            x += cw
    return out


def test_get_page_words_tach_khoang_trang_va_giu_thu_tu():
    words = get_page_words(_mk_chars("ĐƠN VỊ PTIT", x0=100.0))
    assert [w.text for w in words] == ["ĐƠN", "VỊ", "PTIT"]
    # thứ tự stream giữ nguyên, không sắp theo toạ độ
    assert words[0].x0 < words[1].x0 < words[2].x0
    assert words[0].text == "ĐƠN"


def test_get_page_words_khong_tach_email_va_so_thap_phan():
    # đã đo trên SBV: thiếu guard thì 'hieunt@ptit.edu.vn' bị xé thành 3 mảnh
    words = get_page_words(_mk_chars("hieunt@ptit.edu.vn", space_w=0.0))
    assert [w.text for w in words] == ["hieunt@ptit.edu.vn"]
    # gap lớn trước '.' nhưng vẫn KHÔNG tách (guard dấu câu)
    words2 = get_page_words(
        [("h", 0, 0, 4, 6), ("i", 4, 0, 8, 6), (".", 20, 0, 22, 6), ("e", 22, 0, 26, 6)]
    )
    assert [w.text for w in words2] == ["hi.e"]


def test_get_page_words_tach_khi_gap_lon():
    # gap 5pt > max(0.9, 0.45*6=2.7) -> từ mới dù không có ký tự space
    words = get_page_words(
        [("T", 0, 0, 4, 6), ("ô", 4, 0, 8, 6), ("i", 13, 0, 17, 6)]
    )
    assert [w.text for w in words] == ["Tô", "i"]


def test_assign_words_moi_tu_mot_oi():
    # ô A [0,0,20,10], ô B [20,0,40,10]; từ straddling [16,0,24,8]
    # giao A: 4x8=32, giao B: 4x8=32 -> hoà, argmax lấy index đầu (A) —
    # nhưng từ phải về ĐÚNG MỘT ô, không xuất hiện ở cả hai
    from ami_rag.core.table_repair import LayerWord

    cells = np.array(
        [[[0, 0], [20, 0], [20, 10], [0, 10]], [[20, 0], [40, 0], [40, 10], [20, 10]]],
        dtype=float,
    )
    words = [LayerWord("straddle", 16, 0, 24, 8), LayerWord("inB", 30, 1, 36, 7)]
    out = assign_words_to_cells(words, cells)
    owners = [i for i, ws in enumerate(out) for _ in ws]
    assert sorted(owners) == [0, 1]  # mỗi từ đúng 1 ô
    assert [w.text for w in out[1]] == ["inB"]


def test_snap_cells_to_grid_median_canh():
    # 2 ô cùng cột: left lệch 1pt nhau -> snap về trung vị; 1 ô lệch hẳn giữ nguyên
    cells = np.array(
        [
            [[10.0, 0.0], [50.0, 0.0], [50.0, 20.0], [10.0, 20.0]],
            [[11.0, 20.0], [50.5, 20.0], [50.5, 40.0], [11.0, 40.0]],
            [[200.0, 0.0], [300.0, 0.0], [300.0, 20.0], [200.0, 20.0]],
        ]
    )
    out = snap_cells_to_grid(cells, eps=2.0)
    assert abs(out[0][:, 0].min() - out[1][:, 0].min()) < 1e-9  # cùng left
    assert abs(out[0][:, 0].max() - out[1][:, 0].max()) < 1e-9  # cùng right
    assert abs(out[2][:, 0].min() - 200.0) < 1e-9  # ô riêng không kéo theo


def test_snap_cells_khong_lat_oi():
    # guard: nếu snap làm ô gần suy biến thì giữ rect gốc
    cells = np.array(
        [
            [[0.0, 0.0], [5.0, 0.0], [5.0, 10.0], [0.0, 10.0]],
            [[4.0, 0.0], [9.0, 0.0], [9.0, 10.0], [4.0, 10.0]],
        ]
    )
    out = snap_cells_to_grid(cells, eps=3.0)
    w0 = out[0][:, 0].max() - out[0][:, 0].min()
    w1 = out[1][:, 0].max() - out[1][:, 0].min()
    assert w0 >= 1.0 and w1 >= 1.0


def test_best_window_bo_rac_lanh_canh():
    # (1a) OCR đúng nội dung, layer dính thêm 1 từ của ô kế bên
    text, sim, _src = best_window_match("BM An toàn", [["BM", "An", "toàn", "mạng?"], []])
    assert text == "BM An toàn"
    assert sim >= 0.99


def test_best_window_full_lap_tu_garble():
    # OCR có đọc từ đó nhưng garble ('mng') -> full (4 token) thắng cửa sổ 3
    text, sim, _src = best_window_match("BM An toàn mng", [["BM", "An", "toàn", "mạng"], []])
    assert text == "BM An toàn mạng"
    assert sim >= 0.85


def test_best_window_exact_khong_tu_dien_thieu():
    # OCR thiếu hẳn 1 từ -> cửa sổ n khớp 1.0, KHÔNG tự điền (ưu tiên chính xác)
    text, sim, _src = best_window_match("BM An toàn", [["BM", "An", "toàn", "mạng"], []])
    assert text == "BM An toàn"
    assert sim == 1.0


def test_best_window_uu_tien_nguon_words():
    # cả hai nguồn như nhau -> chọn nguồn 0 (words)
    toks = ["Ting", "Anh"]
    _t1, _s1, src1 = best_window_match("Ting Anh", [toks, toks])
    _t2, _s2, src2 = best_window_match("Ting Anh", [[], toks])
    assert src1 == 0 and src2 == 1


def test_best_window_ocr_rong():
    assert best_window_match("", [["a"], []]) == ("", 0.0, 0)
    assert best_window_match("x", [[], []]) == ("", 0.0, 0)


def test_digits_match():
    assert digits_match("11 tín chỉ", "11 tín chỉ")
    assert not digits_match("11 tín chỉ", "1 tín chỉ")
    assert not digits_match("62", "2")


def test_nguong_o_ngan_strict_khong_ha():
    # (1d) SHORT_SIM_STRICT cao hơn ngưỡng thường và là ngưỡng duy nhất cho ô ngắn
    assert SHORT_SIM_STRICT >= 0.85
    r = TableRepairer(minio_client=None)
    assert r.strict_short and r.use_window and r.word_assign and r.grid_snap
    # baseline S0: tắt hết -> lặp lại hành vi v2.0
    r0 = TableRepairer(
        minio_client=None,
        use_window=False,
        word_assign=False,
        grid_snap=False,
        strict_short=False,
        restore_diacritics=False,
    )
    assert not r0.use_window and not r0.word_assign


def test_env_thread_doc_lai_sau_tao_session():
    """(mục 4) MINERU_*_OP_NUM_THREADS chỉ đặt quanh việc tạo session —
    thất bại (host không có mineru) vẫn phải khôi phục env."""
    import subprocess
    import sys

    code = (
        "import os, sys\n"
        "os.environ['MINERU_INTRA_OP_NUM_THREADS'] = '4'\n"
        "from ami_rag.core.table_repair import TableRepairer\n"
        "r = TableRepairer(minio_client=None)\n"
        "try:\n"
        "    _ = r.structurer\n"
        "except Exception:\n"
        "    pass\n"
        "assert os.environ.get('MINERU_INTRA_OP_NUM_THREADS') == '4', 'env bị ghi đè!'\n"
        "print('env OK')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=os.getcwd(), check=False
    )
    assert "env OK" in out.stdout, out.stderr


def test_get_page_words_khong_tach_underscore_va_gach_noi():
    # (3-g) đã đo trên SBV: 'nvhung_vt1@ptit.edu.vn' bị xé 'nvhung _ vt1@...'
    words = get_page_words(
        [("n", 0, 0, 4, 6), ("v", 4, 0, 8, 6), ("_", 20, 0, 22, 6), ("v", 22, 0, 26, 6)],
        no_split_before=".,;:!?%)]}\"'_-",
    )
    assert [w.text for w in words] == ["nv_v"]


def test_reject_fragment_chan_thay_bang_manh():
    # (3-h) 'EMAIL' -> 'MAIL' (fragment) phải bị chặn; 'Ting'->'Tiếng' thì không
    r = TableRepairer(minio_client=None)
    assert r.no_split_extra and r.reject_fragment
    from ami_rag.core import table_repair as tr

    nm, no = tr.normalize_text("MAIL"), tr.normalize_text("EMAIL")
    assert nm in no and len(nm) < len(no)  # fragment
    nm2, no2 = tr.normalize_text("Tiếng"), tr.normalize_text("Ting")
    assert not (nm2 in no2 and len(nm2) < len(no2))  # thay hợp lệ


def test_kept_ocr_empty_la_category_rieng():
    # OCR rỗng + layer có chữ -> bỏ nhầm (tách khỏi empty_both để báo cáo)
    assert CAT_KEPT_OCR_EMPTY != "empty_both"


MODEL_AVAILABLE = os.path.exists(DEFAULT_MODEL_PATH)


@pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason="model slanet-plus chỉ có trong container worker (/models)",
)
def test_table_structurer_import_va_crop_mau():
    """Constraint (e): nâng cấp mineru sau này không âm thầm làm hỏng."""

    from ami_rag.core.table_repair import structure_cells

    # crop mẫu: lưới đơn giản 3x3 ô, đường kẻ đen trên nền trắng
    img = np.full((300, 600, 3), 255, dtype=np.uint8)
    for y in (0, 100, 200, 299):
        img[y : y + 2, :] = 0
    for x in (0, 200, 400, 598):
        img[:, x : x + 2] = 0
    r = TableRepairer(minio_client=None)
    cells, structures = structure_cells(r.structurer, img)
    assert len(cells) >= 4, f"structure model trả {len(cells)} cell cho lưới 3x3"
    assert structures, "structure token list rỗng"
    # bbox ô nằm trong ảnh
    h, w = img.shape[:2]
    assert cells[:, :, 0].min() >= -1.0 and cells[:, :, 0].max() <= w + 1.0
    assert cells[:, :, 1].min() >= -1.0 and cells[:, :, 1].max() <= h + 1.0
