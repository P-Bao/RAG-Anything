"""Sửa chữ mất dấu trong bảng bằng cách ghép text layer của PDF vào ô bảng.

MinerU nhận dạng bảng bằng OCR model Trung Quốc (ch_PP-OCRv6) nên chữ trong ô
bảng mất dấu tiếng Việt, trong khi text layer của PDF luôn đúng dấu. Module này
render crop bảng trực tiếp từ PDF (DPI 200), chạy slanet-plus độc lập để lấy
bbox từng ô + cấu trúc hàng, đối chiếu cấu trúc với HTML OCR, rồi đọc chữ từ
text layer (pypdfium2 ``get_text_bounded``) cho từng ô và thay nội dung ô.

Không patch MinerU, không re-parse, không phụ thuộc ảnh crop đã lưu (jpg bị
nén/cắt mất hàng cuối — đã đo: cùng bảng cho 18 hàng từ jpg vs 19 hàng từ PDF
render, khớp HTML exact). HTML gốc (OCR) không bị ghi đè — bản vá lưu ở trường
riêng ``repaired_table_body`` kèm metadata, re-embed từ bản vá, rollback được ngay.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import statistics
import unicodedata
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

MODULE_VERSION = "table_repair/2.1"

DEFAULT_MODEL_PATH = (
    "/models/huggingface/hub/models--opendatalab--PDF-Extract-Kit-1.0/"
    "snapshots/ed6b654c018d742e65a17671e379c5e6ecc87ec9/models/TabRec/"
    "SlanetPlus/slanet-plus.onnx"
)

# Render crop trực tiếp từ PDF ở DPI 200 — trùng DPI render gốc của MinerU
# (đã đo: cấu trúc hàng khớp HTML exact ở DPI 200, lệch ở DPI 300).
RENDER_DPI = 200
RENDER_SCALE = RENDER_DPI / 72.0

# Cổng tương đồng: bỏ dấu + NFC, so chữ OCR cũ vs chữ text layer của ô.
# Ô ngắn (< SHORT_CELL_LEN ký tự) yêu cầu khớp gần chính xác.
SHORT_CELL_LEN = 12
SHORT_SIM_THRESHOLD = 0.80
LONG_SIM_THRESHOLD = 0.60
# Cổng theo bảng: sim trung vị bảng < 0.5 -> bỏ cả bảng (structure/bbox đáng ngờ);
# >= 0.7 -> hạ ngưỡng cho ô DÀI (ô ngắn không bao giờ hạ — số/1 từ sai là hỏng dữ liệu).
TABLE_MEDIAN_DROP = 0.50
TABLE_MEDIAN_RELAX = 0.70
RELAXED_SHORT_THRESHOLD = 0.65
RELAXED_LONG_THRESHOLD = 0.45
# (1d) ô ngắn ở chế độ strict: khớp gần chính xác sau bỏ dấu, không bao giờ hạ.
SHORT_SIM_STRICT = 0.85

# Tách từ: khoảng ngang lớn hơn max(GAP_MIN, GAP_H_FRAC x cao ký tự) -> từ mới.
# Không tách trước dấu câu (email 'x.ptit.edu.vn', số '3.5' đã bị tách oai khi đo).
# Vòng 3 bổ sung '_-' — đã đo: thiếu 2 ký tự này làm hỏng email 'nvhung_vt1@...'
# ('nvhung _ vt1@...') và 'E-Marketing' ('E -Marketing') ở sim 0.95+.
WORD_GAP_MIN = 0.9
WORD_GAP_H_FRAC = 0.45
NO_SPLIT_BEFORE = ".,;:!?%)]}\"'"
NO_SPLIT_EXTRA = "_-"

# (1c) snap cạnh ô về trung vị nhóm cạnh trùng nhau (lưới cột/hàng), dung sai pt.
GRID_SNAP_EPS = 2.0

RESIZED = 488

STATUS_REPAIRED = "repaired"
STATUS_SKIPPED = "skipped"

# Phân loại ô
CAT_ALREADY_CORRECT = "already_correct"  # giống sau bỏ dấu (chỉ lệch dấu/hoa/khoảng trắng)
CAT_REPLACED = "replaced"                # đã thay bằng text layer
CAT_KEPT_LOW_SIM = "kept_low_sim"        # giữ OCR vì sim thấp (bbox lệch) — thất bại
CAT_KEPT_NUMERIC = "kept_numeric"        # ô toàn số giữ OCR — không tính thất bại
CAT_KEPT_DIGITS = "kept_digits_mismatch"  # nhóm số lệch -> giữ OCR (số OCR tin được) — an toàn
CAT_KEPT_NO_TEXT = "kept_no_text"        # bbox không có chữ — thất bại
CAT_KEPT_OCR_EMPTY = "kept_ocr_empty"    # OCR rỗng nhưng layer có chữ — bỏ nhầm (không vá theo ràng buộc)
CAT_EMPTY_BOTH = "empty_both"            # cả hai rỗng — ô rỗng hợp lệ


def normalize_text(text: str) -> str:
    """Bỏ dấu + NFC + collapse khoảng trắng + lowercase — để so tương đồng.

    "đ" là chữ cái riêng (U+0111/U+0110) không phải combining mark nên NFD
    không strip được — map tường minh đ->d, Đ->D.
    """
    s = unicodedata.normalize("NFC", str(text))
    s = s.replace("đ", "d").replace("Đ", "D")
    s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", s).strip().lower()


def normalize_nfc_lower(text: str) -> str:
    """NFC + collapse + lowercase, GIỮ dấu — dùng cho chỉ số token."""
    s = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", s).strip().lower()


def text_similarity(a: str, b: str) -> float:
    """Tỉ lệ tương đồng sau bỏ dấu + NFC, [0, 1]."""
    na, nb = normalize_text(a), normalize_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def is_numeric_text(text: str) -> bool:
    """Ô toàn số (kèm . , - + / và khoảng trắng) — giữ OCR không tính thất bại."""
    s = re.sub(r"\s+", "", str(text))
    return bool(s) and bool(re.fullmatch(r"[\d.,\-+/()]+", s))


def digits_match(a: str, b: str) -> bool:
    """Nhóm số của 2 text phải trùng nhau — chống thay đổi số khi vá.

    Đã đo: '12'->'1', '62'->'2', '11'->'1' qua cổng sim 0.65-0.93 (số bị cắt mất
    chữ số nhưng sim vẫn cao). Số OCR tin được (chữ số nằm trong bảng chữ cái
    của OCR model) nên nhóm số lệch -> giữ OCR.
    """
    return re.findall(r"\d+", str(a)) == re.findall(r"\d+", str(b))


@dataclass
class CellRecord:
    index: int
    ocr_text: str
    layer_text: str
    final_text: str
    similarity: float | None = None
    category: str = CAT_EMPTY_BOTH
    match_text: str = ""    # text nguồn đã chọn (cửa sổ/bounded)
    match_source: int = 0  # 0 = words, 1 = bounded
    words_text: str = ""    # join các từ được gán vào ô (để phán quyết kept_ocr_empty)


@dataclass
class TableRepairResult:
    page_idx: int
    status: str
    reason: str | None = None
    n_cells: int = 0
    n_html_td: int = 0
    median_sim: float | None = None
    sim_thresholds: tuple[float, float] | None = None  # (short, long) đã áp dụng
    position_shift: tuple[float, float, float] | None = None  # (dx, dy, dscale)
    n_cut_chars: int | None = None
    cells: list[CellRecord] = field(default_factory=list)
    short_sim: float | None = None
    long_sim: float | None = None
    n_short_cells: int = 0
    n_long_cells: int = 0
    token_coverage_before: float | None = None
    token_coverage_after: float | None = None
    repaired_html: str | None = None

    def count_by_category(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.cells:
            out[c.category] = out.get(c.category, 0) + 1
        return out


@dataclass
class DocumentRepairReport:
    doc_id: str
    results: list[TableRepairResult] = field(default_factory=list)

    @property
    def n_tables(self) -> int:
        return len(self.results)

    @property
    def n_repaired(self) -> int:
        return sum(1 for r in self.results if r.status == STATUS_REPAIRED)

    def skipped_by_reason(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            if r.status == STATUS_SKIPPED and r.reason:
                out[r.reason] = out.get(r.reason, 0) + 1
        return out


def extract_ocr_cells(html: str) -> list[str]:
    """Trích text từng ô theo thứ tự từ HTML OCR gốc."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    return [td.get_text(" ", strip=True) for td in soup.find_all("td")]


def replace_td_texts(html: str, new_texts: list[str]) -> str:
    """Thay text từng ô trong HTML, giữ nguyên cấu trúc (rowspan/colspan)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    tds = soup.find_all("td")
    for td, txt in zip(tds, new_texts):
        td.string = txt if txt else ""
    return str(soup)


def structure_cells(structurer, img: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Chạy slanet-plus trên ảnh crop -> bbox từng ô (đã adapt về px của ảnh).

    Trả về (cells (N,4,2) theo px ảnh, structure token list). Mảng là PRE-mask
    (trước khi MinerU bỏ hàng bbox [0,0,0,0]).
    """
    h_px, w_px = img.shape[:2]
    structures, bboxes, _elapse = structurer.process(img.copy())
    b = np.asarray(bboxes).copy()
    if b.size == 0:
        return np.zeros((0, 4, 2)), list(structures)
    ratio = min(RESIZED / h_px, RESIZED / w_px)
    b[:, 0::2] *= RESIZED / (w_px * ratio)
    b[:, 1::2] *= RESIZED / (h_px * ratio)
    return b.reshape(len(b), 4, 2), list(structures)


def structure_td_per_tr(structures: list[str]) -> list[int]:
    """Số <td> trên từng <tr> từ token structure của model."""
    rows: list[int] = []
    cur = 0
    for tok in structures:
        s = str(tok)
        if "<tr" in s:
            cur = 0
        cur += s.count("<td")
        if "</tr>" in s:
            rows.append(cur)
    return rows


def html_td_per_tr(html: str) -> list[int]:
    """Số <td> trên từng <tr> từ HTML OCR."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    return [len(tr.find_all("td")) for tr in soup.find_all("tr")]


def map_cells_to_page(
    cells_px: np.ndarray,
    bbox_norm: list[float],
    page_size: tuple[float, float],
    img: np.ndarray,
    render_scale: float | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Map bbox ô (px của ảnh crop) về toạ độ trang (PDF points, top-origin).

    render_scale có -> crop render trực tiếp từ PDF, tỉ lệ = 1/render_scale (chính xác).
    Không có -> tự suy từ vùng bbox và kích thước ảnh (cách cũ, dùng cho ảnh jpg).
    """
    page_w, page_h = page_size
    jpg_h, jpg_w = img.shape[:2]
    x0 = bbox_norm[0] * page_w / 1000.0
    y0 = bbox_norm[1] * page_h / 1000.0
    x1 = bbox_norm[2] * page_w / 1000.0
    y1 = bbox_norm[3] * page_h / 1000.0
    if render_scale:
        sx = sy = 1.0 / render_scale
    else:
        sx = (x1 - x0) / max(jpg_w, 1)
        sy = (y1 - y0) / max(jpg_h, 1)
    cells = (cells_px.reshape(-1, 2) * np.array([sx, sy]) + np.array([x0, y0])).reshape(
        len(cells_px), 4, 2
    )
    return cells, (sx, sy)


def get_page_chars(page) -> list[tuple[str, float, float, float, float]]:
    """Ký tự + bbox (top-origin, PDF points) qua pypdfium2 char-level API.

    pdftext có lỗi frame (span x âm) nên dùng FPDFText_GetCharBox trực tiếp
    (đã kiểm chứng: bbox chuẩn, bottom-origin -> đổi sang top-origin).
    """
    from ctypes import byref, c_double

    import pypdfium2.raw as pdfium_raw

    tp = page.get_textpage()
    n = pdfium_raw.FPDFText_CountChars(tp)
    _pw, ph = page.get_size()
    out: list[tuple[str, float, float, float, float]] = []
    for i in range(n):
        left, right, bottom, top = c_double(), c_double(), c_double(), c_double()
        pdfium_raw.FPDFText_GetCharBox(tp, i, byref(left), byref(right), byref(bottom), byref(top))
        ch = chr(pdfium_raw.FPDFText_GetUnicode(tp, i))
        # bottom-origin -> top-origin: y_top = page_h - T, y_bottom = page_h - B
        out.append((ch, left.value, ph - top.value, right.value, ph - bottom.value))
    return out


def count_cut_chars(cells: np.ndarray, chars: list, region: tuple[float, float, float, float]) -> int:
    """Số ký tự bị cắt ngang biên ô: tâm nằm trong vùng bảng nhưng bbox không
    nằm gọn trong bất kỳ ô nào. Bỏ qua ký tự khoảng trắng (nhiễu giữa các ô)."""
    if not chars or cells.size == 0:
        return 0
    real = [c for c in chars if c[0].strip()]
    if not real:
        return 0
    cx0 = np.array([c[1] for c in real])
    cy0 = np.array([c[2] for c in real])
    cx1 = np.array([c[3] for c in real])
    cy1 = np.array([c[4] for c in real])
    ccx, ccy = (cx0 + cx1) / 2, (cy0 + cy1) / 2
    rx0, ry0, rx1, ry1 = region
    in_region = (ccx >= rx0) & (ccx <= rx1) & (ccy >= ry0) & (ccy <= ry1)
    bx0 = cells[:, :, 0].min(axis=1)
    by0 = cells[:, :, 1].min(axis=1)
    bx1 = cells[:, :, 0].max(axis=1)
    by1 = cells[:, :, 1].max(axis=1)
    inside = (
        (cx0[:, None] >= bx0[None, :])
        & (cx1[:, None] <= bx1[None, :])
        & (cy0[:, None] >= by0[None, :])
        & (cy1[:, None] <= by1[None, :])
    )
    cut = in_region & ~inside.any(axis=1)
    return int(cut.sum())


@dataclass
class LayerWord:
    """Một từ của text layer: text + bbox (PDF points, top-origin)."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float


def get_page_words(chars: list, no_split_before: str = NO_SPLIT_BEFORE) -> list[LayerWord]:
    """Tách từ từ char boxes, GIỮ thứ tự stream của PDF (= thứ tự đọc).

    Tách khi: ký tự khoảng trắng, khoảng ngang lớn, hoặc xuống dòng (y nhảy).
    Không tách trước dấu câu — đã đo email 'hieunt@ptit.edu.vn' bị xé thành
    'hieunt@ptit .edu .vn' nếu thiếu guard này. Thứ tự stream quan trọng vì
    ô chữ xoay dọc (trang 17) chỉ đúng theo stream, sắp theo (y,x) sẽ hỏng.
    """
    words: list[LayerWord] = []
    cur: list[tuple[str, float, float, float, float]] = []

    def flush() -> None:
        if cur:
            words.append(
                LayerWord(
                    text="".join(c[0] for c in cur),
                    x0=min(c[1] for c in cur),
                    y0=min(c[2] for c in cur),
                    x1=max(c[3] for c in cur),
                    y1=max(c[4] for c in cur),
                )
            )

    for ch, x0, y0, x1, y1 in chars:
        if not str(ch).strip():
            flush()
            cur = []
            continue
        if cur:
            ph = cur[-1]
            h = max(y1 - y0, 0.5)
            gap = x0 - ph[3]
            new_line = abs((y0 + y1) / 2 - (ph[2] + ph[4]) / 2) > 0.6 * h
            if (gap > max(WORD_GAP_MIN, WORD_GAP_H_FRAC * h) or new_line) and str(ch) not in no_split_before:
                flush()
                cur = []
        cur.append((ch, x0, y0, x1, y1))
    flush()
    return words


def assign_words_to_cells(words: list[LayerWord], cells: np.ndarray) -> list[list[LayerWord]]:
    """(1b) Mỗi từ thuộc đúng MỘT ô — ô có diện tích giao lớn nhất.

    Không cắt ngang token: từ straddling biên 2 ô chỉ về một ô, ô còn lại
    mất từ đó (an toàn hơn là ô này THÊM rác của ô kia).
    """
    n = len(cells)
    out: list[list[LayerWord]] = [[] for _ in range(n)]
    if not words or n == 0:
        return out
    bx0 = cells[:, :, 0].min(axis=1)
    by0 = cells[:, :, 1].min(axis=1)
    bx1 = cells[:, :, 0].max(axis=1)
    by1 = cells[:, :, 1].max(axis=1)
    for w in words:
        ox = np.maximum(0.0, np.minimum(w.x1, bx1) - np.maximum(w.x0, bx0))
        oy = np.maximum(0.0, np.minimum(w.y1, by1) - np.maximum(w.y0, by0))
        area = ox * oy
        j = int(np.argmax(area))
        if area[j] > 0:
            out[j].append(w)  # giữ thứ tự stream
    return out


def snap_cells_to_grid(cells: np.ndarray, eps: float = GRID_SNAP_EPS) -> np.ndarray:
    """(1c) Snap cạnh ô về trung vị của nhóm cạnh trùng nhau (|lép| <= eps).

    Cùng cột chia biên trái/phải, cùng hàng chia biên trên/dưới — trung vị
    hoá hết rung sub-pixel của model cấu trúc. Ô rowspan/colspan không cần
    xử lý riêng: cạnh nào trùng nhóm nào thì snap nhóm đó, cạnh riêng thì giữ.
    """
    n = len(cells)
    if n < 2:
        return cells
    rects = np.stack(
        [
            cells[:, :, 0].min(axis=1),
            cells[:, :, 1].min(axis=1),
            cells[:, :, 0].max(axis=1),
            cells[:, :, 1].max(axis=1),
        ],
        axis=1,
    )
    snapped = rects.copy()
    for e in range(4):
        col = rects[:, e]
        for i in range(n):
            grp = col[np.abs(col - col[i]) <= eps]
            if len(grp) >= 2:
                snapped[i, e] = float(np.median(grp))
    bad = (snapped[:, 2] - snapped[:, 0] < 1.0) | (snapped[:, 3] - snapped[:, 1] < 1.0)
    snapped[bad] = rects[bad]
    out = np.empty((n, 4, 2))
    out[:, 0, 0], out[:, 0, 1] = snapped[:, 0], snapped[:, 1]
    out[:, 1, 0], out[:, 1, 1] = snapped[:, 2], snapped[:, 1]
    out[:, 2, 0], out[:, 2, 1] = snapped[:, 2], snapped[:, 3]
    out[:, 3, 0], out[:, 3, 1] = snapped[:, 0], snapped[:, 3]
    return out


def best_window_match(ocr: str, token_lists: list[list[str]]) -> tuple[str, float, int]:
    """(1a) Cửa sổ khớp tốt nhất: trượt độ dài n-1..n+1 trên từng nguồn token.

    n = số token OCR; kèm cửa sổ FULL (toàn bộ từ của ô). Chọn sim cao nhất;
    hoà: cửa sổ sát n hơn, start sớm hơn, nguồn trước (words ưu tiên bounded).
    Hệ quả cố ý: từ OCR đọc THIẾU hẳn không được điền (cửa sổ n khớp 1.0 mọi
    nội dung) — ưu tiên chính xác hơn là phủ; full chỉ thắng khi OCR có đọc
    từ đó nhưng garble ('mng' vs 'mạng'). Trả về (text, sim, nguồn).
    """
    if not str(ocr).strip():
        return "", 0.0, 0
    n = max(len(ocr.split()), 1)
    cands: list[tuple[float, int, int, int, str]] = []
    for src, toks in enumerate(token_lists):
        m = len(toks)
        if m == 0:
            continue
        for size in {max(n - 1, 1), n, n + 1, m}:
            if size > m:
                continue
            for start in range(m - size + 1):
                text = " ".join(toks[start : start + size])
                cands.append((text_similarity(ocr, text), -abs(size - n), -start, -src, text))
    if not cands:
        return "", 0.0, 0
    best = max(cands, key=lambda t: t[:4])
    return best[4], best[0], -best[3]


def token_coverage(html_text: str, page_word_set: set[str]) -> float | None:
    """Tỉ lệ token của bảng có mặt trong tập từ của text layer trang, [0, 1]."""
    toks = [t for t in re.findall(r"\S+", normalize_nfc_lower(html_text)) if t]
    if not toks:
        return None
    return sum(1 for t in toks if t in page_word_set) / len(toks)


def page_word_set(page_text: str) -> set[str]:
    return set(re.findall(r"\S+", normalize_nfc_lower(page_text)))


class TableRepairer:
    """Sửa chữ bảng mất dấu cho một tài liệu PDF đã parse."""

    def __init__(
        self,
        minio_client=None,
        bucket: str = "ami-data-documents",
        model_path: str | None = None,
        use_cuda: bool = False,
        use_window: bool = True,
        word_assign: bool = True,
        grid_snap: bool = True,
        strict_short: bool = True,
        restore_diacritics: bool = True,
        restrict_bounded: bool = True,
        flat_threshold: bool = True,
        no_split_extra: bool = True,
        reject_fragment: bool = True,
    ):
        self._minio = minio_client
        self._bucket = bucket
        self._model_path = model_path or DEFAULT_MODEL_PATH
        self._use_cuda = use_cuda
        self._structurer = None
        self._render_cache: dict[int, np.ndarray] = {}
        self._chars_cache: dict[int, list] = {}
        self._words_cache: dict[int, list[LayerWord]] = {}
        # Cờ ablation (mặc định = cấu hình cuối):
        #   use_window       (1a) cửa sổ khớp tốt nhất thay vì lấy toàn bộ text ô
        #   word_assign      (1b) gán theo TỪ (max-overlap), bounded text chỉ là
        #                     ứng viên fallback (ô chữ xoay dọc chỉ đúng theo stream)
        #   grid_snap        (1c) snap cạnh ô về trung vị lưới
        #   strict_short     (1d) ô ngắn không hạ ngưỡng, khớp gần chính xác
        #   restore_diacritics ô chỉ lệch dấu (already_correct) cũng được thay
        #                     bằng text layer — v2.0 bỏ sót ô mất dấu thuần
        #   restrict_bounded (vòng 2-e) bounded chỉ là ứng viên khi ô KHÔNG có
        #                     từ nào hoặc chữ xoay dọc — đã đo: 77% thay bằng
        #                     bounded là sai/rác (get_text_bounded cắt chữ ở biên)
        #   flat_threshold   (vòng 2-f) một ngưỡng 0.85 cho mọi ô, bỏ relax —
        #                     đã đo: mọi thay sai còn lại đều ở sim <= 0.83
        self.use_window = use_window
        self.word_assign = word_assign
        self.grid_snap = grid_snap
        self.strict_short = strict_short
        self.restore_diacritics = restore_diacritics
        self.restrict_bounded = restrict_bounded
        self.flat_threshold = flat_threshold
        # (vòng 3-g) không tách từ trước '_-'; (vòng 3-h) chặn thay bằng fragment
        self.no_split_extra = no_split_extra
        self.reject_fragment = reject_fragment

    @property
    def structurer(self):
        if self._structurer is None:
            # ONNX đa luồng làm structure model KHÔNG tất định (đã đo: ~1% bảng
            # dao động 10<->5, 39<->61 ô giữa các lần chạy). MinerU đọc
            # MINERU_INTRA/INTER_OP_NUM_THREADS khi tạo session (os_env_config.py);
            # = 1 thì mọi lần chạy cho cùng kết quả. OMP_NUM_THREADS KHÔNG tác
            # dụng — onnxruntime dùng threadpool riêng (đã đo, bị nhầm là may mắn).
            saved = {}
            try:
                for name in ("MINERU_INTRA_OP_NUM_THREADS", "MINERU_INTER_OP_NUM_THREADS"):
                    saved[name] = os.environ.get(name)
                    os.environ[name] = "1"
                from mineru.model.table.rec.slanet_plus.table_structure import (
                    TableStructurer,
                )

                self._structurer = TableStructurer(
                    {"model_path": self._model_path, "use_cuda": self._use_cuda}
                )
            finally:
                for name, prev in saved.items():
                    if prev is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = prev
        return self._structurer

    def clear_cache(self) -> None:
        self._render_cache.clear()
        self._chars_cache.clear()
        self._words_cache.clear()

    def repair_document(
        self,
        pdf,
        content_list: list[dict],
        dry_run: bool = True,
    ) -> DocumentRepairReport:
        """Sửa mọi bảng trong content_list. dry_run=True chỉ đo, không ghi.

        pdf: pypdfium2.PdfDocument. content_list: list entry của MinerU
        (type='table', bbox chuẩn hoá 0-1000...). Khi dry_run=False, thêm
        ``repaired_table_body`` + ``repair_meta`` vào từng entry; HTML OCR gốc
        (``table_body``) giữ nguyên.
        """
        self.clear_cache()
        report = DocumentRepairReport(doc_id="")
        tables = [
            (idx, it)
            for idx, it in enumerate(content_list)
            if it.get("type") == "table" and it.get("table_body")
        ]
        # sắp theo trang để cache render/chars dùng lại
        tables.sort(key=lambda kv: kv[1].get("page_idx", -1))
        for idx, entry in tables:
            page_idx = entry.get("page_idx")
            page = pdf[page_idx]
            result = self.repair_table(page=page, table_entry=entry)
            report.results.append(result)
            if result.status == STATUS_REPAIRED and not dry_run:
                entry["repaired_table_body"] = result.repaired_html
                entry["repair_meta"] = {
                    "module_version": MODULE_VERSION,
                    "method": "text_layer_splice",
                    "median_sim": result.median_sim,
                    "position_shift": list(result.position_shift or ()),
                    "n_replaced": sum(
                        1 for c in result.cells if c.category == CAT_REPLACED
                    ),
                    "n_already_correct": sum(
                        1 for c in result.cells if c.category == CAT_ALREADY_CORRECT
                    ),
                }
        return report

    def _render_crop(self, page, bbox_norm: list[float], key: int | None = None) -> np.ndarray:
        """Render crop bảng trực tiếp từ PDF ở DPI 200 (không phụ thuộc jpg).

        key: page_idx — bắt buộc truyền khi gọi lặp lại qua nhiều trang. Không
        dùng id(page): pdf[i] tạo object mới mỗi lần, id bị tái sử dụng sau GC
        → cache có thể trả render của TRANG KHÁC.
        """
        cache_key = key if key is not None else id(page)
        if cache_key not in self._render_cache:
            import cv2

            bitmap = page.render(scale=RENDER_SCALE)
            pil = bitmap.to_pil()
            self._render_cache[cache_key] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
            if len(self._render_cache) > 4:
                self._render_cache.pop(next(iter(self._render_cache)))
        full = self._render_cache[cache_key]
        page_w, page_h = page.get_size()
        x0 = int(bbox_norm[0] * page_w / 1000.0 * RENDER_SCALE)
        y0 = int(bbox_norm[1] * page_h / 1000.0 * RENDER_SCALE)
        x1 = int(bbox_norm[2] * page_w / 1000.0 * RENDER_SCALE)
        y1 = int(bbox_norm[3] * page_h / 1000.0 * RENDER_SCALE)
        h, w = full.shape[:2]
        return full[max(y0, 0) : min(y1, h), max(x0, 0) : min(x1, w)]

    def _get_chars(self, page, key: int | None = None) -> list:
        cache_key = key if key is not None else id(page)
        if cache_key not in self._chars_cache:
            self._chars_cache[cache_key] = get_page_chars(page)
            if len(self._chars_cache) > 4:
                self._chars_cache.pop(next(iter(self._chars_cache)))
        return self._chars_cache[cache_key]

    def _get_words(self, page, key: int | None = None) -> list[LayerWord]:
        cache_key = key if key is not None else id(page)
        if cache_key not in self._words_cache:
            no_split = NO_SPLIT_BEFORE + (NO_SPLIT_EXTRA if self.no_split_extra else "")
            self._words_cache[cache_key] = get_page_words(
                self._get_chars(page, cache_key), no_split_before=no_split
            )
            if len(self._words_cache) > 4:
                self._words_cache.pop(next(iter(self._words_cache)))
        return self._words_cache[cache_key]

    def repair_table(self, *, page, table_entry: dict) -> TableRepairResult:
        """Sửa một bảng. Trả về kết quả kèm HTML đã vá (nếu repaired)."""
        page_idx = table_entry.get("page_idx", -1)
        ocr_html = table_entry.get("table_body", "") or ""
        result = TableRepairResult(page_idx=page_idx, status=STATUS_SKIPPED)

        ocr_cells = extract_ocr_cells(ocr_html)
        result.n_html_td = len(ocr_cells)
        if not ocr_cells:
            result.reason = "no_ocr_cells"
            return result

        crop = self._render_crop(page, table_entry["bbox"], key=page_idx)
        if crop is None or crop.size == 0:
            result.reason = "render_crop_failed"
            return result

        cells_px, structures = structure_cells(self.structurer, crop)
        result.n_cells = len(cells_px)
        if not cells_px.size:
            result.reason = "no_cells_detected"
            return result

        # Cổng cấu trúc: số <td> trên từng <tr> của model và HTML phải trùng nhau,
        # không thì ghép theo chỉ số sẽ lệch sau điểm phân kỳ -> bỏ bảng.
        model_rows = structure_td_per_tr(structures)
        html_rows = html_td_per_tr(ocr_html)
        if model_rows != html_rows:
            result.reason = "row_structure_mismatch"
            return result

        page_size = page.get_size()
        cells, (sx, sy) = map_cells_to_page(
            cells_px, table_entry["bbox"], page_size, crop, render_scale=RENDER_SCALE
        )
        bb = table_entry["bbox"]
        page_w, page_h = page_size
        region = (
            bb[0] * page_w / 1000.0,
            bb[1] * page_h / 1000.0,
            bb[2] * page_w / 1000.0,
            bb[3] * page_h / 1000.0,
        )

        # Tinh chỉnh vị trí ±3%: chọn mapping làm số ký tự cắt ngang biên ô ít nhất.
        chars = self._get_chars(page, key=page_idx)
        base_cut = count_cut_chars(cells, chars, region)
        best_cells, best_shift, best_cut = cells, (0.0, 0.0, 1.0), base_cut
        if base_cut > 0:
            for dx in (-0.03, 0.0, 0.03):
                for dy in (-0.03, 0.0, 0.03):
                    for ds in (0.97, 1.0, 1.03):
                        if dx == 0.0 and dy == 0.0 and ds == 1.0:
                            continue
                        cand = (
                            (cells_px.reshape(-1, 2) * np.array([sx * ds, sy * ds]))
                            + np.array(
                                [
                                    region[0] + dx * (region[2] - region[0]),
                                    region[1] + dy * (region[3] - region[1]),
                                ]
                            )
                        ).reshape(len(cells_px), 4, 2)
                        c = count_cut_chars(cand, chars, region)
                        if c < best_cut:
                            best_cells, best_shift, best_cut = cand, (dx, dy, ds), c
        cells = best_cells
        result.position_shift = best_shift
        result.n_cut_chars = best_cut
        if self.grid_snap:
            cells = snap_cells_to_grid(cells)

        # Đọc chữ text layer từng ô + so tương đồng
        textpage = page.get_textpage()
        page_h = page_size[1]
        words_per_cell = (
            assign_words_to_cells(self._get_words(page, key=page_idx), cells)
            if self.word_assign
            else None
        )
        legacy = not self.use_window and not self.word_assign
        records: list[CellRecord] = []
        for i, cell in enumerate(cells):
            cx0, cy0 = float(cell[:, 0].min()), float(cell[:, 1].min())
            cx1, cy1 = float(cell[:, 0].max()), float(cell[:, 1].max())
            # pypdfium2 dùng bottom-origin: bottom = page_h - y1, top = page_h - y0
            bounded = re.sub(
                r"\s+",
                " ",
                textpage.get_text_bounded(
                    left=cx0, bottom=page_h - cy1, right=cx1, top=page_h - cy0
                )
                or "",
            ).strip()
            ocr = ocr_cells[i] if i < len(ocr_cells) else ""
            has_words = bool(words_per_cell is not None and words_per_cell[i])
            rec = CellRecord(index=i, ocr_text=ocr, layer_text=bounded, final_text=ocr)
            if words_per_cell is not None:
                rec.words_text = " ".join(w.text for w in words_per_cell[i])
            if legacy:
                # đúng nhánh v2.0 để baseline ablation S0 lặp lại kết quả cũ
                if not bounded and not ocr:
                    rec.category = CAT_EMPTY_BOTH
                elif not bounded:
                    rec.category = CAT_KEPT_NO_TEXT
                else:
                    rec.similarity = round(text_similarity(ocr, bounded), 4)
                    rec.match_text = bounded
                    rec.category = (
                        CAT_ALREADY_CORRECT
                        if normalize_text(ocr) == normalize_text(bounded)
                        else "pending"
                    )
            elif not ocr.strip():
                rec.category = (
                    CAT_KEPT_OCR_EMPTY if (bounded or has_words) else CAT_EMPTY_BOTH
                )
            elif not bounded and not has_words:
                rec.category = CAT_KEPT_NO_TEXT
            else:
                cell_words = words_per_cell[i] if words_per_cell is not None else []
                # (2-e) bounded chỉ fallback khi không có từ nào hoặc chữ xoay dọc
                allow_bounded = True
                if self.restrict_bounded and cell_words:
                    vertical = len(cell_words) >= 3 and all(
                        len(w.text.strip()) <= 2 for w in cell_words
                    )
                    allow_bounded = vertical
                if self.use_window:
                    token_lists = []
                    if cell_words:
                        token_lists.append([w.text for w in cell_words])
                    if allow_bounded:
                        token_lists.append(bounded.split())
                    match_text, sim, src = best_window_match(ocr, token_lists)
                else:
                    match_text = " ".join(w.text for w in cell_words)
                    sim = text_similarity(ocr, match_text)
                    src = 0
                rec.match_text, rec.match_source = match_text, src
                rec.similarity = round(sim, 4)
                rec.category = (
                    CAT_ALREADY_CORRECT
                    if normalize_text(ocr) == normalize_text(match_text)
                    else "pending"
                )
            records.append(rec)
        result.cells = records

        # Cổng theo bảng: sim trung vị
        sims = [c.similarity for c in records if c.similarity is not None]
        if not sims:
            # không có ô nào có chữ để so — giữ OCR gốc
            result.reason = "no_layer_text"
            return result
        median_sim = statistics.median(sims)
        result.median_sim = round(median_sim, 4)
        if median_sim < TABLE_MEDIAN_DROP:
            result.reason = "table_median_sim_low"
            return result
        relaxed = median_sim >= TABLE_MEDIAN_RELAX
        if self.flat_threshold:
            # (2-f) một ngưỡng cho mọi ô — đã đo mọi thay sai còn lại ở sim<=0.83
            th_short = th_long = SHORT_SIM_STRICT
        elif self.strict_short:
            # (1d) ô ngắn không bao giờ hạ ngưỡng
            th_short = SHORT_SIM_STRICT
            th_long = RELAXED_LONG_THRESHOLD if relaxed else LONG_SIM_THRESHOLD
        else:
            th_short = RELAXED_SHORT_THRESHOLD if relaxed else SHORT_SIM_THRESHOLD
            th_long = RELAXED_LONG_THRESHOLD if relaxed else LONG_SIM_THRESHOLD
        result.sim_thresholds = (th_short, th_long)

        for rec in records:
            if rec.category == CAT_ALREADY_CORRECT:
                # (sửa regression v2.0) "đúng sau bỏ dấu" = OCR chỉ THIẾU DẤU —
                # phải thay bằng text layer có dấu, không phải giữ OCR mất dấu
                if self.restore_diacritics and rec.match_text:
                    rec.final_text = rec.match_text
                continue
            if rec.category != "pending":
                continue
            threshold = th_short if len(rec.ocr_text) < SHORT_CELL_LEN else th_long
            # (3-h) chặn thay bằng fragment: normalize(match) là chuỗi con THẬT sự
            # của normalize(ocr) -> cửa sổ làm mất nội dung ('EMAIL'->'MAIL',
            # 'Plus)'->'Plu'). Giữ OCR an toàn hơn.
            if self.reject_fragment:
                nm, no = normalize_text(rec.match_text), normalize_text(rec.ocr_text)
                if nm and nm in no and len(nm) < len(no):
                    rec.category = CAT_KEPT_LOW_SIM
                    continue
            # Lưới an toàn số: nhóm số phải trùng — '11'->'1' là hỏng dữ liệu.
            if rec.similarity >= threshold and digits_match(rec.ocr_text, rec.match_text):
                rec.final_text = rec.match_text
                rec.category = CAT_REPLACED
            elif is_numeric_text(rec.ocr_text):
                rec.category = CAT_KEPT_NUMERIC
            elif not digits_match(rec.ocr_text, rec.match_text):
                rec.category = CAT_KEPT_DIGITS
            else:
                rec.category = CAT_KEPT_LOW_SIM

        # Tổng hợp
        short_sims = [c.similarity for c in records if c.similarity is not None and len(c.ocr_text) < SHORT_CELL_LEN]
        long_sims = [c.similarity for c in records if c.similarity is not None and len(c.ocr_text) >= SHORT_CELL_LEN]
        result.short_sim = round(sum(short_sims) / len(short_sims), 4) if short_sims else None
        result.long_sim = round(sum(long_sims) / len(long_sims), 4) if long_sims else None
        result.n_short_cells = len(short_sims)
        result.n_long_cells = len(long_sims)

        # Chỉ số token: trước (OCR) và sau (đã vá)
        pw = self._page_words(page, key=page_idx)
        result.token_coverage_before = token_coverage(ocr_html, pw)
        final_html = replace_td_texts(ocr_html, [c.final_text for c in records])
        result.token_coverage_after = token_coverage(final_html, pw)

        result.repaired_html = final_html
        result.status = STATUS_REPAIRED
        return result

    def _page_words(self, page, key: int | None = None) -> set[str]:
        cache_key = key if key is not None else id(page)
        cached = self._chars_cache.get(cache_key)
        text = "".join(c[0] for c in cached) if cached is not None else None
        if text is None:
            tp = page.get_textpage()
            import pypdfium2.raw as pdfium_raw

            n = pdfium_raw.FPDFText_CountChars(tp)
            text = "".join(chr(pdfium_raw.FPDFText_GetUnicode(tp, i)) for i in range(n))
        return page_word_set(text)
