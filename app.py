"""AI 輔助互動式影像處理與人臉分析工具（Streamlit 主程式）。

本機執行：streamlit run app.py
"""

from __future__ import annotations

import hashlib
import math
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from imgtool import __version__, charts, color, face, filtering, io_utils

# 部署後把 GitHub 儲存庫網址填在這裡，側邊欄就會顯示「原始碼」連結
SOURCE_CODE_URL = ""

APP_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = APP_DIR / "samples"
SAMPLES = {
    "咖啡（自然影像）": "coffee.png",
    "貓咪 Chelsea（自然影像）": "chelsea.png",
    "色彩測試圖（程式產生）": "color_chart.png",
}
TABS = ["① 影像資訊", "② 色彩空間與直方圖", "③ 卷積濾波與 PSNR", "④ 人臉偵測"]
DISPLAY_MAX_SIDE = 1400  # 畫面預覽的長邊上限（運算仍使用完整的處理解析度）
CUSTOM_PRESET = "（自訂，手動輸入）"
DEFAULT_PRESET = "銳化（4 鄰域）"

st.set_page_config(
    page_title="AI 影像處理與人臉分析工具",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)
# 卷積核輸入框改用等寬字型，矩陣的欄位才對得齊
st.html("<style>.st-key-kernel_text textarea { font-family: 'Source Code Pro', monospace; }</style>")


# ───────────────────────────── 快取包裝 ─────────────────────────────
# 參數名稱以底線開頭的引數不參與快取雜湊，改用 image_id（檔案 SHA-1）辨識影像，又快又精確。


@st.cache_data(show_spinner="讀取影像中…", max_entries=6)
def prepare_image(data: bytes, file_name: str, max_side: int):
    rgb_full, info = io_utils.load_image(data, file_name)
    rgb, scale = io_utils.limit_size(rgb_full, max_side)
    image_id = f"{hashlib.sha1(data).hexdigest()[:16]}-{max_side}"
    return rgb, info, scale, image_id


@st.cache_data(show_spinner=False, max_entries=8)
def cached_yuv(image_id: str, _rgb: np.ndarray, standard: str) -> color.YUVResult:
    return color.rgb_to_yuv(_rgb, standard)


@st.cache_data(show_spinner="濾波運算中…", max_entries=24)
def cached_filter(image_id: str, _rgb: np.ndarray, kernel_rows: tuple, params: filtering.FilterParams):
    result = filtering.apply_filter(_rgb, np.array(kernel_rows, dtype=np.float64), params)
    scores = {
        "psnr": filtering.psnr(_rgb, result.image),
        "mse": filtering.mse(_rgb, result.image),
        "per_channel": filtering.psnr_per_channel(_rgb, result.image),
    }
    return result, scores


@st.cache_data(show_spinner="批次比較所有預設濾波器中…", max_entries=4)
def cached_compare(image_id: str, _rgb: np.ndarray, flip: bool, border: str) -> list[dict]:
    rows = []
    for preset in filtering.PRESETS:
        result = filtering.apply_filter(
            _rgb, np.array(preset.kernel, dtype=np.float64), filtering.params_for(preset, flip=flip, border=border)
        )
        rows.append(
            {
                "name": preset.name,
                "psnr": filtering.psnr(_rgb, result.image),
                "mse": filtering.mse(_rgb, result.image),
                "ms": result.elapsed_ms,
                "method": result.method,
            }
        )
    return rows


@st.cache_resource(show_spinner="載入 YuNet 人臉偵測模型…")
def get_yunet() -> face.YuNetDetector:
    return face.YuNetDetector()


@st.cache_resource(show_spinner="載入 Haar 分類器…")
def get_haar() -> face.HaarDetector:
    return face.HaarDetector()


@st.cache_data(show_spinner="偵測人臉中…", max_entries=16)
def cached_yunet(image_id: str, _rgb: np.ndarray, score: float, nms: float, multiscale: bool) -> face.DetectionResult:
    return get_yunet().detect(_rgb, score_threshold=score, nms_threshold=nms, multiscale=multiscale)


@st.cache_data(show_spinner="偵測人臉中…", max_entries=16)
def cached_haar(image_id: str, _rgb: np.ndarray, scale_factor: float, min_neighbors: int, min_size: int):
    return get_haar().detect(_rgb, scale_factor=scale_factor, min_neighbors=min_neighbors, min_size=min_size)


# ───────────────────────────── 小工具 ─────────────────────────────


def preview(img: np.ndarray) -> np.ndarray:
    """縮小到適合畫面顯示的尺寸，減少傳到瀏覽器的資料量。"""
    height, width = img.shape[:2]
    longest = max(height, width)
    if longest <= DISPLAY_MAX_SIDE:
        return img
    s = DISPLAY_MAX_SIDE / longest
    return cv2.resize(img, (max(1, round(width * s)), max(1, round(height * s))), interpolation=cv2.INTER_AREA)


def fmt_db(value: float) -> str:
    return "∞ dB" if math.isinf(value) else f"{value:.2f} dB"


def fmt_num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.6g}"


def stem(file_name: str) -> str:
    return Path(file_name).stem or "image"


# ───────────────────────────── 側邊欄 ─────────────────────────────


def sidebar() -> tuple[bytes, str, int]:
    with st.sidebar:
        st.header("影像來源")
        upload = st.file_uploader(
            "上傳影像",
            type=list(io_utils.SUPPORTED_EXTENSIONS),
            key="upload",
            help="支援 PNG、JPG/JPEG、BMP，單檔上限 50 MB。",
        )
        if upload is None:
            sample = st.selectbox("或選擇範例影像", list(SAMPLES), key="sample")
            data, name = (SAMPLES_DIR / SAMPLES[sample]).read_bytes(), SAMPLES[sample]
        else:
            data, name = upload.getvalue(), upload.name
            st.caption("移除上傳的檔案即可改回範例影像。")

        st.header("處理設定")
        max_side = st.select_slider(
            "處理解析度上限（長邊像素）",
            options=[512, 1024, 1600, 2048, 3072, 4096],
            value=2048,
            key="max_side",
            help="大圖會先等比例縮小到這個尺寸再運算，避免雲端主機記憶體不足；「影像資訊」仍顯示原始檔案的數值。",
        )
        st.divider()
        st.caption(
            f"版本 {__version__}｜Python {platform.python_version()}｜"
            f"OpenCV {cv2.__version__}｜NumPy {np.__version__}｜Streamlit {st.__version__}"
        )
        if SOURCE_CODE_URL:
            st.link_button("原始碼（GitHub）", SOURCE_CODE_URL, icon=":material/code:", width="stretch")
    return data, name, max_side


# ───────────────────────────── ① 影像資訊 ─────────────────────────────


def render_info(rgb: np.ndarray, info: io_utils.ImageInfo, image_id: str) -> None:
    height, width = rgb.shape[:2]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("解析度（寬 × 高）", f"{info.width} × {info.height}")
    m2.metric(
        "色彩維度",
        f"{info.channels} 通道 × {info.bits_per_channel} bit",
        help=f"{info.mode}：{info.mode_description}",
    )
    m3.metric("檔案大小", io_utils.human_size(info.file_size), help=f"{info.file_size:,} bytes")
    m4.metric("檔案格式", info.format)

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.image(preview(rgb), caption=f"{info.file_name}（處理尺寸 {width} × {height}）", width="stretch")
    with right:
        st.subheader("中繼資訊")
        rows = [
            ("檔名", info.file_name),
            ("檔案格式", info.format),
            ("解析度", f"{info.width} × {info.height} px"),
            ("總像素數", f"{info.pixel_count:,}"),
            ("色彩模式", f"{info.mode}（{info.mode_description}）"),
            ("色彩維度（通道數）", str(info.channels)),
            ("位元深度", f"{info.bits_per_channel} bit / 通道，{info.bits_per_pixel} bit / 像素"),
            ("透明通道（Alpha）", "有" if info.has_alpha else "無"),
            ("檔案大小", f"{io_utils.human_size(info.file_size)}（{info.file_size:,} bytes）"),
            ("未壓縮資料量", f"{io_utils.human_size(info.raw_size)}（壓縮率 {info.compression_ratio:.1f} : 1）"),
            ("解析度標記（DPI）", f"{info.dpi[0]:.0f} × {info.dpi[1]:.0f}" if info.dpi else "未標記"),
            ("EXIF 方向", "已自動轉正" if info.exif_transposed else "不需轉正"),
            ("處理用陣列（H, W, C）", f"{rgb.shape}　{rgb.dtype}"),
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["項目", "數值"]),
            hide_index=True,
            width="stretch",
            height=36 * (len(rows) + 1) + 2,
        )
        for note in info.notes:
            st.caption(f"• {note}")


# ───────────────────────────── ② 色彩空間與直方圖 ─────────────────────────────

VIEW_LABELS = {
    "Original": "原始影像",
    "R": "R 紅",
    "G": "G 綠",
    "B": "B 藍",
    "Y": "Y 亮度",
    "U": "U 色度",
    "V": "V 色度",
}


def build_view(rgb: np.ndarray, yuv: np.ndarray, view: str, pseudo: bool, standard: str):
    """依顯示模式回傳 (要顯示的影像, {通道名: 直方圖}, 圖說)。"""
    if view == "Original":
        hists = {name: color.histogram(rgb[..., i]) for i, name in enumerate("RGB")}
        return rgb, hists, "原始影像（RGB 三通道疊合直方圖）"
    if view in "RGB":
        i = "RGB".index(view)
        image = color.tint_channel(rgb, i) if pseudo else rgb[..., i]
        mode = "單色顯示" if pseudo else "灰階強度"
        return image, {view: color.histogram(rgb[..., i])}, f"{VIEW_LABELS[view]} 分量（{mode}）"
    i = "YUV".index(view)
    channel = yuv[..., i]
    if i == 0:
        return channel, {view: color.histogram(channel)}, "Y 亮度分量（灰階）"
    image = color.chroma_preview(yuv, i, standard) if pseudo else channel
    mode = "偽彩色：亮度固定 128" if pseudo else "灰階強度，128 代表沒有色彩偏移"
    return image, {view: color.histogram(channel)}, f"{VIEW_LABELS[view]} 分量（{mode}）"


def stats_table(hists: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, hist in hists.items():
        s = color.channel_stats(hist)
        rows.append(
            {
                "通道": name,
                "平均": round(s.mean, 2),
                "標準差": round(s.std, 2),
                "最小": s.minimum,
                "最大": s.maximum,
                "中位數": s.median,
                "熵": round(s.entropy, 3),
            }
        )
    return pd.DataFrame(rows)


STATS_COLUMN_CONFIG = {"熵": st.column_config.NumberColumn("熵", help="資訊熵（bits），越大代表強度分布越分散")}


def render_color(rgb: np.ndarray, info: io_utils.ImageInfo, image_id: str) -> None:
    view = st.segmented_control(
        "顯示模式",
        list(VIEW_LABELS),
        format_func=VIEW_LABELS.get,
        default="Original",
        required=True,
        key="color_view",
        persist_state="page",
    )
    c1, c2, c3 = st.columns([1, 1, 1.3], vertical_alignment="center")
    pseudo = c1.toggle(
        "分量以色彩顯示",
        key="pseudo",
        persist_state="page",
        help="R/G/B 以該原色顯示；U/V 以「亮度固定、只保留該色度」換算回 RGB，看得出色度方向。",
    )
    log_scale = c2.toggle(
        "直方圖用對數刻度",
        key="hist_log",
        persist_state="page",
        help="像素數量差距很大時（例如 U/V 集中在 128 附近），對數刻度較容易看出分布的尾端。",
    )
    standard = c3.selectbox(
        "YUV 轉換定義",
        list(color.YUV_STANDARDS),
        format_func=lambda k: color.YUV_STANDARDS[k].label,
        key="yuv_std",
        persist_state="page",
        label_visibility="collapsed",
    )

    yuv_result = cached_yuv(image_id, rgb, standard)
    image, hists, caption = build_view(rgb, yuv_result.image, view, pseudo, standard)

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.image(preview(image), caption=caption, width="stretch")
    with right:
        st.markdown(f"**{VIEW_LABELS[view]}：強度分布直方圖**")
        st.altair_chart(charts.histogram_chart(hists, log_scale=log_scale), width="stretch")
        st.dataframe(stats_table(hists), hide_index=True, width="stretch", column_config=STATS_COLUMN_CONFIG)

    std = color.YUV_STANDARDS[standard]
    st.caption(f"轉換公式：Y = 0.299R + 0.587G + 0.114B；{std.description}")
    clipped = yuv_result.clipped
    if max(clipped) > 0:
        detail = "、".join(f"{n} 分量 {p:.2%}" for n, p in zip("YUV", clipped) if p > 0)
        st.caption(f"⚠️ 這張影像有像素超出 0–255 被截斷：{detail}（可改用 YCbCr 定義比較）。")

    with st.expander("並排總覽：RGB 與 YUV 全部分量"):
        for group in (("R", "G", "B"), ("Y", "U", "V")):
            cols = st.columns(3)
            for col, name in zip(cols, group):
                img, hist, _ = build_view(rgb, yuv_result.image, name, pseudo, standard)
                with col:
                    st.image(preview(img), caption=VIEW_LABELS[name], width="stretch")
                    st.altair_chart(charts.histogram_chart(hist, log_scale=log_scale, height=150), width="stretch")


# ───────────────────────────── ③ 卷積濾波與 PSNR ─────────────────────────────

KERNEL_HELP = """\
**輸入格式**
- 每一列換一行，數字用空白或逗號分隔，例如 `0 -1 0`
- 也可以用分號分列：`1 2 1; 2 4 2; 1 2 1`，或 `[[1,2,1],[2,4,2],[1,2,1]]`
- 支援負數、小數、科學記號（`1e-3`）與分數（`1/16`），全形數字也可以
- 支援任意 m×n 大小（含非方陣，例如 `1 1 1 1 1`），單邊最多 31
"""


def _load_preset_into_state(preset: filtering.Preset) -> None:
    ss = st.session_state
    ss.kernel_text = filtering.format_kernel(np.array(preset.kernel, dtype=np.float64))
    ss.norm_mode = preset.normalize
    ss.divisor = float(preset.divisor)
    ss.bias = float(preset.bias)
    ss.out_mode = preset.output


def _init_filter_state() -> None:
    ss = st.session_state
    if "kernel_text" not in ss:
        ss.preset = DEFAULT_PRESET
        _load_preset_into_state(filtering.PRESET_BY_NAME[DEFAULT_PRESET])
    ss.setdefault("preset", CUSTOM_PRESET)
    ss.setdefault("conv_mode", "conv")
    ss.setdefault("border", "reflect101")


def _on_preset_change() -> None:
    preset = filtering.PRESET_BY_NAME.get(st.session_state.preset)
    if preset is not None:
        _load_preset_into_state(preset)


def _on_kernel_edit() -> None:
    st.session_state.preset = CUSTOM_PRESET  # 手動修改後，範本選單改顯示「自訂」


def _on_gaussian() -> None:
    ss = st.session_state
    kernel = filtering.gaussian_kernel(int(ss.gauss_size), float(ss.gauss_sigma))
    ss.kernel_text = filtering.format_kernel(kernel, precision=4)
    ss.norm_mode, ss.bias, ss.out_mode = "sum", 0.0, "clip"
    ss.preset = CUSTOM_PRESET


def render_filter(rgb: np.ndarray, info: io_utils.ImageInfo, image_id: str) -> None:
    _init_filter_state()
    ss = st.session_state
    controls, results = st.columns([1, 1.7], gap="large")

    with controls:
        st.selectbox(
            "預設範本",
            [CUSTOM_PRESET] + [p.name for p in filtering.PRESETS],
            key="preset",
            on_change=_on_preset_change,
            persist_state="page",
            help="選擇後會自動填入下方矩陣與建議的正規化設定，之後仍可自由修改任何數值。",
        )
        preset = filtering.PRESET_BY_NAME.get(ss.preset)
        if preset is not None:
            st.caption(preset.description)
        st.text_area(
            "卷積核矩陣",
            key="kernel_text",
            height=150,
            on_change=_on_kernel_edit,
            persist_state="page",
            help=KERNEL_HELP,
        )
        kernel = None
        try:
            kernel = filtering.parse_kernel(ss.kernel_text)
        except filtering.KernelError as exc:
            st.error(str(exc), icon="⚠️")
        if kernel is not None:
            st.caption(
                f"尺寸 {kernel.shape[0]} × {kernel.shape[1]}｜係數總和 {fmt_num(kernel.sum())}｜"
                f"非零係數 {np.count_nonzero(kernel)} 個"
            )

        with st.expander("高斯核產生器"):
            g1, g2 = st.columns(2)
            g1.selectbox("尺寸", [3, 5, 7, 9, 11, 13, 15], index=1, key="gauss_size", persist_state="page")
            g2.slider("σ（標準差）", 0.3, 5.0, 1.0, 0.1, key="gauss_sigma", persist_state="page")
            st.button("產生並填入矩陣", on_click=_on_gaussian, width="stretch")

        st.markdown("**正規化與輸出**")
        st.radio(
            "正規化方式",
            list(filtering.NORMALIZE_MODES),
            format_func=filtering.NORMALIZE_MODES.get,
            key="norm_mode",
            persist_state="page",
        )
        if ss.norm_mode == "custom":
            st.number_input(
                "自訂除數 d（實際係數 = 輸入值 ÷ d）",
                step=1.0,
                format="%g",
                key="divisor",
                persist_state="page",
            )
        st.number_input(
            "偏移量（濾波後每個像素再加上此值）",
            min_value=-1000.0,
            max_value=1000.0,
            step=8.0,
            format="%g",
            key="bias",
            persist_state="page",
            help="浮雕效果常用 128，讓 0 變成中間灰。",
        )
        st.radio(
            "超出 0–255 的數值",
            list(filtering.OUTPUT_MODES),
            format_func=filtering.OUTPUT_MODES.get,
            key="out_mode",
            persist_state="page",
        )
        with st.expander("進階：運算方式與邊界處理"):
            st.radio(
                "運算方式",
                ["conv", "corr"],
                format_func={
                    "conv": "卷積（核先旋轉 180°，數學定義）",
                    "corr": "相關運算（不旋轉，等同 OpenCV filter2D）",
                }.get,
                key="conv_mode",
                persist_state="page",
                help="對稱的核（模糊、銳化、拉普拉斯）兩者結果相同；Sobel、浮雕等不對稱的核方向會相反。",
            )
            st.selectbox(
                "邊界處理（Padding）",
                list(filtering.BORDER_MODES),
                format_func=lambda k: filtering.BORDER_MODES[k][0],
                key="border",
                persist_state="page",
            )

    with results:
        if kernel is None:
            st.info("修正左側的卷積核後，這裡就會顯示濾波結果。", icon="✏️")
            st.image(preview(rgb), caption="原始影像", width="stretch")
        else:
            params = filtering.FilterParams(
                normalize=ss.norm_mode,
                divisor=float(ss.get("divisor", 1.0)),
                bias=float(ss.bias),
                output=ss.out_mode,
                flip=ss.conv_mode == "conv",
                border=ss.border,
            )
            try:
                result, scores = cached_filter(image_id, rgb, tuple(map(tuple, kernel.tolist())), params)
            except filtering.KernelError as exc:
                st.error(str(exc), icon="⚠️")
            else:
                show_filter_result(rgb, info, result, scores)

    st.divider()
    with st.expander("PSNR 公式與計算方式"):
        st.latex(
            r"\mathrm{MSE}=\frac{1}{M\,N\,C}\sum_{c=1}^{C}\sum_{y=1}^{M}\sum_{x=1}^{N}"
            r"\bigl[I(x,y,c)-K(x,y,c)\bigr]^{2}"
        )
        st.latex(
            r"\mathrm{PSNR}=10\log_{10}\!\left(\frac{\mathrm{MAX}_I^{2}}{\mathrm{MSE}}\right)"
            r"=20\log_{10}\mathrm{MAX}_I-10\log_{10}\mathrm{MSE}\quad(\mathrm{dB})"
        )
        st.markdown(
            "- $I$ 為原圖、$K$ 為濾波後影像，$M \\times N$ 為解析度，$C=3$ 為通道數，"
            "8-bit 影像的 $\\mathrm{MAX}_I = 255$。\n"
            "- 兩張圖完全相同時 MSE = 0、分母為 0，程式直接回傳 PSNR = ∞，避免除以零。\n"
            "- 經驗值：40 dB 以上幾乎看不出差異；30–40 dB 品質良好；20 dB 以下差異明顯。\n"
            "- PSNR 衡量的是「和原圖差多少」。銳化、邊緣偵測本來就會大幅改變影像，"
            "PSNR 低不代表濾波效果不好，要搭配目視結果一起判讀。"
        )

    with st.expander("批次比較：所有預設濾波器的 PSNR（客觀評估用）"):
        st.caption("以目前的影像、運算方式與邊界處理，一次套用全部預設範本並比較 PSNR，方便寫入報告。")
        if st.toggle("執行批次比較", key="run_compare", persist_state="page"):
            rows = cached_compare(image_id, rgb, ss.conv_mode == "conv", ss.border)
            table = pd.DataFrame(
                {
                    "濾波器": [r["name"] for r in rows],
                    "PSNR (dB)": [fmt_db(r["psnr"]).replace(" dB", "") for r in rows],
                    "MSE": [round(r["mse"], 2) for r in rows],
                    "方法": ["FFT" if r["method"] == "fft" else "直接卷積" for r in rows],
                    "耗時 (ms)": [round(r["ms"], 1) for r in rows],
                }
            )
            st.dataframe(table, hide_index=True, width="stretch", height=36 * (len(table) + 1) + 2)
            st.altair_chart(charts.psnr_bar_chart(rows), width="stretch")


def show_filter_result(rgb, info, result: filtering.FilterResult, scores: dict) -> None:
    method = "直接卷積（逐係數平移相加）" if result.method == "direct" else "FFT（卷積定理，頻率域相乘）"
    m1, m2, m3 = st.columns(3)
    m1.metric("PSNR", fmt_db(scores["psnr"]), help="以原圖為基準；兩圖完全相同時 MSE = 0，PSNR 為 ∞。")
    m2.metric("MSE", f"{scores['mse']:.2f}")
    m3.metric(
        "運算時間",
        f"{result.elapsed_ms:.0f} ms",
        help="非零係數少時用直接卷積；大型稠密核（約 7×7 以上）自動改用 FFT，速度與核大小無關。",
    )

    for warning in result.warnings:
        st.warning(warning, icon="⚠️")
    summary = f"運算方法：{method}｜映射前數值範圍 {result.response_min:.1f} ～ {result.response_max:.1f}"
    clipped = [
        f"{ratio:.2%} 被截斷成 {value}"
        for ratio, value in ((result.clipped_low, 0), (result.clipped_high, 255))
        if ratio > 0
    ]
    if clipped:
        summary += "；輸出時有 " + "、".join(clipped)
    st.caption(summary + "。")

    mode = st.segmented_control(
        "比對方式",
        ["side", "split", "diff"],
        format_func={"side": "並排比較", "split": "滑桿分割比對", "diff": "差異圖"}.get,
        default="side",
        required=True,
        key="cmp_view",
        persist_state="page",
    )
    if mode == "side":
        a, b = st.columns(2)
        a.image(preview(rgb), caption="濾波前（原圖）", width="stretch")
        b.image(preview(result.image), caption="濾波後", width="stretch")
    elif mode == "split":
        position = st.slider("分割線位置（左：原圖｜右：濾波後）", 0, 100, 50, key="split_pos", persist_state="page")
        st.image(filtering.split_view(preview(rgb), preview(result.image), position / 100), width="stretch")
    else:
        gain = st.select_slider("差異放大倍率", options=[1, 2, 4, 8, 16], value=4, key="diff_gain", persist_state="page")
        st.image(
            preview(filtering.difference_image(rgb, result.image, gain)),
            caption=f"|原圖 − 濾波後| × {gain}（越亮代表差異越大）",
            width="stretch",
        )

    per_channel = "、".join(f"{n} {fmt_db(v)}" for n, v in zip("RGB", scores["per_channel"]))
    st.caption(f"各通道 PSNR：{per_channel}")
    if not math.isclose(result.factor, 1.0):
        with st.expander(f"實際使用的係數（已乘上正規化係數 {result.factor:.6g}）"):
            st.dataframe(pd.DataFrame(result.effective_kernel).round(6), width="stretch")
    st.download_button(
        "下載濾波結果（PNG）",
        data=lambda img=result.image: io_utils.to_png_bytes(img),
        file_name=f"{stem(info.file_name)}_filtered.png",
        mime="image/png",
        on_click="ignore",
        icon=":material/download:",
    )


# ───────────────────────────── ④ 人臉偵測 ─────────────────────────────

METHOD_LABELS = {"yunet": "YuNet 深度學習", "haar": "Haar 經典特徵"}


def render_face(rgb: np.ndarray, info: io_utils.ImageInfo, image_id: str) -> None:
    ss = st.session_state
    c1, c2 = st.columns([1.25, 2.3], gap="large")
    with c1:
        method = st.segmented_control(
            "偵測方法",
            list(METHOD_LABELS),
            format_func=METHOD_LABELS.get,
            default="yunet",
            required=True,
            key="face_method",
            persist_state="page",
        )
    with c2:
        if method == "yunet":
            a, b = st.columns(2)
            a.slider(
                "信心度門檻", 0.30, 0.99, 0.70, 0.01, key="yn_score", persist_state="page",
                help="低於門檻的候選框不顯示。調高可減少誤判，調低可找回模糊、側臉等較難的人臉。",
            )
            b.slider(
                "NMS 重疊門檻（IoU）", 0.10, 0.90, 0.30, 0.05, key="yn_nms", persist_state="page",
                help="兩個框重疊程度（交集 ÷ 聯集）超過此值時，只保留分數較高的那個。",
            )
            a.toggle(
                "多尺度偵測（影像金字塔）", value=True, key="yn_multiscale", persist_state="page",
                help="YuNet 對 10～300 px 的臉最準。開啟後會在多個縮放比例偵測再合併，大臉、小臉都穩定。",
            )
            b.toggle("標示五官關鍵點", value=True, key="yn_landmarks", persist_state="page")
        else:
            a, b, c = st.columns(3)
            a.slider(
                "scaleFactor", 1.05, 1.50, 1.10, 0.01, key="haar_scale", persist_state="page",
                help="影像金字塔每層縮小的比例。越接近 1 越仔細，但越慢。",
            )
            b.slider(
                "minNeighbors", 1, 15, 5, 1, key="haar_neighbors", persist_state="page",
                help="一個位置至少要被偵測到幾次才算人臉。越大誤判越少，但可能漏抓。",
            )
            c.slider("最小人臉（px）", 10, 200, 30, 5, key="haar_minsize", persist_state="page")

    try:
        if method == "yunet":
            result = cached_yunet(image_id, rgb, float(ss.yn_score), float(ss.yn_nms), bool(ss.yn_multiscale))
        else:
            result = cached_haar(
                image_id, rgb, float(ss.haar_scale), int(ss.haar_neighbors), int(ss.haar_minsize)
            )
    except (cv2.error, FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(f"人臉偵測失敗：{exc}", icon="🚫")
        return

    is_yunet = method == "yunet"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("偵測到的人臉數", f"{result.count} 張")
    if result.count:
        mean_label = f"{result.mean_score:.3f}" if is_yunet else f"{result.mean_score:.2f}"
    else:
        mean_label = "—"
    m2.metric(
        "平均信心度" if is_yunet else "平均 levelWeight",
        mean_label,
        help="YuNet 輸出 0～1 的信心分數。" if is_yunet else "Haar 的 levelWeight 是最後一級分類器的加權分數，越大越像人臉，但不是機率。",
    )
    m3.metric("推論時間", f"{result.elapsed_ms:.0f} ms")
    if is_yunet:
        m4.metric("偵測尺度", f"{len(result.scales)} 層" if len(result.scales) > 1 else "單一尺度",
                  help="各層長邊：" + "、".join(f"{round(max(result.image_size) * s)} px" for s in result.scales))
    else:
        m4.metric("偵測解析度", f"{result.image_size[0]} × {result.image_size[1]}")

    show_landmarks = is_yunet and bool(ss.get("yn_landmarks", True))
    annotated = face.draw_faces(rgb, result.faces, show_landmarks=show_landmarks)
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.image(preview(annotated), caption=f"偵測結果：{result.count} 張人臉（#編號 由左到右）", width="stretch")
    with right:
        if result.count == 0:
            if info.file_name in SAMPLES.values():
                st.info("範例影像裡沒有人臉。請從左側上傳含人臉的照片（例如合照）來測試多人臉偵測。", icon="📷")
            else:
                st.info(
                    "沒有偵測到人臉。可以試著降低信心度門檻、改用另一種方法，"
                    "或換一張正臉、光線充足的照片。",
                    icon="🔍",
                )
        else:
            table = pd.DataFrame(
                {
                    "編號": [f"#{i}" for i in range(1, result.count + 1)],
                    "x": [round(f.x) for f in result.faces],
                    "y": [round(f.y) for f in result.faces],
                    "寬": [round(f.w) for f in result.faces],
                    "高": [round(f.h) for f in result.faces],
                    ("信心度" if is_yunet else "levelWeight"): [round(f.score, 4) for f in result.faces],
                }
            )
            score_column = "信心度" if is_yunet else "levelWeight"
            column_config = {score_column: st.column_config.NumberColumn(score_column, format="%.3f")}
            st.dataframe(table, hide_index=True, width="stretch", column_config=column_config)
            if show_landmarks:
                legend = "、".join(
                    f"<span style='color:rgb{c}'>●</span> {n}" for n, c in zip(face.LANDMARK_NAMES, face.LANDMARK_COLORS)
                )
                st.caption(f"關鍵點：{legend}（左右以照片中的人為準）", unsafe_allow_html=True)
        st.download_button(
            "下載標註結果（PNG）",
            data=lambda img=annotated: io_utils.to_png_bytes(img),
            file_name=f"{stem(info.file_name)}_faces.png",
            mime="image/png",
            on_click="ignore",
            icon=":material/download:",
        )

    with st.expander("方法說明"):
        st.markdown(
            "**YuNet**（OpenCV `FaceDetectorYN`）：約 230 KB 的輕量 CNN 偵測器，"
            "在 WIDER Face 驗證集達到 AP 0.884（easy）/ 0.866（medium）/ 0.750（hard）。"
            "每張臉輸出邊界框、0～1 的信心度與 5 個關鍵點。官方說明它對 10～300 px 的臉最準，"
            "因此本工具預設使用影像金字塔：每次縮小一半重新偵測，各層只保留可靠尺寸範圍內的框，"
            "再把座標換算回原圖、用 NMS 合併重複的框。\n\n"
            "**Haar Cascade**（Viola–Jones）：以積分影像快速計算 Haar-like 特徵，"
            "透過 AdaBoost 串接多級分類器篩選候選框，再以 minNeighbors 合併重疊結果。"
            "速度快、不需深度學習框架，但對側臉、遮擋、光線變化較敏感。"
            "信心分數取 `detectMultiScale3` 的 levelWeight。"
        )


# ───────────────────────────── 主流程 ─────────────────────────────


def main() -> None:
    data, name, max_side = sidebar()
    st.title("AI 影像處理與人臉分析工具")
    st.caption("影像載入與解析 · RGB / YUV 色彩空間與直方圖 · 自訂卷積核濾波與 PSNR · 人臉偵測（YuNet / Haar）")

    try:
        rgb, info, scale, image_id = prepare_image(data, name, max_side)
    except io_utils.ImageLoadError as exc:
        st.error(f"**讀取失敗：** {exc}", icon="🚫")
        st.info("請在左側重新上傳 PNG、JPG/JPEG 或 BMP 檔，或移除檔案改用範例影像。")
        st.stop()

    height, width = rgb.shape[:2]
    if scale < 1:
        st.info(
            f"原圖 {info.width} × {info.height} 超過處理上限，已等比例縮小為 {width} × {height}"
            f"（× {scale:.3f}）後再運算。可在左側調整上限。",
            icon="ℹ️",
        )

    tabs = st.tabs(TABS, key="tab", bind="query-params")
    for tab, render in zip(tabs, (render_info, render_color, render_filter, render_face)):
        if tab.open:
            with tab:
                render(rgb, info, image_id)


main()
