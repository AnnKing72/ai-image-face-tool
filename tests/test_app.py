"""Streamlit 介面自動化測試（streamlit.testing.v1.AppTest，不需要開瀏覽器）。

驗證四個分頁都能正常渲染、防呆訊息正確顯示，以及 samples/edge_cases 的每個邊界檔案
上傳後在每個分頁都不會讓程式當掉。
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from imgtool import filtering

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
EDGE = ROOT / "samples" / "edge_cases"
TABS = ["① 影像資訊", "② 色彩空間與直方圖", "③ 卷積濾波與 PSNR", "④ 人臉偵測"]
CUSTOM = "（自訂，手動輸入）"


def open_app(tab: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    if tab:
        at.query_params["tab"] = tab  # 分頁綁定網址參數，可直接開在指定分頁
    return at.run()


def assert_ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("sample", ["咖啡（自然影像）", "貓咪 Chelsea（自然影像）", "色彩測試圖（程式產生）"])
@pytest.mark.parametrize("tab", TABS)
def test_every_tab_renders(tab, sample):
    at = open_app(tab)
    at.selectbox(key="sample").set_value(sample).run()
    assert_ok(at)
    assert not at.error


def test_info_tab_metadata():
    at = open_app(TABS[0])
    values = {m.label: m.value for m in at.metric}
    assert values["解析度（寬 × 高）"] == "600 × 400"
    assert values["色彩維度"] == "3 通道 × 8 bit"
    assert values["檔案格式"] == "PNG"


def test_color_tab_all_views_and_options():
    at = open_app(TABS[1])
    for view in ["R", "G", "B", "Y", "U", "V", "Original"]:
        at.segmented_control(key="color_view").set_value(view).run()
        assert_ok(at)
    at.toggle(key="pseudo").set_value(True).run()
    at.toggle(key="hist_log").set_value(True).run()
    at.selectbox(key="yuv_std").set_value("ycbcr").run()
    for view in ["R", "U", "V"]:
        at.segmented_control(key="color_view").set_value(view).run()
        assert_ok(at)


@pytest.mark.parametrize(
    "text, message",
    [
        ("1 2 3\n4 5", "元素個數必須相同"),
        ("1/0 1", "分母為 0"),
        ("1 abc 2", "不是有效數字"),
        ("", "空的"),
        ("1e9", "數值過大"),
    ],
)
def test_filter_invalid_kernels_show_errors(text, message):
    at = open_app(TABS[2])
    at.text_area(key="kernel_text").set_value(text).run()
    assert_ok(at)
    assert any(message in e.value for e in at.error)
    assert at.selectbox(key="preset").value == CUSTOM  # 手動修改後範本選單切到「自訂」


def test_filter_identity_gives_infinite_psnr():
    at = open_app(TABS[2])
    at.selectbox(key="preset").set_value("恆等（Identity）").run()
    assert_ok(at)
    assert next(m for m in at.metric if m.label == "PSNR").value == "∞ dB"


def test_filter_division_by_zero_guards():
    at = open_app(TABS[2])
    at.selectbox(key="preset").set_value("拉普拉斯邊緣（4 鄰域）").run()
    at.radio(key="norm_mode").set_value("sum").run()
    assert_ok(at)
    assert any("總和為 0" in w.value for w in at.warning)
    at.radio(key="norm_mode").set_value("custom").run()
    at.number_input(key="divisor").set_value(0.0).run()
    assert_ok(at)
    assert any("不可為 0" in e.value for e in at.error)


def test_filter_non_square_and_large_kernels():
    at = open_app(TABS[2])
    at.selectbox(key="preset").set_value("動態模糊（水平 1×9）").run()
    assert_ok(at)
    assert not at.error
    at.text_area(key="kernel_text").set_value("\n".join(" ".join(["1"] * 31) for _ in range(31))).run()
    at.radio(key="norm_mode").set_value("sum").run()
    assert_ok(at)
    assert not at.error


def test_filter_views_options_and_batch_compare():
    at = open_app(TABS[2])
    for mode in ["split", "diff", "side"]:
        at.segmented_control(key="cmp_view").set_value(mode).run()
        assert_ok(at)
    at.radio(key="out_mode").set_value("minmax").run()
    at.radio(key="conv_mode").set_value("corr").run()
    for border in filtering.BORDER_MODES:
        at.selectbox(key="border").set_value(border).run()
        assert_ok(at)
    at.toggle(key="run_compare").set_value(True).run()
    assert_ok(at)
    assert any(len(df.value) == len(filtering.PRESETS) for df in at.dataframe)


def test_gaussian_generator():
    at = open_app(TABS[2])
    at.selectbox(key="gauss_size").set_value(7).run()
    next(b for b in at.button if b.label == "產生並填入矩陣").click().run()
    assert_ok(at)
    assert len(at.text_area(key="kernel_text").value.strip().splitlines()) == 7
    assert at.radio(key="norm_mode").value == "sum"


def test_face_tab_methods_and_options():
    at = open_app(TABS[3])
    assert_ok(at)
    at.toggle(key="yn_multiscale").set_value(False).run()
    at.slider(key="yn_score").set_value(0.3).run()
    assert_ok(at)
    at.segmented_control(key="face_method").set_value("haar").run()
    assert_ok(at)
    at.slider(key="haar_neighbors").set_value(1).run()
    assert_ok(at)


EDGE_FILES = sorted(p.name for p in EDGE.iterdir() if p.is_file() and p.suffix != ".md")


@pytest.mark.parametrize("tab", TABS)
@pytest.mark.parametrize("name", EDGE_FILES)
def test_upload_edge_case_files_on_every_tab(name, tab):
    at = open_app(tab)
    at.file_uploader(key="upload").upload(name, (EDGE / name).read_bytes()).run()
    assert_ok(at)
    if name.startswith(("11_", "12_")):  # 損毀檔、假圖片：要顯示友善的錯誤訊息
        assert any("讀取失敗" in e.value for e in at.error)
    else:
        assert not at.error
