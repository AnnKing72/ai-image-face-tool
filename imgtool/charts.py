"""圖表繪製（Altair，Streamlit 內建支援）。"""

from __future__ import annotations

import math

import altair as alt
import numpy as np
import pandas as pd

CHANNEL_COLORS = {
    "R": "#e5484d",
    "G": "#30a46c",
    "B": "#3e63dd",
    "Y": "#6b7280",
    "U": "#8e4ec6",
    "V": "#f76b15",
}

_TICKS = [0, 32, 64, 96, 128, 160, 192, 224, 255]


def histogram_chart(
    histograms: dict[str, np.ndarray],
    *,
    log_scale: bool = False,
    height: int = 300,
) -> alt.LayerChart | alt.Chart:
    """直方圖：橫軸像素強度 0–255，縱軸像素數量。

    單一通道畫成長條；多個通道（原始影像的 R/G/B）疊在一起畫成半透明階梯面積圖。
    每個強度 v 佔據 [v, v+1) 的寬度，所以 0 與 255 兩端的長條都完整可見。
    """
    frames = []
    for name, hist in histograms.items():
        counts = np.asarray(hist, dtype=np.int64)
        frames.append(
            pd.DataFrame({"強度": np.arange(256), "結束": np.arange(1, 257), "像素數量": counts, "通道": name})
        )
    data = pd.concat(frames, ignore_index=True)
    names = list(histograms)
    color = alt.Color(
        "通道:N",
        scale=alt.Scale(domain=names, range=[CHANNEL_COLORS.get(n, "#6b7280") for n in names]),
        legend=alt.Legend(orient="top", title=None) if len(names) > 1 else None,
    )
    x = alt.X(
        "強度:Q",
        title="像素強度（0–255）",
        scale=alt.Scale(domain=[0, 256], nice=False, zero=False),
        axis=alt.Axis(values=_TICKS, grid=False),
    )
    y = alt.Y(
        "像素數量:Q",
        title="像素數量" + ("（對數刻度）" if log_scale else ""),
        scale=alt.Scale(type="symlog") if log_scale else alt.Scale(),
        stack=None,
    )
    tooltip = [
        alt.Tooltip("通道:N"),
        alt.Tooltip("強度:Q", title="強度"),
        alt.Tooltip("像素數量:Q", title="像素數量", format=","),
    ]
    base = alt.Chart(data).encode(x=x, y=y, color=color, tooltip=tooltip)
    if len(names) == 1:
        chart = base.mark_rect(opacity=0.85).encode(x2="結束:Q", y2=alt.datum(0))
    else:
        chart = base.mark_area(interpolate="step-after", opacity=0.28) + base.mark_line(
            interpolate="step-after", strokeWidth=1.3
        )
    return chart.properties(height=height)


def psnr_bar_chart(rows: list[dict], height: int | None = None) -> alt.Chart:
    """各預設濾波器的 PSNR 長條圖（∞ 的恆等濾波不畫，數值另外列在表格中）。"""
    finite = [r for r in rows if math.isfinite(r["psnr"])]
    data = pd.DataFrame(finite)
    chart = (
        alt.Chart(data)
        .mark_bar(color="#3e63dd", cornerRadiusEnd=3)
        .encode(
            x=alt.X("psnr:Q", title="PSNR (dB)，越高代表與原圖越接近"),
            y=alt.Y("name:N", title=None, sort="-x", axis=alt.Axis(labelLimit=260)),
            tooltip=[
                alt.Tooltip("name:N", title="濾波器"),
                alt.Tooltip("psnr:Q", title="PSNR (dB)", format=".2f"),
                alt.Tooltip("mse:Q", title="MSE", format=".2f"),
            ],
        )
    )
    text = chart.mark_text(align="left", dx=4, color="#374151").encode(text=alt.Text("psnr:Q", format=".1f"))
    return (chart + text).properties(height=height or max(160, 26 * len(finite)))
