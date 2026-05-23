import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont


def fs_pt_to_px(size_pt: int, dpi: int) -> int:
    return max(1, int(round(size_pt * dpi / 72.0)))


def try_load_font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesib.ttf" if bold else r"C:\Windows\Fonts\timesi.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def text_size(draw, text, font):
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return x1 - x0, y1 - y0


def draw_text_top_left(draw, x, y, text, font, fill):
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    draw.text((x - x0, y - y0), text, font=font, fill=fill)


def draw_text_centered(draw, cx, cy, text, font, fill):
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    w = x1 - x0
    h = y1 - y0
    draw.text((cx - w // 2 - x0, cy - h // 2 - y0), text, font=font, fill=fill)


def draw_dashed_vline(draw, x, y0, y1, color, width, dash, gap):
    y = y0
    while y < y1:
        y2 = min(y1, y + dash)
        draw.line([x, y, x, y2], fill=color, width=width)
        y = y2 + gap


def rotated_text_image(draw, text, font, fill, angle):
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    tw = x1 - x0
    th = y1 - y0
    pad = max(4, int(round(min(tw, th) * 0.20)))
    text_img = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    td = ImageDraw.Draw(text_img)
    td.text((pad - x0, pad - y0), text, font=font, fill=fill)
    return text_img.rotate(angle, expand=True, resample=Image.BICUBIC)


def draw_grouped_bar_chart(
    out_path: Path,
    fig_w: float,
    fig_h: float,
    dpi: int,
    title: str,
    y_label: str,
    x_label: str,
    categories: List[str],
    series: List[Tuple[str, str, List[Optional[float]]]],
    ymin: float,
    ymax: float,
    ytick_step: float,
    title_fs: int,
    tick_fs: int,
    label_fs: int,
    value_fs: int,
    stamp: bool,
    group_annotations: Optional[List[str]] = None,
    group_annotation_color: str = "#12b886",
):
    w_px = max(1, int(round(fig_w * dpi)))
    h_px = max(1, int(round(fig_h * dpi)))

    img = Image.new("RGBA", (w_px, h_px), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_title = try_load_font(fs_pt_to_px(title_fs, dpi), bold=True)
    font_tick = try_load_font(fs_pt_to_px(tick_fs, dpi), bold=False)
    font_label = try_load_font(fs_pt_to_px(label_fs, dpi), bold=False)
    font_value = try_load_font(fs_pt_to_px(value_fs, dpi), bold=True)

    title_w, title_h = text_size(draw, title, font_title)
    pad = max(10, int(round(min(w_px, h_px) * 0.02)))

    tick_texts = []
    t = ymin
    while t <= ymax + 1e-9:
        tick_texts.append(f"{t:.0f}")
        t += ytick_step
    ytick_w_max = max((text_size(draw, txt, font_tick)[0] for txt in tick_texts), default=0)
    xcat_h_max = max((text_size(draw, c, font_tick)[1] for c in categories), default=0)
    x_label_w, x_label_h = text_size(draw, x_label, font_label)

    rot_y = rotated_text_image(draw, y_label, font_label, "#000000", angle=90)

    legend_entries = [name for name, _, _ in series]
    legend_w = 0
    legend_h = 0
    legend_box = max(8, int(round(h_px * 0.018)))
    for name in legend_entries:
        tw, th = text_size(draw, name, font_tick)
        legend_w = max(legend_w, legend_box + pad // 2 + tw)
        legend_h += max(th, legend_box) + pad // 4

    left_margin = pad + rot_y.size[0] + pad + ytick_w_max + pad
    right_margin = pad + legend_w + pad
    top_margin = pad + title_h + pad
    bottom_margin = pad + xcat_h_max + pad + x_label_h + pad

    plot_x0 = left_margin
    plot_y0 = top_margin
    plot_x1 = w_px - right_margin
    plot_y1 = h_px - bottom_margin
    if plot_x1 <= plot_x0 + 1:
        plot_x1 = plot_x0 + 1
    if plot_y1 <= plot_y0 + 1:
        plot_y1 = plot_y0 + 1

    draw.text(((w_px - title_w) // 2, pad), title, font=font_title, fill="#000000")

    grid_color = "#d0d0d0"
    axis_color = "#000000"

    def y_to_px(v: float) -> int:
        if ymax == ymin:
            return plot_y1
        t01 = (v - ymin) / (ymax - ymin)
        return int(round(plot_y1 - t01 * (plot_y1 - plot_y0)))

    draw.line([plot_x0, plot_y1, plot_x1, plot_y1], fill=axis_color, width=1)
    draw.line([plot_x0, plot_y0, plot_x0, plot_y1], fill=axis_color, width=1)

    t = ymin
    while t <= ymax + 1e-9:
        y = y_to_px(t)
        draw.line([plot_x0, y, plot_x1, y], fill=grid_color, width=1)
        txt = f"{t:.0f}"
        tw, th = text_size(draw, txt, font_tick)
        draw.text((plot_x0 - pad // 2 - tw, y - th // 2), txt, font=font_tick, fill="#000000")
        t += ytick_step

    n_groups = len(categories)
    n_series = len(series)
    group_w = (plot_x1 - plot_x0) / max(1, n_groups)
    inner_pad = group_w * 0.18
    bar_w = (group_w - inner_pad) / max(1, n_series)

    for gi, cat in enumerate(categories):
        gx0 = plot_x0 + gi * group_w
        gx1 = gx0 + group_w
        cx = int(round((gx0 + gx1) / 2))
        tw, th = text_size(draw, cat, font_tick)
        draw.text((cx - tw // 2, plot_y1 + pad // 2), cat, font=font_tick, fill="#000000")

        for si, (name, color, vals) in enumerate(series):
            v = vals[gi]
            if v is None:
                continue
            x0 = int(round(gx0 + inner_pad / 2 + si * bar_w))
            x1 = int(round(x0 + bar_w * 0.92))
            y0 = y_to_px(v)
            draw.rectangle([x0, y0, x1, plot_y1], fill=color, outline=None)

            val_txt = f"{v:.1f}%"
            bx0, by0, bx1, by1 = draw.textbbox((0, 0), val_txt, font=font_value)
            tvw = bx1 - bx0
            tvh = by1 - by0
            tx = int(round((x0 + x1 - tvw) / 2))
            ty = int(round(y0 - tvh - max(2, pad // 4)))
            draw_text_top_left(draw, tx, ty, val_txt, font=font_value, fill="#000000")

        if group_annotations and gi < len(group_annotations):
            ann = group_annotations[gi]
            if ann:
                atw, ath = text_size(draw, ann, font_value)
                draw_text_top_left(draw, cx - atw // 2, plot_y0 + pad // 2, ann, font=font_value, fill=group_annotation_color)

    x_label_x = int(round((plot_x0 + plot_x1 - x_label_w) / 2))
    x_label_y = int(round(plot_y1 + pad // 2 + xcat_h_max + pad // 2))
    draw.text((x_label_x, x_label_y), x_label, font=font_label, fill="#000000")

    img.alpha_composite(rot_y, dest=(pad, int(round((plot_y0 + plot_y1) / 2 - rot_y.size[1] / 2))))

    legend_x1 = w_px - pad
    legend_y1 = plot_y1 - pad // 2
    legend_x0 = legend_x1 - legend_w - pad
    legend_y0 = legend_y1 - legend_h - pad
    draw.rectangle([legend_x0, legend_y0, legend_x1, legend_y1], fill=(255, 255, 255, 235), outline="#dddddd", width=1)

    ly = legend_y0 + pad // 2
    for name, color, _ in series:
        tw, th = text_size(draw, name, font_tick)
        cy = ly + max(th, legend_box) // 2
        bx0 = legend_x0 + pad // 2
        by0 = cy - legend_box // 2
        draw.rectangle([bx0, by0, bx0 + legend_box, by0 + legend_box], fill=color, outline="#000000", width=1)
        draw.text((bx0 + legend_box + pad // 2, cy - th // 2), name, font=font_tick, fill="#000000")
        ly += max(th, legend_box) + pad // 4

    if stamp:
        stamp_txt = datetime.now().strftime("generated %Y-%m-%d %H:%M:%S")
        sw, sh = text_size(draw, stamp_txt, font_tick)
        draw.text((w_px - pad - sw, h_px - pad - sh), stamp_txt, font=font_tick, fill="#666666")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path, format="PNG")


def draw_fig3a_ablation(
    out_path: Path,
    fig_w: float, fig_h: float, dpi: int,
    title_fs: int, tick_fs: int, label_fs: int, value_fs: int,
    xmin: float, xmax: float, tick_step: float, stamp: bool,
):
    labels = [
        "Full 2D-CL+SAR",
        "w/o Router",
        "w/o Staging",
        "w/o Difficulty",
        "w/o CRS",
        "w/o Inheritance",
        "Std. SFT",
    ]
    values = [82.60, 82.12, 78.64, 80.84, 79.48, 77.32, 76.01]

    w_px = max(1, int(round(fig_w * dpi)))
    h_px = max(1, int(round(fig_h * dpi)))

    img = Image.new("RGBA", (w_px, h_px), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_title = try_load_font(fs_pt_to_px(title_fs, dpi), bold=True)
    font_tick = try_load_font(fs_pt_to_px(tick_fs, dpi), bold=False)
    font_label = try_load_font(fs_pt_to_px(label_fs, dpi), bold=False)
    font_value = try_load_font(fs_pt_to_px(value_fs, dpi), bold=True)

    title = "Ablation Study: Component Contributions"
    title_w, title_h = text_size(draw, title, font_title)

    pad = max(10, int(round(min(w_px, h_px) * 0.02)))

    y_label_w_max = max((text_size(draw, t, font_tick)[0] for t in labels), default=0)
    x_label = "ChartQA Accuracy (%)"
    x_label_w, x_label_h = text_size(draw, x_label, font_label)

    tick_vals = []
    t = xmin
    while t <= xmax + 1e-9:
        tick_vals.append(f"{t:.0f}")
        t += tick_step
    tick_h_max = max((text_size(draw, t, font_tick)[1] for t in tick_vals), default=0)
    value_w_max = max((text_size(draw, f"{v:.2f}%", font_value)[0] for v in values), default=0)

    left_margin = pad + y_label_w_max + pad
    right_margin = pad + value_w_max + pad
    top_margin = pad + title_h + pad
    bottom_margin = pad + tick_h_max + pad + x_label_h + pad

    plot_x0 = left_margin
    plot_y0 = top_margin
    plot_x1 = w_px - right_margin
    plot_y1 = h_px - bottom_margin
    if plot_x1 <= plot_x0 + 1:
        plot_x1 = plot_x0 + 1
    if plot_y1 <= plot_y0 + 1:
        plot_y1 = plot_y0 + 1

    draw.text(((w_px - title_w) // 2, pad), title, font=font_title, fill="#000000")

    n = len(labels)
    slot_h = (plot_y1 - plot_y0) / max(1, n)
    bar_h = max(1, int(round(slot_h * 0.55)))

    def x_to_px(v):
        if xmax == xmin:
            return plot_x0
        return int(round(plot_x0 + (v - xmin) / (xmax - xmin) * (plot_x1 - plot_x0)))

    grid_color = "#d0d0d0"
    axis_color = "#000000"

    draw.line([plot_x0, plot_y1, plot_x1, plot_y1], fill=axis_color, width=1)

    t = xmin
    while t <= xmax + 1e-9:
        x = x_to_px(t)
        draw.line([x, plot_y0, x, plot_y1], fill=grid_color, width=1)
        txt = f"{t:.0f}"
        tw, th = text_size(draw, txt, font_tick)
        draw.text((x - tw // 2, plot_y1 + pad // 2), txt, font=font_tick, fill="#000000")
        t += tick_step

    full_val = values[0]
    x_ref = x_to_px(full_val)
    dash = max(4, int(round(h_px * 0.01)))
    gap = max(3, int(round(h_px * 0.008)))
    draw_dashed_vline(draw, x_ref, plot_y0, plot_y1, color="#12b886", width=2, dash=dash, gap=gap)

    colors = ["#12b886", "#1aad6e", "#2b6de8", "#2b6de8", "#2b6de8", "#2b6de8", "#6c757d"]

    for i, (lbl, v, c) in enumerate(zip(labels, values, colors)):
        cy = int(round(plot_y0 + (i + 0.5) * slot_h))
        y0 = cy - bar_h // 2
        y1 = y0 + bar_h
        x0 = x_to_px(xmin)
        x1 = x_to_px(v)

        draw.rectangle([x0, y0, x1, y1], fill=c, outline=None)

        ly = cy - text_size(draw, lbl, font_tick)[1] // 2
        draw.text((plot_x0 - pad - text_size(draw, lbl, font_tick)[0], ly), lbl, font=font_tick, fill="#000000")

        val_txt = f"{v:.2f}%"
        bx0, by0, bx1, by1 = draw.textbbox((0, 0), val_txt, font=font_value)
        tw = bx1 - bx0
        th = by1 - by0
        vx = min(w_px - pad - tw, x1 + int(round(pad * 0.3)))
        draw_text_centered(draw, vx + tw // 2, cy, val_txt, font=font_value, fill="#000000")

    draw.text(((plot_x0 + plot_x1 - x_label_w) // 2, plot_y1 + pad // 2 + tick_h_max + pad // 2), x_label, font=font_label, fill="#000000")

    if stamp:
        stamp_txt = datetime.now().strftime("generated %Y-%m-%d %H:%M:%S")
        sw, sh = text_size(draw, stamp_txt, font_tick)
        draw.text((w_px - pad - sw, h_px - pad - sh), stamp_txt, font=font_tick, fill="#666666")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path, format="PNG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="figs")
    parser.add_argument("--w", type=float, default=12.0)
    parser.add_argument("--h", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title-fs", type=int, default=18)
    parser.add_argument("--tick-fs", type=int, default=13)
    parser.add_argument("--label-fs", type=int, default=14)
    parser.add_argument("--value-fs", type=int, default=13)
    parser.add_argument("--stamp", action="store_true")
    parser.add_argument("--mode", type=str, default="all", choices=["ablation", "cross-dataset", "cross-model", "all"])
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (script_dir / out_dir).resolve()

    if args.mode in ("ablation", "all"):
        ablation_out = out_dir / "fig3a_v2_ablation_study.png"
        draw_fig3a_ablation(
            out_path=ablation_out,
            fig_w=args.w, fig_h=args.h, dpi=args.dpi,
            title_fs=args.title_fs, tick_fs=args.tick_fs,
            label_fs=args.label_fs, value_fs=args.value_fs,
            xmin=74.0, xmax=85.0, tick_step=2.0, stamp=args.stamp,
        )
        print(f"[OK] saved to: {ablation_out}")

    if args.mode in ("cross-dataset", "all"):
        dataset_out = out_dir / "fig3c_v2_cross_dataset.png"
        categories = ["ChartQA", "PlotQA", "FigureQA"]
        sar  = [82.6, 58.20, 98.06]
        cl   = [82.1, 56.5, 92.3]
        sft  = [76.0, 51.9, 89.7]
        draw_grouped_bar_chart(
            out_path=dataset_out,
            fig_w=args.w, fig_h=args.h, dpi=args.dpi,
            title="Cross-Dataset Generalization (Zero-Shot Transfer)",
            y_label="Accuracy (%)",
            x_label="",
            categories=categories,
            series=[
                ("2D-CL+SAR", "#12b886", sar),
                ("2D-CL", "#2b6de8", cl),
                ("Std. SFT", "#6c757d", sft),
            ],
            ymin=45.0, ymax=100.0, ytick_step=10.0,
            title_fs=args.title_fs, tick_fs=args.tick_fs,
            label_fs=args.label_fs, value_fs=args.value_fs,
            stamp=args.stamp,
        )
        print(f"[OK] saved to: {dataset_out}")

    if args.mode in ("cross-model", "all"):
        model_out = out_dir / "fig3b_v2_cross_model.png"
        categories = ["Qwen2.5-VL-7B", "Qwen2.5-VL-3B", "InternVL2-8B"]
        sft = [76.01, 72.41, 77.36]
        sar = [82.60, 77.58, 81.94]
        ann = ["+6.59", "+5.17", "+4.58"]
        draw_grouped_bar_chart(
            out_path=model_out,
            fig_w=args.w, fig_h=args.h, dpi=args.dpi,
            title="Generalization Across Vision-Language Models",
            y_label="ChartQA Accuracy (%)",
            x_label="",
            categories=categories,
            series=[
                ("Std. SFT", "#6c757d", sft),
                ("2D-CL+SAR", "#12b886", sar),
            ],
            ymin=68.0, ymax=88.0, ytick_step=2.5,
            title_fs=args.title_fs, tick_fs=args.tick_fs,
            label_fs=args.label_fs, value_fs=args.value_fs,
            stamp=args.stamp,
            group_annotations=ann,
        )
        print(f"[OK] saved to: {model_out}")


if __name__ == "__main__":
    main()
