"""wordcloud_gen — 词云图生成 ==============================================
基于 wordcloud + matplotlib,支持中文(需指定字体路径)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from src.config import CHINESE_FONT_PATH, IMAGE_DIR
from src.utils.logger import get_logger

logger = get_logger("nlp.wordcloud_gen")


def generate_wordcloud(
    word_freq: Iterable[Tuple[str, int]],
    output_name: str = "wordcloud.png",
    width: int = 1200,
    height: int = 600,
    background_color: str = "white",
    max_words: int = 200,
    font_path: str = CHINESE_FONT_PATH,
) -> Path:
    """生成词云图并保存到 images/。

    Args:
        word_freq: [(word, count), ...]
        output_name: 文件名
        width/height: 像素
        background_color: 背景色
        max_words: 最大词数
        font_path: 中文字体路径

    Returns:
        保存的文件路径
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = IMAGE_DIR / output_name

    freq_dict = dict(word_freq)

    # 字体兜底:尝试 msyh.ttc → simhei.ttf → 默认
    fp = font_path
    if fp and not Path(fp).exists():
        for alt in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
            if Path(alt).exists():
                fp = alt
                break
        else:
            logger.warning("未找到中文字体,词云可能显示方块")
            fp = None

    wc = WordCloud(
        font_path=fp,
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        collocations=False,
        prefer_horizontal=0.9,
    )
    wc.generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)

    logger.info(f"词云图已保存: {output_path}")
    return output_path


if __name__ == "__main__":
    sample = [("新能源", 50), ("汽车", 40), ("光伏", 30), ("降准", 25), ("金融", 20)]
    p = generate_wordcloud(sample, output_name="sample.png")
    print(p)