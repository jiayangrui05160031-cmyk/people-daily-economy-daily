"""kg.entities - 实体识别 (规则 + LLM 增强)"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from src.config import ENTITY_KEYWORDS
from src.utils.logger import get_logger

logger = get_logger("kg.entities")


def extract_rule_based(text):
    if not text:
        return []
    out = []
    for ent_type, kws in ENTITY_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                out.append((ent_type, kw))
    return out


def extract_with_llm(router, articles, top_k=8):
    if not router or not articles:
        return []
    from src.ai.prompts import format_articles_for_llm
    text = format_articles_for_llm(articles, max_chars=4000, per_article_limit=200)
    prompt = (
        "请从以下经济新闻中抽取最关键的 8 个实体。"
        "实体可以是机构(央行/工信部/某公司)、政策(降准/以旧换新)、地名(上海/广东)。\n"
        "返回 JSON 数组,每个元素包含 type(分类) 和 value(实体值)。\n"
        "示例:[{\"type\":\"机构\",\"value\":\"央行\"},{\"type\":\"政策\",\"value\":\"降准\"}]\n\n"
        f"新闻:\n{text}\n\n只返回 JSON。"
    )
    try:
        raw, _, _, _ = router.chat(
            messages=[{"role": "system", "content": "你是金融实体抽取专家,只返回 JSON。"},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
            use_json_mode=True,
        )
        from src.ai.router import extract_json
        parsed = extract_json(raw or "")
        items = parsed.get("entities") or parsed.get("items") or []
        result = []
        for it in items[:top_k]:
            t = str(it.get("type", "其他")).strip()
            v = str(it.get("value", "")).strip()
            if v:
                result.append((t, v))
        return result
    except Exception as e:
        logger.warning(f"LLM entity extraction failed: {e}")
        return []


def extract(articles, router=None):
    counter = Counter()
    for art in articles:
        text = (art.content_text or "") + (art.title or "")
        for t, v in extract_rule_based(text):
            counter[(t, v)] += 1

    if router:
        for t, v in extract_with_llm(router, articles):
            counter[(t, v)] += 1

    return [(t, v, c) for (t, v), c in counter.most_common(50)]


if __name__ == "__main__":
    sample_text = "央行宣布降准1万亿,工信部发布新型储能行动方案,上海市发布扩大消费政策。"
    print("rule:", extract_rule_based(sample_text))

    from src.scraper.pipeline import Article
    sample_arts = [
        Article(title="央行降准支持实体经济",
                content=["央行决定下调存款准备金率支持实体经济"],
                content_text="央行决定下调存款准备金率支持实体经济,工信部发布新型储能行动方案"),
        Article(title="上海市扩大消费",
                content=["上海市发布扩大消费若干措施"],
                content_text="上海市发布扩大消费若干措施,涉及新能源汽车以旧换新补贴"),
    ]
    out = extract(sample_arts)
    print("combined:", out)
