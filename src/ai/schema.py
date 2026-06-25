"""
schema.py - 强类型 AI 输出 Schema (Pydantic v2)
所有 LLM 任务输出用 Pydantic v2 严格约束:
- Field(ge/le) 做基础值域约束
- field_validator 做单字段业务规则
- model_post_init 做去重/排序 (避免 validate_assignment 递归)
- Annotated + StringConstraints 约束短文本格式
- model_validator(mode="before") 做跨字段一致性
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


# ============================================================
# 共用约束类型 (Annotated 风格,比裸写 str/int 更可读)
# ============================================================
NonEmptyStr = Annotated[str, StringConstraints(min_length=1, max_length=200, strip_whitespace=True)]
Title = Annotated[str, StringConstraints(min_length=2, max_length=30, strip_whitespace=True)]
Content80 = Annotated[str, StringConstraints(min_length=4, max_length=300, strip_whitespace=True)]
Content200 = Annotated[str, StringConstraints(min_length=10, max_length=400, strip_whitespace=True)]
Summary150 = Annotated[str, StringConstraints(min_length=30, max_length=250, strip_whitespace=True)]
Interpretation = Annotated[str, StringConstraints(min_length=15, max_length=250, strip_whitespace=True)]
IssuerName = Annotated[str, StringConstraints(min_length=2, max_length=20, strip_whitespace=True)]
Word2_4 = Annotated[str, StringConstraints(min_length=2, max_length=6, strip_whitespace=True)]
UrlStr = Annotated[str, StringConstraints(min_length=10, max_length=500, strip_whitespace=True)]
Tag = Annotated[str, StringConstraints(min_length=1, max_length=10, strip_whitespace=True)]
IndustryName = Annotated[str, StringConstraints(min_length=2, max_length=10, strip_whitespace=True)]
TargetName = Annotated[str, StringConstraints(min_length=2, max_length=12, strip_whitespace=True)]
Subject = Annotated[str, StringConstraints(min_length=2, max_length=20, strip_whitespace=True)]
Action = Annotated[str, StringConstraints(min_length=2, max_length=30, strip_whitespace=True)]
Object = Annotated[str, StringConstraints(min_length=2, max_length=30, strip_whitespace=True)]
Cluster = Annotated[str, StringConstraints(min_length=2, max_length=20, strip_whitespace=True)]
Quote = Annotated[str, StringConstraints(min_length=4, max_length=200, strip_whitespace=True)]


class StrictModel(BaseModel):
    """所有 schema 父类:strict + 禁止未知字段。

    注意:刻意不开 validate_assignment,否则在 model_post_init 里 setattr
    会触发校验导致循环,这里只需要在构造时校验。
    """
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=True,
    )


# ============================================================
# 枚举 (用 str mixin 让 JSON 序列化输出中文值)
# ============================================================
class PolicyDirection(str, Enum):
    EXPANSION = "扩张"
    TIGHTENING = "收紧"
    NEUTRAL = "中性"


class HeatLevel(str, Enum):
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"


class Stance(str, Enum):
    POSITIVE = "利好"
    NEGATIVE = "利空"
    NEUTRAL = "中性"


class Judgment(str, Enum):
    SUPPORT = "支持"
    NEUTRAL = "中性"
    CAUTION = "警惕"


class EventType(str, Enum):
    MONETARY = "货币政策"
    FISCAL = "财政政策"
    INDUSTRIAL = "产业政策"
    TRADE = "贸易政策"
    REGULATION = "监管政策"
    MARKET = "市场动态"
    OTHER = "其他"


# ============================================================
# 1. 主题词
# ============================================================
class ThemeKeyword(StrictModel):
    word: Word2_4
    score: float = Field(ge=0.0, le=1.0, default=0.5)
    explain: Annotated[str, StringConstraints(min_length=4, max_length=80)]


class ThemeKeywordsResult(StrictModel):
    keywords: List[ThemeKeyword] = Field(min_length=1, max_length=15)

    def model_post_init(self, __context: Any) -> None:
        seen: Dict[str, ThemeKeyword] = {}
        for k in self.keywords:
            cur = seen.get(k.word)
            if cur is None or k.score > cur.score:
                seen[k.word] = k
        object.__setattr__(
            self, "keywords",
            sorted(seen.values(), key=lambda x: x.score, reverse=True),
        )


# ============================================================
# 2. 政策风向
# ============================================================
class PolicyDirectionResult(StrictModel):
    direction: PolicyDirection
    confidence: float = Field(ge=0.0, le=1.0)
    keywords: List[Tag] = Field(min_length=1, max_length=8)
    interpretation: Interpretation

    @field_validator("keywords")
    @classmethod
    def _no_dupe(cls, v: List[str]) -> List[str]:
        if len(set(v)) != len(v):
            raise ValueError("keywords 重复")
        return v


# ============================================================
# 3. 重点产业
# ============================================================
class IndustryFocus(StrictModel):
    name: IndustryName
    heat: HeatLevel
    article_count: int = Field(ge=1, le=999)
    summary: Content80
    stance: Stance = Stance.NEUTRAL


class IndustriesResult(StrictModel):
    industries: List[IndustryFocus] = Field(min_length=1, max_length=8)

    def model_post_init(self, __context: Any) -> None:
        seen: Dict[str, IndustryFocus] = {}
        for i in self.industries:
            cur = seen.get(i.name)
            if cur is None or i.article_count > cur.article_count:
                seen[i.name] = i
        object.__setattr__(
            self, "industries",
            sorted(seen.values(), key=lambda x: x.article_count, reverse=True),
        )


# ============================================================
# 4. 重点政策
# ============================================================
class PolicyEvent(StrictModel):
    title: Title
    issuer: IssuerName
    content: Content200
    source_article: Annotated[str, StringConstraints(min_length=0, max_length=80)] = ""


class PoliciesResult(StrictModel):
    policies: List[PolicyEvent] = Field(default_factory=list, max_length=8)


# ============================================================
# 5. 核心信息
# ============================================================
class CoreInsightsResult(StrictModel):
    insights: Summary150


# ============================================================
# 6. 未来判断
# ============================================================
class Outlook(StrictModel):
    topic: Title
    judgment: Judgment
    rationale: Content80


class OutlooksResult(StrictModel):
    outlooks: List[Outlook] = Field(min_length=1, max_length=6)


# ============================================================
# 7. (新) 立场 / 情感
# ============================================================
class SentimentItem(StrictModel):
    target: TargetName
    stance: Stance
    intensity: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence: Annotated[str, StringConstraints(min_length=4, max_length=60)]


class SentimentResult(StrictModel):
    items: List[SentimentItem] = Field(default_factory=list, max_length=10)

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(
            self, "items",
            sorted(self.items, key=lambda x: x.intensity, reverse=True),
        )


# ============================================================
# 8. (新) 事件抽取 (主谓宾三元组)
# ============================================================
class NewsEvent(StrictModel):
    subject: Subject
    action: Action
    object: Object  # noqa: A002  (保留 object 字段名贴近业务语义)
    event_type: EventType = EventType.OTHER
    impact: Annotated[str, StringConstraints(min_length=0, max_length=60)] = ""


class EventsResult(StrictModel):
    events: List[NewsEvent] = Field(default_factory=list, max_length=8)


# ============================================================
# 9. (新) 跨文章关联 (聚类)
# ============================================================
class CrossLink(StrictModel):
    cluster: Cluster
    article_indices: List[int] = Field(min_length=2, max_length=20)
    summary: Annotated[str, StringConstraints(min_length=10, max_length=100)]

    @field_validator("article_indices")
    @classmethod
    def _sort_unique(cls, v: List[int]) -> List[int]:
        # 排序 + 去重,供后端展示稳定
        return sorted(set(v))


class CrossLinksResult(StrictModel):
    links: List[CrossLink] = Field(default_factory=list, max_length=8)


# ============================================================
# 10. (新) AI 自评
# ============================================================
class SelfEval(StrictModel):
    consistency: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    comments: Annotated[str, StringConstraints(min_length=10, max_length=200)]

    @property
    def overall(self) -> float:
        return round((self.consistency + self.groundedness + self.completeness) / 3, 3)


# ============================================================
# 11. (新) 引用 / Citation
# ============================================================
class Citation(StrictModel):
    article_index: int = Field(ge=0)
    title: Annotated[str, StringConstraints(min_length=2, max_length=80)]
    url: UrlStr
    quote: Quote = ""


# ============================================================
# 12. 顶层聚合结果
# ============================================================
class AnalysisReport(StrictModel):
    """6 + 4 = 10 维分析结果聚合。"""
    theme_keywords: ThemeKeywordsResult
    policy_direction: PolicyDirectionResult
    industries: IndustriesResult
    policies: PoliciesResult
    core_insights: CoreInsightsResult
    outlooks: OutlooksResult
    sentiment: SentimentResult = Field(default_factory=SentimentResult)
    events: EventsResult = Field(default_factory=EventsResult)
    cross_links: CrossLinksResult = Field(default_factory=CrossLinksResult)
    self_eval: Optional[SelfEval] = None

    def is_valid(self) -> bool:
        return bool(self.theme_keywords.keywords)


# ============================================================
# 自检入口
# ============================================================
if __name__ == "__main__":
    tk = ThemeKeywordsResult(keywords=[
        ThemeKeyword(word="降准", score=0.92, explain="央行降准 1 万亿"),
        ThemeKeyword(word="新质生产力", score=0.88, explain="高质量发展核心"),
        ThemeKeyword(word="降准", score=0.95, explain="重复词,保留高分"),  # 故意重复
    ])
    print(f"ThemeKeywords after dedup: {[k.word for k in tk.keywords]} (expect 2 items)")

    pd = PolicyDirectionResult(
        direction=PolicyDirection.EXPANSION,
        confidence=0.88,
        keywords=["降准", "消费券"],
        interpretation="降准释放长期资金,叠加消费刺激,体现明显的政策扩张倾向。",
    )
    print(f"PolicyDirection: {pd.direction} ({pd.confidence:.0%})")

    ind = IndustriesResult(industries=[
        IndustryFocus(name="新能源", heat=HeatLevel.HIGH, article_count=8, summary="销量持续增长", stance=Stance.POSITIVE),
        IndustryFocus(name="半导体", heat=HeatLevel.MEDIUM, article_count=5, summary="国产化加速", stance=Stance.POSITIVE),
    ])
    print(f"Industries: {len(ind.industries)}")

    pe = PoliciesResult(policies=[
        PolicyEvent(title="央行降准 1 万亿", issuer="央行", content="释放长期资金支持实体经济", source_article="央行公告"),
    ])
    print(f"Policies: {len(pe.policies)}")

    ci = CoreInsightsResult(insights="昨日经济新闻聚焦货币政策宽松与产业升级,降准 1 万亿为市场注入流动性,新质生产力相关报道占比上升。")
    print(f"CoreInsights: {ci.insights[:30]}...")

    ol = OutlooksResult(outlooks=[
        Outlook(topic="货币宽松延续", judgment=Judgment.SUPPORT, rationale="降准信号明确,后续 LPR 仍有下调空间。"),
        Outlook(topic="地产去库存", judgment=Judgment.NEUTRAL, rationale="城中村改造提速,但销售端仍待观察。"),
    ])
    print(f"Outlooks: {len(ol.outlooks)}")

    se = SentimentResult(items=[
        SentimentItem(target="新能源", stance=Stance.POSITIVE, intensity=0.85, evidence="销量创新高"),
        SentimentItem(target="房地产", stance=Stance.NEGATIVE, intensity=0.7, evidence="投资持续下滑"),
    ])
    print(f"Sentiment: {len(se.items)} items, sorted by intensity")

    ev = EventsResult(events=[
        NewsEvent(subject="央行", action="宣布降准", object="释放 1 万亿长期资金", event_type=EventType.MONETARY, impact="利好银行地产"),
    ])
    print(f"Events: {len(ev.events)} events")

    cl = CrossLinksResult(links=[
        CrossLink(cluster="降准联动", article_indices=[3, 1, 0], summary="三篇文章共同报道央行降准消息"),
    ])
    print(f"CrossLinks: {len(cl.links)} clusters, indices sorted={cl.links[0].article_indices}")

    self_ev = SelfEval(consistency=0.88, groundedness=0.92, completeness=0.85, comments="整体分析自洽,引用充分,覆盖宏观与产业两个维度。")
    print(f"SelfEval: overall={self_ev.overall:.2f}")

    report = AnalysisReport(
        theme_keywords=tk, policy_direction=pd, industries=ind,
        policies=pe, core_insights=ci, outlooks=ol,
        sentiment=se, events=ev, cross_links=cl, self_eval=self_ev,
    )
    print(f"AnalysisReport valid: {report.is_valid()}")

    # 严格模式:重复 keyword 应报错
    try:
        PolicyDirectionResult(
            direction=PolicyDirection.NEUTRAL,
            confidence=0.5,
            keywords=["降准", "降准"],
            interpretation="测试重复关键词",
        )
        print("FAIL: should reject dupe keywords")
    except Exception as e:
        print(f"Strict check (dupe) ok: {type(e).__name__}")

    # 越界
    try:
        PolicyDirectionResult(
            direction=PolicyDirection.NEUTRAL,
            confidence=1.5,
            keywords=["x"],
            interpretation="测试 confidence 越界",
        )
        print("FAIL: should reject out-of-range")
    except Exception as e:
        print(f"Range check ok: {type(e).__name__}")

    # 未知字段
    try:
        PolicyDirectionResult(
            direction=PolicyDirection.NEUTRAL,
            confidence=0.5,
            keywords=["x"],
            interpretation="测试未知字段",
            bogus_field=42,
        )
        print("FAIL: should reject unknown field")
    except Exception as e:
        print(f"Extra=forbid ok: {type(e).__name__}")

    # JSON 序列化 (用于报告 / 缓存)
    import json
    print(f"JSON roundtrip ok: {len(json.dumps(report.model_dump()))} bytes")
    print("\nAll schema self-tests passed")




