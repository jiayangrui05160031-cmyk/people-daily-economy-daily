"""config — 全局配置 ==============================================
加载 .env、定义路径常量、维护产业关键词映射、栏目编码表。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 容错:无 dotenv 时仍能跑(仅 LLM 调用受限)
    pass


# ============================================================
# 路径常量
# ============================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
REPORT_DIR: Path = PROJECT_ROOT / "reports"
IMAGE_DIR: Path = PROJECT_ROOT / "images"

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, REPORT_DIR, IMAGE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ============================================================
# AI 配置(从 .env 读取)
# ============================================================
AI_PROVIDER: str = os.getenv("AI_PROVIDER", "deepseek")
AI_API_KEY: str = os.getenv("AI_API_KEY", "")
AI_BASE_URL: str = os.getenv("AI_BASE_URL", "https://api.deepseek.com/v1")
AI_MODEL: str = os.getenv("AI_MODEL", "deepseek-chat")
AI_TIMEOUT: int = int(os.getenv("AI_TIMEOUT", "60"))
AI_MAX_RETRIES: int = int(os.getenv("AI_MAX_RETRIES", "3"))
REPORT_DATE_OVERRIDE: str = os.getenv("REPORT_DATE", "").strip()


# ============================================================
# 爬虫配置
# ============================================================
BASE_URL: str = "http://finance.people.com.cn"
LIST_URL_TEMPLATE: str = f"{BASE_URL}/index{{n}}.html"  # index1.html ~ index5.html
LIST_PAGES: int = 1  # 实测:index1-5 内容完全相同,只抓 1 页即可
REQUEST_INTERVAL_SEC: float = 2.0  # 反爬友好间隔
REQUEST_TIMEOUT_SEC: int = 15
USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
REFERER: str = BASE_URL + "/"

# 中文字体(Windows 自带,词云用)
CHINESE_FONT_PATH: str = r"C:\Windows\Fonts\msyh.ttc"


# ============================================================
# 栏目编码 → 名称 映射(从 URL 中 c{数字} 推断)
# ============================================================
CHANNEL_CODE_MAP: Dict[str, str] = {
    "1004": "财经要闻",
    "1005": "宏观",
    "1006": "金融",
    "1007": "产业",
    "1008": "公司",
    "1009": "国际",
    "1010": "证券",
    "1011": "银行",
    "1012": "基金",
    "1013": "股市",
    "1014": "保险",
    "1015": "消费",
    "1016": "能源",
    "1017": "科技",
    "1018": "汽车",
    "1019": "房产",
    "1020": "区域",
}


# ============================================================
# 产业关键词词库(用于产业匹配 + 兜底报告)
# ============================================================
INDUSTRY_KEYWORDS: Dict[str, List[str]] = {
    "新能源": ["光伏", "风电", "储能", "锂电池", "新能源汽车", "氢能", "充电桩", "动力电池"],
    "半导体": ["芯片", "集成电路", "EDA", "光刻机", "晶圆", "封装测试", "国产芯片", "AI芯片"],
    "人工智能": ["大模型", "生成式AI", "算力", "智能驾驶", "具身智能", "人形机器人", "AGI"],
    "数字经济": ["数据要素", "数字人民币", "工业互联网", "平台经济", "数字贸易"],
    "房地产": ["楼市", "房企", "保障房", "城中村改造", "二手房", "保交楼", "房地产融资"],
    "金融": ["央行", "货币政策", "降准", "降息", "人民币汇率", "LPR", "金融监管", "普惠金融"],
    "制造业": ["先进制造", "智能制造", "工业母机", "机器人", "高端装备", "专精特新"],
    "消费": ["扩大内需", "消费券", "消费补贴", "以旧换新", "国货", "消费升级", "下沉市场"],
    "外贸": ["进出口", "外贸新业态", "跨境电商", "一带一路", "自贸区", "RCEP"],
    "农业": ["乡村振兴", "粮食安全", "种业", "高标准农田", "农村电商", "共同富裕"],
    "医疗健康": ["医药集采", "创新药", "医疗器械", "生物医药", "养老", "银发经济"],
    "基础设施": ["新基建", "5G", "特高压", "城际铁路", "数据中心", "算力中心"],
}


# ============================================================
# 停用词(扩展 jieba 默认,补充财经/政府语境虚词)
# ============================================================
EXTRA_STOPWORDS: List[str] = [
    "本报讯", "记者", "编辑", "报道", "近日", "日前", "目前", "今年以来",
    "今年", "去年", "明年", "今后", "日前", "不久前", "近年来", "一直",
    "可以", "可能", "应该", "需要", "进一步", "不断", "持续", "继续",
    "通过", "进行", "推动", "促进", "加强", "提升", "实现", "推进",
    "表示", "指出", "强调", "认为", "建议", "指出", "介绍", "透露",
    "相关", "有关", "一些", "多个", "各种", "全面", "整体", "总体",
    "我国", "国内", "国外", "全球", "全国", "各地", "各部门",
    "方面", "领域", "行业", "地区", "企业", "项目", "工作", "任务",
    "人民网", "人民日报", "来源", "作者", "责任编辑",
]


def load_stopwords() -> set[str]:
    """加载停用词集合(内置 + 扩展)。"""
    base = set()
    jieba_stop = PROJECT_ROOT / "src" / "nlp" / "stopwords.txt"
    if jieba_stop.exists():
        base.update(w.strip() for w in jieba_stop.read_text(encoding="utf-8").splitlines() if w.strip())
    base.update(EXTRA_STOPWORDS)
    return base