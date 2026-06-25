"""config — 全局配置 ==============================================
加载 .env、定义路径常量、维护产业关键词映射、栏目编码表、
时序数据库路径、量化指标阈值、产业链-A股映射。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================
# 路径常量
# ============================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
DATA_HISTORICAL_DIR: Path = PROJECT_ROOT / "data" / "historical"
REPORT_DIR: Path = PROJECT_ROOT / "reports"
IMAGE_DIR: Path = PROJECT_ROOT / "images"
DASHBOARD_DIR: Path = PROJECT_ROOT / "dashboard"
LOG_DIR: Path = PROJECT_ROOT / "logs"

for _d in (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_HISTORICAL_DIR,
    REPORT_DIR, IMAGE_DIR, DASHBOARD_DIR, LOG_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# 时序数据库 (SQLite) 路径
TIMESERIES_DB_PATH: Path = DATA_HISTORICAL_DIR / "economy_timeseries.sqlite3"


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

AI_REFLECTION_MAX_ROUNDS: int = int(os.getenv("AI_REFLECTION_MAX_ROUNDS", "2"))
AI_REFLECTION_THRESHOLD: float = float(os.getenv("AI_REFLECTION_THRESHOLD", "0.7"))


# ============================================================
# 爬虫配置
# ============================================================
BASE_URL: str = "http://finance.people.com.cn"
LIST_URL_TEMPLATE: str = f"{BASE_URL}/index{{n}}.html"
LIST_PAGES: int = 1
REQUEST_INTERVAL_SEC: float = 2.0
REQUEST_TIMEOUT_SEC: int = 15
REFERER: str = BASE_URL + "/"
CHINESE_FONT_PATH: str = r"C:\Windows\Fonts\msyh.ttc"

# 多源爬虫开关
ENABLE_PEOPLE: bool = os.getenv("ENABLE_PEOPLE", "1") == "1"
ENABLE_CE: bool = os.getenv("ENABLE_CE", "1") == "1"
ENABLE_XINHUA: bool = os.getenv("ENABLE_XINHUA", "0") == "1"
ENABLE_21CBH: bool = os.getenv("ENABLE_21CBH", "0") == "1"
ENABLE_GOV_PBOC: bool = os.getenv("ENABLE_GOV_PBOC", "0") == "1"

# 多 UA 池(轮询降低反爬概率)
USER_AGENT_POOL: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]
USER_AGENT: str = USER_AGENT_POOL[0]


# ============================================================
# 栏目编码 -> 名称 映射
# ============================================================
CHANNEL_CODE_MAP: Dict[str, str] = {
    "1004": "财经要闻", "1005": "宏观", "1006": "金融", "1007": "产业",
    "1008": "公司", "1009": "国际", "1010": "证券", "1011": "银行",
    "1012": "基金", "1013": "股市", "1014": "保险", "1015": "消费",
    "1016": "能源", "1017": "科技", "1018": "汽车", "1019": "房产",
    "1020": "区域",
}


# ============================================================
# 产业关键词词库
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
# 产业链-A股概念板块映射
# ============================================================
INDUSTRY_TO_STOCKS: Dict[str, List[str]] = {
    "新能源": ["宁德时代", "比亚迪", "隆基绿能", "阳光电源", "通威股份", "金风科技"],
    "半导体": ["中芯国际", "北方华创", "韦尔股份", "长电科技", "兆易创新", "中微公司"],
    "人工智能": ["科大讯飞", "海光信息", "寒武纪", "商汤", "云从科技", "拓尔思"],
    "数字经济": ["中国移动", "中国电信", "中国联通", "紫光股份", "中兴通讯", "深桑达"],
    "房地产": ["保利发展", "万科A", "招商蛇口", "金地集团", "龙湖集团", "中海地产"],
    "金融": ["工商银行", "建设银行", "中国平安", "招商银行", "中信证券", "东方财富"],
    "制造业": ["三一重工", "汇川技术", "埃斯顿", "绿的谐波", "徐工机械", "中国一重"],
    "消费": ["贵州茅台", "五粮液", "伊利股份", "海天味业", "美的集团", "格力电器"],
    "外贸": ["中远海控", "中国外运", "中集集团", "招商轮船", "宁波港", "上港集团"],
    "农业": ["北大荒", "隆平高科", "牧原股份", "温氏股份", "新希望", "海大集团"],
    "医疗健康": ["恒瑞医药", "迈瑞医疗", "药明康德", "智飞生物", "片仔癀", "爱尔眼科"],
    "基础设施": ["中国中铁", "中国铁建", "中国交建", "中国建筑", "中国电建", "中国能建"],
}


# ============================================================
# 量化指标阈值
# ============================================================
SENTIMENT_STANCE_SCORE: Dict[str, float] = {"利好": 1.0, "中性": 0.0, "利空": -1.0}
POLICY_STANCE_SCORE: Dict[str, float] = {"扩张": 1.0, "中性": 0.0, "收紧": -1.0}
ATTENTION_ENTROPY_HIGH: float = 0.85
TREND_EMERGING_RATIO: float = 3.0
TREND_DECLINING_RATIO: float = 0.33


# ============================================================
# 实体词典(规则 NER 兜底)
# ============================================================
ENTITY_KEYWORDS: Dict[str, List[str]] = {
    "央行/货币当局": ["央行", "中国人民银行", "外汇局", "银保监会", "证监会"],
    "国务院/发改委": ["国务院", "发改委", "国家发改委", "国办", "国常会"],
    "工信部": ["工信部", "工业和信息化部"],
    "财政部": ["财政部", "国税总局", "税务总局"],
    "地方政府": ["上海市", "北京市", "广东省", "江苏省", "浙江省", "深圳市"],
    "国资央企": ["国资委", "央企", "国企改革"],
    "国际组织": ["IMF", "世界银行", "WTO", "美联储", "欧央行", "G20"],
    "重大政策": ["十四五", "十五五", "新质生产力", "供给侧", "双循环", "共同富裕", "高质量发展"],
}


# ============================================================
# 停用词
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


def load_stopwords() -> set:
    base = set()
    jieba_stop = PROJECT_ROOT / "src" / "nlp" / "stopwords.txt"
    if jieba_stop.exists():
        base.update(w.strip() for w in jieba_stop.read_text(encoding="utf-8").splitlines() if w.strip())
    base.update(EXTRA_STOPWORDS)
    return base
