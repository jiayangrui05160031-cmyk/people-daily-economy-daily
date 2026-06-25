"""regime.py - v9 前沿: Hamilton (1989) 马尔可夫区制切换模型

给市场/情绪时序打"区制定位": 今天是处于"冷静/低波动" 还是 "狂热/高波动" 区制?
给出每日的"后验区制概率", 让信号引擎可以做"区制条件"的决策。

学术参考:
  - Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle". Econometrica 57(2).
  - Hamilton (1994) "Time Series Analysis" 第 22 章 MS 模型

实现要点 (纯 numpy, 无 statsmodels 依赖):
  - 2 状态 MS-AR(0) 模型 (简化版, 但保留 Hamilton 滤波的精髓)
  - 前向-后向算法 (Kim smoother) 得到平滑后验 P(S_t = k | Y_{1:T})
  - EM 迭代估计转移矩阵 P 和状态条件均值/方差
  - 每个时点输出: 平滑后验 P(冷静), P(狂热), 当前主导区制
  - 收敛判据: 参数变化 < 1e-4 或达到 max_iter

关键设计:
  - 数据用 daily_metric.sentiment_index (0~100), 30 天滚动输入
  - 状态 0 = 冷静 (低均值/低方差), 状态 1 = 狂热 (高均值/高方差)
  - 输出 RegimeReport 含: 平滑概率序列 + 转移矩阵 + 当前定位 + 区制条件指标

依赖: numpy
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("analysis.regime")


# ============================================================
# 数据类
# ============================================================
@dataclass
class RegimePoint:
    date: str
    value: float                # 原始值
    p_calm: float               # P(S=0 | 全部数据) 平滑后验
    p_euphoric: float           # P(S=1 | 全部数据)
    regime: str                 # "calm" / "euphoric"
    regime_id: int              # 0 / 1

    def as_dict(self):
        return asdict(self)


@dataclass
class RegimeReport:
    n_obs: int
    n_iter: int
    converged: bool
    log_likelihood: float
    transition_matrix: List[List[float]]  # 2x2, [[p00, p01], [p10, p11]]
    state_means: List[float]              # [mu_calm, mu_euphoric]
    state_vars: List[float]               # [var_calm, var_euphoric]
    initial_probs: List[float]            # [pi0, pi1]
    current_regime: str                   # "calm" / "euphoric"
    current_p_calm: float
    current_p_euphoric: float
    avg_sojourn_calm: float               # 平均持续天数 (1/p01)
    avg_sojourn_euphoric: float           # 1/p10
    regime_distribution: Dict[str, float] # 历史天数占比
    points: List[RegimePoint] = field(default_factory=list)
    generated_by: str = "hamilton-em"

    def as_dict(self):
        d = asdict(self)
        d["points"] = [p.as_dict() for p in self.points]
        return d


# ============================================================
# Hamilton 滤波 (前向)
# ============================================================
def _forward_filter(
    y: np.ndarray,
    mu: np.ndarray,        # shape (K,)
    sigma2: np.ndarray,    # shape (K,)
    P: np.ndarray,         # shape (K, K)
    pi0: np.ndarray,       # shape (K,)
) -> Tuple[np.ndarray, np.ndarray, float]:
    """前向滤波: 返回 (alpha, xi, loglik).
       alpha[t, k] = P(S_t=k, Y_{1:t}) 未归一化
       xi[t, i, j] = P(S_t=i, S_{t+1}=j, Y_{1:t}) 用于 EM
    """
    T = len(y)
    K = len(mu)
    alpha = np.zeros((T, K))
    xi = np.zeros((T - 1, K, K)) if T > 1 else np.zeros((0, K, K))

    # t=0
    emit = np.exp(-0.5 * (y[0] - mu) ** 2 / sigma2) / np.sqrt(2 * np.pi * sigma2)
    alpha[0] = pi0 * emit
    scale0 = alpha[0].sum()
    if scale0 < 1e-300:
        scale0 = 1e-300
    alpha[0] /= scale0
    loglik = math.log(scale0)

    # t=1..T-1
    for t in range(1, T):
        # 预测: alpha_pred[k] = sum_i alpha[t-1, i] * P[i, k]
        alpha_pred = alpha[t - 1] @ P
        emit = np.exp(-0.5 * (y[t] - mu) ** 2 / sigma2) / np.sqrt(2 * np.pi * sigma2)
        alpha[t] = alpha_pred * emit
        scale = alpha[t].sum()
        if scale < 1e-300:
            scale = 1e-300
        alpha[t] /= scale
        loglik += math.log(scale)

        # xi[t-1, i, j] = alpha[t-1, i] * P[i, j] * emit[j]
        # 注意: 这里 emit[j] 是 t 时刻的, alpha[t-1] 已归一化
        # xi 应满足 sum_{i,j} xi = 1
        if t < T:
            xi_unnorm = alpha[t - 1][:, None] * P * emit[None, :]
            xi_sum = xi_unnorm.sum()
            if xi_sum < 1e-300:
                xi_sum = 1e-300
            xi[t - 1] = xi_unnorm / xi_sum

    return alpha, xi, loglik


# ============================================================
# Kim smoother (后向)
# ============================================================
def _backward_smoother(alpha: np.ndarray, P: np.ndarray) -> np.ndarray:
    """返回 gamma[t, k] = P(S_t=k | Y_{1:T}) 平滑后验。"""
    T, K = alpha.shape
    gamma = np.zeros_like(alpha)
    beta = np.ones((T, K))
    gamma[-1] = alpha[-1]

    for t in range(T - 2, -1, -1):
        # beta[t, i] = sum_j P[i, j] * beta[t+1, j]
        beta[t] = (P @ beta[t + 1])
        gamma[t] = alpha[t] * beta[t]
        s = gamma[t].sum()
        if s < 1e-300:
            s = 1e-300
        gamma[t] /= s

    return gamma


# ============================================================
# EM 算法
# ============================================================
def _em_step(
    y: np.ndarray,
    mu: np.ndarray,
    sigma2: np.ndarray,
    P: np.ndarray,
    pi0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """单步 EM, 返回新参数 + loglik。"""
    alpha, xi, loglik = _forward_filter(y, mu, sigma2, P, pi0)
    gamma = _backward_smoother(alpha, P)
    T, K = gamma.shape

    # 更新 mu, sigma2
    new_mu = np.zeros(K)
    new_sigma2 = np.zeros(K)
    for k in range(K):
        w = gamma[:, k]
        s = w.sum()
        if s < 1e-12:
            new_mu[k] = mu[k]
            new_sigma2[k] = sigma2[k]
            continue
        new_mu[k] = (w * y).sum() / s
        new_sigma2[k] = ((w * (y - new_mu[k]) ** 2).sum() / s) + 1e-6

    # 更新 pi0
    new_pi0 = gamma[0] / max(gamma[0].sum(), 1e-12)

    # 更新 P: P[i, j] = sum_t xi[t, i, j] / sum_t gamma[t, i]
    new_P = np.zeros_like(P)
    if T > 1:
        for i in range(K):
            denom = gamma[:-1, i].sum()
            if denom < 1e-12:
                new_P[i] = P[i]
                continue
            for j in range(K):
                new_P[i, j] = xi[:, i, j].sum() / denom
    else:
        new_P = P

    return new_mu, new_sigma2, new_P, new_pi0, loglik


def _ensure_calm_low(mu: np.ndarray, sigma2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """保证 state 0 是"冷静" (低均值或低方差)。"""
    # 优先级: 1) 方差小的为冷静, 2) 均值小的为冷静
    if sigma2[0] > sigma2[1]:
        return mu[::-1].copy(), sigma2[::-1].copy()
    if abs(sigma2[0] - sigma2[1]) < 0.01 and mu[0] > mu[1]:
        return mu[::-1].copy(), sigma2[::-1].copy()
    return mu.copy(), sigma2.copy()


def _ensure_P_order(P: np.ndarray) -> np.ndarray:
    """配套上面 _ensure_calm_low: 如果已交换 mu, P 也要对应行/列交换。"""
    # 通过 P 的非对角比例判断当前状态 0 是否是"短平均持续"
    # 如果 P[0,1] > P[1,0] 表明状态 0 更易跳走 -> 状态 0 应该是冷静 (短周期)
    # 这里简化: 调用方已决定顺序, 我们不动 P
    return P


def fit_hamilton(
    values: List[float],
    max_iter: int = 50,
    tol: float = 1e-4,
    kmeans_init: bool = True,
) -> RegimeReport:
    """对外主入口: 对一维时序拟合 2 状态 MS 模型。

    Args:
        values: 时序值列表 (建议 14+ 天, 30 天最佳)
        max_iter: 最大迭代次数
        tol: 参数变化收敛阈值
        kmeans_init: 是否用 2-means 初始化 (vs 随机)

    Returns:
        RegimeReport
    """
    y = np.asarray(values, dtype=np.float64)
    if len(y) < 6:
        return RegimeReport(
            n_obs=len(y), n_iter=0, converged=False, log_likelihood=0.0,
            transition_matrix=[[0.9, 0.1], [0.1, 0.9]],
            state_means=[float(y.mean()) if len(y) else 0, float(y.mean()) if len(y) else 0],
            state_vars=[float(y.var() + 1e-6) if len(y) else 1.0,
                        float(y.var() + 1e-6) if len(y) else 1.0],
            initial_probs=[0.5, 0.5],
            current_regime="unknown",
            current_p_calm=0.5, current_p_euphoric=0.5,
            avg_sojourn_calm=10.0, avg_sojourn_euphoric=10.0,
            regime_distribution={"calm": 0.5, "euphoric": 0.5},
            generated_by="hamilton-em-insufficient-data",
        )

    K = 2
    y_mean = float(y.mean())
    y_var = float(y.var() + 1e-6)

    # 初始化: 2-means on values
    if kmeans_init:
        sorted_y = np.sort(y)
        # 简单切分: 中位数分成两组
        med = np.median(y)
        low_mask = y <= med
        if low_mask.sum() < 2 or (~low_mask).sum() < 2:
            # 退到等宽分位
            q = np.quantile(y, 0.5)
            low_mask = y <= q
        mu = np.array([float(y[low_mask].mean()) if low_mask.any() else y_mean - 1,
                       float(y[~low_mask].mean()) if (~low_mask).any() else y_mean + 1])
        sigma2 = np.array([float(y[low_mask].var() + 1e-6) if low_mask.any() else y_var,
                           float(y[~low_mask].var() + 1e-6) if (~low_mask).any() else y_var])
    else:
        mu = np.array([y_mean - 1, y_mean + 1])
        sigma2 = np.array([y_var, y_var])

    # 转移矩阵初始化 (略倾向于持续)
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    pi0 = np.array([0.5, 0.5])

    # 强制 state 0 = 冷静 (低均值)
    mu, sigma2 = _ensure_calm_low(mu, sigma2)

    prev_params = np.concatenate([mu, sigma2, P.flatten(), pi0])
    converged = False
    n_iter = 0
    last_loglik = -np.inf

    for it in range(1, max_iter + 1):
        n_iter = it
        new_mu, new_sigma2, new_P, new_pi0, loglik = _em_step(y, mu, sigma2, P, pi0)
        # 强制排序
        new_mu, new_sigma2 = _ensure_calm_low(new_mu, new_sigma2)
        # 如果状态 0/1 交换了, P 也交换
        # 检测: 新 mu 是否被交换 (mu[0] > mu[1] 表明旧逻辑把它们当成反向)
        if new_mu[0] > new_mu[1]:
            # 反转 P 行/列
            new_P = new_P[::-1, ::-1].copy()

        new_params = np.concatenate([new_mu, new_sigma2, new_P.flatten(), new_pi0])
        delta = float(np.abs(new_params - prev_params).max())

        mu, sigma2, P, pi0 = new_mu, new_sigma2, new_P, new_pi0
        prev_params = new_params

        if abs(loglik - last_loglik) < tol and delta < tol:
            converged = True
            break
        last_loglik = loglik

    # 最终一次前向 + 后向得到平滑后验
    alpha, _, _ = _forward_filter(y, mu, sigma2, P, pi0)
    gamma = _backward_smoother(alpha, P)

    # 转 RegimePoint (无 date, 调用方补)
    points: List[RegimePoint] = []
    for t in range(len(y)):
        g0 = float(gamma[t, 0])
        g1 = float(gamma[t, 1])
        rid = 0 if g0 >= g1 else 1
        points.append(RegimePoint(
            date="", value=float(y[t]),
            p_calm=round(g0, 4), p_euphoric=round(g1, 4),
            regime="calm" if rid == 0 else "euphoric",
            regime_id=rid,
        ))

    # 当前定位
    last = points[-1] if points else None
    cur_regime = last.regime if last else "unknown"
    cur_p_calm = last.p_calm if last else 0.5
    cur_p_euphoric = last.p_euphoric if last else 0.5

    # 平均持续天数 (sojourn time)
    # E[duration in state k] = 1 / (1 - P[k, k])  (期望几何分布)
    p00 = float(P[0, 0]); p01 = float(P[0, 1])
    p10 = float(P[1, 0]); p11 = float(P[1, 1])
    avg_soj_calm = (1.0 / max(1 - p00, 1e-3)) if p01 > 1e-6 else 1e6
    avg_soj_euphoric = (1.0 / max(1 - p11, 1e-3)) if p10 > 1e-6 else 1e6

    # 历史天数占比
    calm_count = sum(1 for p in points if p.regime_id == 0)
    euph_count = len(points) - calm_count
    total = max(len(points), 1)
    distribution = {
        "calm": round(calm_count / total, 4),
        "euphoric": round(euph_count / total, 4),
    }

    return RegimeReport(
        n_obs=len(y), n_iter=n_iter, converged=converged,
        log_likelihood=round(last_loglik, 4),
        transition_matrix=[[round(p00, 4), round(p01, 4)],
                           [round(p10, 4), round(p11, 4)]],
        state_means=[round(float(mu[0]), 4), round(float(mu[1]), 4)],
        state_vars=[round(float(sigma2[0]), 4), round(float(sigma2[1]), 4)],
        initial_probs=[round(float(pi0[0]), 4), round(float(pi0[1]), 4)],
        current_regime=cur_regime,
        current_p_calm=round(cur_p_calm, 4),
        current_p_euphoric=round(cur_p_euphoric, 4),
        avg_sojourn_calm=round(avg_soj_calm, 2),
        avg_sojourn_euphoric=round(avg_soj_euphoric, 2),
        regime_distribution=distribution,
        points=points,
        generated_by="hamilton-em",
    )


def fit_from_daily_metric(target_date: str, lookback: int = 30,
                          metric_col: str = "sentiment_index") -> Optional[RegimeReport]:
    """从 daily_metric 读时序, 拟合 Hamilton 模型。

    Args:
        target_date: 截止日期 (含)
        lookback: 回看天数
        metric_col: 哪一列 ('sentiment_index' | 'policy_stance_score' | ...)
    """
    try:
        from datetime import timedelta
        from src.storage import db as db_mod
        from src.utils.date_utils import parse_date

        end = parse_date(target_date)
        start = end - timedelta(days=lookback)
        with db_mod.get_conn() as conn:
            rows = conn.execute(
                f"SELECT date, {metric_col} FROM daily_metric "
                f"WHERE date BETWEEN ? AND ? ORDER BY date",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        if not rows:
            return None
        values = [float(r[metric_col]) for r in rows if r[metric_col] is not None]
        if len(values) < 6:
            return None

        report = fit_hamilton(values)
        # 补 date
        for i, r in enumerate(rows):
            if i < len(report.points):
                report.points[i].date = r["date"]
        return report
    except Exception as e:
        logger.warning(f"regime 拟合失败: {e}")
        return None


def regime_to_signal(report: RegimeReport) -> Dict[str, Any]:
    """把区制报告转成"信号", 给信号引擎消费。
    - 当前冷静 + 高 p_calm: 利好布局 (+0.3 ~ +0.5)
    - 当前狂热 + 高 p_euphoric: 警示减仓 (-0.3 ~ -0.5)
    - 过渡期 (0.4~0.6): 中性
    """
    if report.n_obs < 6:
        return {"name": "regime", "score": 0.0, "weight": 0.0,
                "reason": "insufficient data", "direction": "neutral"}
    p_calm = report.current_p_calm
    if p_calm >= 0.7:
        score = +0.4
        reason = ("区制: 冷静(" + str(round(p_calm, 2)) +
                  "), 适合布局, 均值 " + str(round(report.state_means[0], 2)))
    elif p_calm >= 0.55:
        score = +0.15
        reason = "区制: 偏冷静, 中性偏多"
    elif p_calm <= 0.3:
        score = -0.4
        reason = ("区制: 狂热(" + str(round(report.current_p_euphoric, 2)) +
                  "), 警惕回调, 均值 " + str(round(report.state_means[1], 2)))
    elif p_calm <= 0.45:
        score = -0.15
        reason = "区制: 偏狂热, 中性偏空"
    else:
        score = 0.0
        reason = "区制: 过渡期, 观望"

    # 持续性高的区制更有信号意义
    weight = 0.15 if max(report.avg_sojourn_calm, report.avg_sojourn_euphoric) > 5 else 0.08

    direction = "bullish" if score > 0.05 else ("bearish" if score < -0.05 else "neutral")
    return {
        "name": "regime",
        "score": round(score, 4),
        "weight": round(weight, 4),
        "reason": reason,
        "direction": direction,
        "current_regime": report.current_regime,
        "p_calm": report.current_p_calm,
        "p_euphoric": report.current_p_euphoric,
        "converged": report.converged,
    }


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    print("== regime self-test ==")

    # 1. 合成测试数据: 30 天, 前 15 天低波动, 后 15 天高波动 (典型"冷静->狂热"过渡)
    np.random.seed(42)
    y_synth = np.concatenate([
        np.random.normal(50, 3, 15),   # 冷静
        np.random.normal(75, 8, 15),   # 狂热
    ])
    rep = fit_hamilton(y_synth.tolist(), max_iter=30)
    print(f"n_iter={rep.n_iter}, converged={rep.converged}, loglik={rep.log_likelihood:.2f}")
    print(f"means: calm={rep.state_means[0]:.2f}, euphoric={rep.state_means[1]:.2f}")
    print(f"vars:  calm={rep.state_vars[0]:.2f}, euphoric={rep.state_vars[1]:.2f}")
    print(f"P: [[{rep.transition_matrix[0][0]}, {rep.transition_matrix[0][1]}],"
          f" [{rep.transition_matrix[1][0]}, {rep.transition_matrix[1][1]}]]")
    print(f"avg_sojourn: calm={rep.avg_sojourn_calm}d, euphoric={rep.avg_sojourn_euphoric}d")
    print(f"distribution: {rep.regime_distribution}")
    print(f"current: {rep.current_regime}, p_calm={rep.current_p_calm}")

    # 2. 信号转换
    sig = regime_to_signal(rep)
    print(f"signal: {sig}")

    # 3. 后验合理: 后期应为"狂热"
    last5_calm_prob = [p.p_calm for p in rep.points[-5:]]
    print(f"last 5 days p_calm: {[round(x, 2) for x in last5_calm_prob]}")
    assert all(p < 0.3 for p in last5_calm_prob), "后期应判定为狂热"

    # 4. 前期应为"冷静"
    first5_calm_prob = [p.p_calm for p in rep.points[:5]]
    print(f"first 5 days p_calm: {[round(x, 2) for x in first5_calm_prob]}")
    assert all(p > 0.7 for p in first5_calm_prob), "前期应判定为冷静"

    # 5. 短数据 graceful
    rep_short = fit_hamilton([50.0, 51.0, 49.0], max_iter=5)
    assert rep_short.n_obs == 3
    print("short data graceful OK")

    print("All regime self-tests passed")