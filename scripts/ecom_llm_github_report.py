#!/usr/bin/env python3
"""
Generate a printable PDF report of GitHub projects related to
e-commerce and LLM applications, sorted by GitHub stars.

This script uses the GitHub REST API and heuristics to infer:
1) Business model
2) Technical architecture (diagram)
3) Algorithm architecture (diagram)
4) Key code logic
5) Commercialization prospects

Usage:
  python3 scripts/ecom_llm_github_report.py \
    --max-repos 50 \
    --output reports/ecommerce_llm_report.pdf
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import textwrap
import time
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager as fm


DEFAULT_QUERY = (
    '(ecommerce OR "e-commerce" OR shopify OR woocommerce OR retail OR shopping) '
    "llm in:readme in:description"
)

GITHUB_API = "https://api.github.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GitHub e-commerce + LLM report generator")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="GitHub search query")
    parser.add_argument("--max-repos", type=int, default=50, help="Max repos to include")
    parser.add_argument("--per-page", type=int, default=100, help="GitHub API page size")
    parser.add_argument("--min-stars", type=int, default=5, help="Minimum stars filter")
    parser.add_argument("--output", default="reports/ecommerce_llm_report.pdf", help="Output PDF path")
    parser.add_argument(
        "--cache-dir", default=".cache/github", help="Cache directory for GitHub API responses"
    )
    parser.add_argument(
        "--include-readme", action="store_true", help="Fetch README for better heuristics"
    )
    parser.add_argument(
        "--include-code", action="store_true", help="Fetch key files to infer code logic"
    )
    parser.add_argument(
        "--font-path",
        default="",
        help="Optional font path for CJK rendering (e.g. NotoSansCJKsc-Regular.otf)",
    )
    parser.add_argument(
        "--download-font",
        action="store_true",
        help="Download Noto Sans CJK SC font if missing",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.6,
        help="Delay between GitHub API calls (seconds)",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sha1_hex(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def http_get(url: str, headers: Dict[str, str], timeout: int = 30) -> Tuple[int, dict, bytes]:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), dict(resp.headers), resp.read()


def github_get(
    url: str,
    token: Optional[str],
    cache_dir: str,
    accept: str = "application/vnd.github+json",
    raw: bool = False,
    sleep_s: float = 0.0,
) -> Tuple[Optional[dict], Optional[bytes], dict]:
    ensure_dir(cache_dir)
    cache_key = sha1_hex(f"{url}|{accept}|{raw}")
    cache_path = os.path.join(cache_dir, f"{cache_key}.bin")
    meta_path = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.exists(cache_path) and os.path.exists(meta_path):
        data = read_bytes(cache_path)
        meta = json.loads(read_bytes(meta_path).decode("utf-8"))
        if raw:
            return None, data, meta
        return json.loads(data.decode("utf-8")), None, meta

    headers = {
        "Accept": accept,
        "User-Agent": "cursor-report-bot",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    status, resp_headers, data = http_get(url, headers=headers)
    meta = {"status": status, "headers": resp_headers, "url": url}
    write_bytes(cache_path, data)
    write_bytes(meta_path, json.dumps(meta, ensure_ascii=True, indent=2).encode("utf-8"))
    if sleep_s > 0:
        time.sleep(sleep_s)

    if raw:
        return None, data, meta
    return json.loads(data.decode("utf-8")), None, meta


def search_repositories(
    query: str,
    token: Optional[str],
    cache_dir: str,
    per_page: int,
    max_repos: int,
    min_stars: int,
    sleep_s: float,
) -> List[dict]:
    repos: List[dict] = []
    page = 1
    while len(repos) < max_repos:
        q = urllib.parse.quote(query)
        url = f"{GITHUB_API}/search/repositories?q={q}&sort=stars&order=desc&per_page={per_page}&page={page}"
        data, _, meta = github_get(
            url, token=token, cache_dir=cache_dir, accept="application/vnd.github+json", sleep_s=sleep_s
        )
        if not data or "items" not in data:
            break
        items = data["items"]
        if not items:
            break
        for item in items:
            if item.get("stargazers_count", 0) >= min_stars:
                repos.append(item)
            if len(repos) >= max_repos:
                break
        if len(items) < per_page:
            break
        page += 1
        if page > 10:
            break
    return repos[:max_repos]


def fetch_readme(
    owner: str,
    repo: str,
    token: Optional[str],
    cache_dir: str,
    sleep_s: float,
) -> str:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/readme"
    _, data, _ = github_get(
        url,
        token=token,
        cache_dir=cache_dir,
        accept="application/vnd.github.raw",
        raw=True,
        sleep_s=sleep_s,
    )
    if data is None:
        return ""
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def fetch_repo_tree(
    owner: str,
    repo: str,
    branch: str,
    token: Optional[str],
    cache_dir: str,
    sleep_s: float,
) -> List[str]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    data, _, meta = github_get(
        url, token=token, cache_dir=cache_dir, accept="application/vnd.github+json", sleep_s=sleep_s
    )
    if not data or data.get("truncated"):
        return []
    paths = [item.get("path") for item in data.get("tree", []) if item.get("type") == "blob"]
    return [p for p in paths if p]


def fetch_file_content(
    owner: str,
    repo: str,
    path: str,
    token: Optional[str],
    cache_dir: str,
    sleep_s: float,
) -> str:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}"
    data, _, _ = github_get(
        url,
        token=token,
        cache_dir=cache_dir,
        accept="application/vnd.github.raw",
        raw=True,
        sleep_s=sleep_s,
    )
    if data is None:
        return ""
    return data.decode("utf-8", errors="ignore")


def detect_keywords(text: str, keywords: Iterable[str]) -> bool:
    lower = text.lower()
    return any(k in lower for k in keywords)


def infer_business_model(text: str) -> str:
    text_lower = text.lower()
    if any(k in text_lower for k in ["customer service", "客服", "chatbot", "support", "assistant"]):
        return "智能客服/导购SaaS"
    if any(k in text_lower for k in ["recommend", "recommender", "recommendation", "推荐"]):
        return "个性化推荐/导购"
    if any(k in text_lower for k in ["search", "semantic", "qa", "问答", "检索"]):
        return "智能搜索/问答"
    if any(k in text_lower for k in ["marketing", "ads", "campaign", "文案", "营销"]):
        return "营销内容生成/投放优化"
    if any(k in text_lower for k in ["pricing", "price", "定价"]):
        return "动态定价/促销优化"
    return "电商运营助手/开发者工具"


def infer_tech_components(text: str) -> List[str]:
    components = ["用户/商家"]

    if detect_keywords(text, ["react", "nextjs", "vue", "svelte", "frontend", "web"]):
        components.append("前端/小程序/插件")
    else:
        components.append("渠道入口/对话界面")

    if detect_keywords(text, ["fastapi", "flask", "django", "express", "nestjs", "spring", "gin", "fiber"]):
        components.append("后端API服务")
    else:
        components.append("应用服务层")

    if detect_keywords(text, ["langchain", "llamaindex", "llama-index", "semantic-kernel", "llm", "gpt"]):
        components.append("LLM应用层")
    else:
        components.append("业务编排层")

    if detect_keywords(
        text, ["faiss", "pinecone", "milvus", "weaviate", "qdrant", "chroma", "vector"]
    ):
        components.append("向量数据库/知识库")

    if detect_keywords(text, ["postgres", "mysql", "mongodb", "redis", "sqlite", "database"]):
        components.append("业务数据库")

    if detect_keywords(text, ["shopify", "woocommerce", "magento", "prestashop", "opencart"]):
        components.append("电商平台/ERP集成")
    else:
        components.append("电商业务系统")

    # De-dup while keeping order
    seen = set()
    ordered = []
    for item in components:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered[:6]


def infer_algorithm_flow(text: str, business_model: str) -> List[str]:
    flow = ["用户问题/指令", "意图识别/任务编排"]

    if detect_keywords(text, ["rag", "retrieval", "vector", "embedding", "knowledge", "检索", "知识库"]):
        flow += ["Embedding生成", "向量检索", "上下文拼接"]

    if business_model == "个性化推荐/导购":
        flow += ["特征处理/召回", "排序/重排"]

    if business_model == "智能搜索/问答":
        flow += ["查询理解", "检索/重排"]

    flow += ["Prompt构建", "LLM推理", "结果校验/响应"]

    # Keep at most 7 steps to avoid layout issues
    if len(flow) > 7:
        flow = flow[:6] + ["结果校验/响应"]
    return flow


def infer_similar_companies(business_model: str) -> List[str]:
    mapping = {
        "智能客服/导购SaaS": ["阿里巴巴", "京东", "拼多多", "抖音电商", "有赞", "微盟"],
        "个性化推荐/导购": ["阿里巴巴", "京东", "拼多多", "抖音电商", "快手电商"],
        "智能搜索/问答": ["阿里巴巴", "京东", "拼多多", "唯品会", "美团电商"],
        "营销内容生成/投放优化": ["有赞", "微盟", "阿里巴巴", "京东", "巨量引擎"],
        "动态定价/促销优化": ["阿里巴巴", "京东", "拼多多", "美团", "苏宁易购"],
        "电商运营助手/开发者工具": ["有赞", "微盟", "阿里云", "京东云", "华为云"],
    }
    return mapping.get(business_model, ["阿里巴巴", "京东", "拼多多"])


def infer_commercialization(business_model: str) -> str:
    if business_model == "智能客服/导购SaaS":
        return "面向商家提供客服自动化与导购能力，可通过SaaS订阅、私有化部署、按对话量计费。"
    if business_model == "个性化推荐/导购":
        return "可嵌入电商平台推荐系统，按GMV增量或调用量计费，亦可与营销系统打包销售。"
    if business_model == "智能搜索/问答":
        return "可用于站内搜索与客服问答优化，提升转化率，适合SaaS或平台插件模式。"
    if business_model == "营销内容生成/投放优化":
        return "可用于营销素材生成与A/B测试，适合SaaS订阅与按内容量计费。"
    if business_model == "动态定价/促销优化":
        return "可为商家提供价格策略与促销建议，适合与ERP/BI系统结合交付。"
    return "可作为电商运营助手与开发者工具，支持SaaS订阅、API计费或私有化部署。"


def pick_key_files(paths: List[str]) -> List[str]:
    candidates = [
        "app.py",
        "main.py",
        "server.py",
        "api.py",
        "wsgi.py",
        "index.js",
        "server.js",
        "app.js",
        "src/index.js",
        "src/main.js",
        "src/server.js",
        "src/app.js",
        "src/main.ts",
        "src/index.ts",
        "src/server.ts",
        "backend/app.py",
        "backend/main.py",
        "api/app.py",
        "api/main.py",
    ]
    path_map = {p.lower(): p for p in paths}
    found = []
    for cand in candidates:
        if cand in path_map:
            found.append(path_map[cand])
        if len(found) >= 3:
            break
    if found:
        return found

    # Fallback: pick up to 2 shallow python/js files
    fallback = []
    for p in paths:
        if p.count("/") <= 2 and re.search(r"\.(py|js|ts)$", p):
            fallback.append(p)
        if len(fallback) >= 2:
            break
    return fallback


def summarize_code_logic(file_path: str, content: str) -> str:
    lower = content.lower()
    notes: List[str] = []
    if "fastapi" in lower:
        notes.append("基于FastAPI暴露API服务")
    if "flask" in lower:
        notes.append("使用Flask构建Web服务")
    if "express" in lower:
        notes.append("使用Express搭建后端")
    if "langchain" in lower:
        notes.append("通过LangChain编排LLM调用")
    if "llamaindex" in lower or "llama_index" in lower:
        notes.append("使用LlamaIndex进行索引与检索")
    if "openai" in lower or "chatgpt" in lower:
        notes.append("对接OpenAI/兼容API进行推理")
    if "embedding" in lower:
        notes.append("生成Embedding向量")
    if "retriever" in lower or "vector" in lower:
        notes.append("向量检索/RAG流程")
    if "shopify" in lower:
        notes.append("对接Shopify平台接口")
    if "woocommerce" in lower:
        notes.append("对接WooCommerce平台接口")
    if "magento" in lower:
        notes.append("对接Magento平台接口")
    if not notes:
        notes.append("入口与路由配置为主，未识别明显框架关键词")
    return f"{file_path}: " + "；".join(dict.fromkeys(notes))


def infer_key_code_logic(
    owner: str,
    repo: str,
    branch: str,
    token: Optional[str],
    cache_dir: str,
    sleep_s: float,
    include_code: bool,
) -> List[str]:
    if not include_code:
        return ["未启用代码抓取，仅基于仓库描述与README推断。"]
    try:
        paths = fetch_repo_tree(owner, repo, branch, token, cache_dir, sleep_s)
    except Exception:
        paths = []
    if not paths:
        return ["未获取到仓库代码结构，可能因仓库过大或权限限制。"]

    key_files = pick_key_files(paths)
    if not key_files:
        return ["未发现明显入口文件，建议查看仓库README或部署脚本。"]

    results = []
    for path in key_files[:2]:
        content = fetch_file_content(owner, repo, path, token, cache_dir, sleep_s)
        if not content:
            results.append(f"{path}: 未能拉取文件内容")
            continue
        results.append(summarize_code_logic(path, content))
    return results


def is_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def wrap_text(text: str, width: int) -> List[str]:
    lines: List[str] = []
    for para in text.splitlines():
        if not para.strip():
            lines.append("")
            continue
        if is_cjk(para):
            line = ""
            for ch in para:
                line += ch
                if len(line) >= width:
                    lines.append(line)
                    line = ""
            if line:
                lines.append(line)
        else:
            lines.extend(textwrap.wrap(para, width=width))
    return lines


def set_cjk_font(font_path: str, download_font: bool) -> None:
    if font_path and os.path.exists(font_path):
        fp = fm.FontProperties(fname=font_path)
        fm.fontManager.addfont(font_path)
        matplotlib.rcParams["font.family"] = fp.get_name()
        return

    candidates = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "WenQuanYi Zen Hei",
    ]
    for name in candidates:
        try:
            fm.findfont(name, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = name
            return
        except Exception:
            continue

    if download_font:
        url = (
            "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/"
            "SimplifiedChinese/NotoSansCJKsc-Regular.otf"
        )
        target_dir = ".cache/fonts"
        ensure_dir(target_dir)
        target_path = os.path.join(target_dir, "NotoSansCJKsc-Regular.otf")
        if not os.path.exists(target_path):
            urllib.request.urlretrieve(url, target_path)
        fp = fm.FontProperties(fname=target_path)
        fm.fontManager.addfont(target_path)
        matplotlib.rcParams["font.family"] = fp.get_name()


def figure_line_height(fig: plt.Figure, fontsize: int, line_spacing: float = 1.2) -> float:
    height_in = fig.get_size_inches()[1]
    return (fontsize / 72.0) / height_in * line_spacing


def draw_paragraph(
    fig: plt.Figure,
    x: float,
    y: float,
    title: str,
    body: str,
    width: int,
    title_size: int = 12,
    body_size: int = 10,
) -> float:
    fig.text(x, y, title, fontsize=title_size, fontweight="bold", va="top")
    y -= figure_line_height(fig, title_size)
    lines = wrap_text(body, width)
    for line in lines:
        fig.text(x, y, line, fontsize=body_size, va="top")
        y -= figure_line_height(fig, body_size)
    return y


def draw_linear_flow(ax, labels: List[str]) -> None:
    ax.set_axis_off()
    n = len(labels)
    if n == 0:
        return
    width = min(0.18, 0.86 / n)
    height = 0.16
    y = 0.5
    xs = [0.07 + i * (0.86 / max(1, n - 1)) for i in range(n)]
    for i, label in enumerate(labels):
        x = xs[i] - width / 2
        rect = FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1,
            edgecolor="#1F4E79",
            facecolor="#E8F1FA",
        )
        ax.add_patch(rect)
        ax.text(xs[i], y, label, ha="center", va="center", fontsize=9)
        if i < n - 1:
            ax.annotate(
                "",
                xy=(xs[i] + width / 2, y),
                xytext=(xs[i + 1] - width / 2, y),
                arrowprops=dict(arrowstyle="->", lw=1),
            )


def render_cover_page(fig: plt.Figure, title: str, subtitle: str, note: str) -> None:
    fig.set_facecolor("white")
    fig.text(0.08, 0.82, title, fontsize=22, fontweight="bold")
    fig.text(0.08, 0.74, subtitle, fontsize=12)
    lines = wrap_text(note, 80)
    y = 0.62
    for line in lines:
        fig.text(0.08, y, line, fontsize=10)
        y -= 0.03


def render_toc_page(fig: plt.Figure, items: List[str], page_start: int) -> None:
    fig.set_facecolor("white")
    fig.text(0.08, 0.94, "目录", fontsize=16, fontweight="bold")
    y = 0.90
    line_h = 0.025
    for idx, line in enumerate(items):
        if y < 0.08:
            break
        fig.text(0.08, y, f"{page_start + idx:03d}. {line}", fontsize=9)
        y -= line_h


def render_project_page_one(
    repo: dict,
    business_model: str,
    tech_components: List[str],
    fig: plt.Figure,
) -> None:
    fig.set_facecolor("white")
    y = 0.95
    title = f"{repo['full_name']}  ★{repo.get('stargazers_count', 0)}"
    fig.text(0.08, y, title, fontsize=16, fontweight="bold", va="top")
    y -= 0.05
    fig.text(0.08, y, repo.get("html_url", ""), fontsize=9, va="top")
    y -= 0.04
    desc = repo.get("description") or "无公开描述"
    y = draw_paragraph(
        fig,
        0.08,
        y,
        "1. 项目名称",
        repo.get("name", ""),
        width=60,
        title_size=12,
        body_size=10,
    )
    y = draw_paragraph(
        fig,
        0.08,
        y - 0.01,
        "2. 项目核心商业模式",
        business_model,
        width=60,
        title_size=12,
        body_size=10,
    )
    y = draw_paragraph(
        fig,
        0.08,
        y - 0.01,
        "项目简介",
        desc,
        width=70,
        title_size=11,
        body_size=9,
    )

    fig.text(0.08, 0.52, "3. 项目的技术架构", fontsize=12, fontweight="bold")
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.38])
    draw_linear_flow(ax, tech_components)


def render_project_page_two(
    algo_flow: List[str],
    key_logic: List[str],
    commercialization: str,
    companies: List[str],
    fig: plt.Figure,
) -> None:
    fig.set_facecolor("white")
    fig.text(0.08, 0.94, "4. 项目的算法架构", fontsize=12, fontweight="bold")
    ax = fig.add_axes([0.08, 0.62, 0.84, 0.25])
    draw_linear_flow(ax, algo_flow)

    y = 0.58
    key_logic_text = "\n".join([f"- {line}" for line in key_logic])
    y = draw_paragraph(
        fig,
        0.08,
        y,
        "5. 关键代码实现逻辑",
        key_logic_text,
        width=75,
        title_size=12,
        body_size=9,
    )
    companies_text = "，".join(companies)
    commercialization_text = f"{commercialization}\n类似商业化公司：{companies_text}"
    draw_paragraph(
        fig,
        0.08,
        y - 0.01,
        "6. 对外商业化前景",
        commercialization_text,
        width=75,
        title_size=12,
        body_size=9,
    )


def generate_report(
    repos: List[dict],
    output_path: str,
    query: str,
    include_readme: bool,
    include_code: bool,
    token: Optional[str],
    cache_dir: str,
    sleep_s: float,
    font_path: str,
    download_font: bool,
) -> None:
    ensure_dir(os.path.dirname(output_path))
    set_cjk_font(font_path, download_font)

    with PdfPages(output_path) as pdf:
        cover = plt.figure(figsize=(8.27, 11.69))
        title = "电商 + AI 大模型应用 GitHub 项目盘点报告"
        subtitle = f"生成时间：{dt.datetime.now().strftime('%Y-%m-%d')}    排序：GitHub Stars 从高到低"
        note = (
            "说明：本报告基于GitHub公开搜索结果与启发式规则生成，"
            "用于快速筛选与对比。GitHub搜索最多返回1000条结果，"
            "若需完整覆盖请调整查询词并分批次生成。"
        )
        render_cover_page(cover, title, subtitle, note)
        pdf.savefig(cover)
        plt.close(cover)

        # Table of contents
        toc_items = [
            f"{repo['full_name']} (★{repo.get('stargazers_count', 0)})"
            for repo in repos
        ]
        page_start = 1
        per_page = 30
        for i in range(0, len(toc_items), per_page):
            fig = plt.figure(figsize=(8.27, 11.69))
            render_toc_page(fig, toc_items[i : i + per_page], page_start + i)
            pdf.savefig(fig)
            plt.close(fig)

        # Project pages
        for repo in repos:
            owner = repo["owner"]["login"]
            name = repo["name"]
            text_source = " ".join(
                [
                    repo.get("description") or "",
                    " ".join(repo.get("topics") or []),
                ]
            )
            readme_text = ""
            if include_readme:
                try:
                    readme_text = fetch_readme(owner, name, token, cache_dir, sleep_s)
                except Exception:
                    readme_text = ""
            text_source += " " + readme_text

            business_model = infer_business_model(text_source)
            tech_components = infer_tech_components(text_source)
            algo_flow = infer_algorithm_flow(text_source, business_model)
            key_logic = infer_key_code_logic(
                owner,
                name,
                repo.get("default_branch") or "main",
                token,
                cache_dir,
                sleep_s,
                include_code,
            )
            commercialization = infer_commercialization(business_model)
            companies = infer_similar_companies(business_model)

            fig1 = plt.figure(figsize=(8.27, 11.69))
            render_project_page_one(repo, business_model, tech_components, fig1)
            pdf.savefig(fig1)
            plt.close(fig1)

            fig2 = plt.figure(figsize=(8.27, 11.69))
            render_project_page_two(algo_flow, key_logic, commercialization, companies, fig2)
            pdf.savefig(fig2)
            plt.close(fig2)


def main() -> None:
    args = parse_args()
    ensure_dir(args.cache_dir)
    token = os.environ.get("GITHUB_TOKEN")

    repos = search_repositories(
        args.query,
        token=token,
        cache_dir=args.cache_dir,
        per_page=args.per_page,
        max_repos=args.max_repos,
        min_stars=args.min_stars,
        sleep_s=args.sleep,
    )

    generate_report(
        repos=repos,
        output_path=args.output,
        query=args.query,
        include_readme=args.include_readme,
        include_code=args.include_code,
        token=token,
        cache_dir=args.cache_dir,
        sleep_s=args.sleep,
        font_path=args.font_path,
        download_font=args.download_font,
    )

    print(f"Saved report to: {args.output}")


if __name__ == "__main__":
    main()
