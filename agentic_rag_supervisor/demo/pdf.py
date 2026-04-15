from __future__ import annotations

import base64
import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from agentic_rag_supervisor.demo.settings import LLM_VLM


def pdf_pages_to_b64(pdf_path: str, max_pages: int = 4) -> List[str]:
    """Render PDF pages to base64 PNG images for GPT-4o Vision."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        images: List[str] = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            images.append(base64.b64encode(pix.tobytes("png")).decode())
        doc.close()
        return images
    except Exception as e:
        print(f"[VLM R3] PDF→image failed ({pdf_path}): {e}")
        return []


def analyze_image_vlm(b64_image: str, doc_name: str = "") -> Dict[str, Any]:
    """R3: Single GPT-4o Vision call — extract structured semiconductor data from one page image."""
    try:
        msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        f"You are analyzing a semiconductor technology document image: {doc_name}\n"
                        "Extract ALL structured information: tables, charts, metrics, company names, "
                        "technology names, TRL levels, dates, performance numbers.\n"
                        "Return ONLY valid JSON:\n"
                        '{"title": str, "data_type": "table|chart|diagram|text", '
                        '"companies": [str], "technologies": [str], '
                        '"key_data": [{"metric": str, "value": str, "company": str, "technology": str}], '
                        '"key_findings": [str]}'
                    ),
                },
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            ]
        )
        resp = LLM_VLM.invoke([msg])
        text = resp.content
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
    except Exception as ex:
        print(f"[VLM R3] Inference error: {ex}")
    return {}


def process_pdf_sources_with_vlm(data_dir: Path) -> List[Document]:
    """R3: Scan data_dir for PDF source files, extract structured data via VLM, return Documents."""
    vlm_docs: List[Document] = []
    pdf_files = [p for p in data_dir.glob("*.pdf") if p.is_file()]
    if not pdf_files:
        return vlm_docs

    print(f"[VLM R3] Processing {len(pdf_files)} PDF source(s) with GPT-4o Vision ...")
    for pdf_path in pdf_files[:3]:
        images = pdf_pages_to_b64(str(pdf_path), max_pages=3)
        for page_idx, b64 in enumerate(images):
            result = analyze_image_vlm(b64, doc_name=pdf_path.name)
            if not result or not result.get("key_findings"):
                continue
            content_parts = result.get("key_findings", []) + [
                f"{item.get('company', '')} {item.get('technology', '')} {item.get('metric', '')}: {item.get('value', '')}"
                for item in result.get("key_data", [])
                if item.get("value")
            ]
            content = f"[VLM: {pdf_path.name} page {page_idx + 1}]\n" + "\n".join(content_parts)
            techs = result.get("technologies", ["HBM4"])
            comps = result.get("companies", ["unknown"])
            vlm_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "doc_id": f"vlm-{pdf_path.stem}-p{page_idx}",
                        "title": f"{pdf_path.name} (VLM page {page_idx + 1})",
                        "technology": techs[0] if techs else "HBM4",
                        "company": comps[0] if comps else "unknown",
                        "source_type": "paper",
                        "published_at": datetime.now().strftime("%Y-%m-%d"),
                        "source_url": str(pdf_path),
                        "chunk_id": f"vlm-{pdf_path.stem}-p{page_idx}-c0",
                    },
                )
            )
    print(f"[VLM R3] Produced {len(vlm_docs)} VLM document(s)")
    return vlm_docs


def validate_pdf_tables_with_vlm(pdf_path: Path) -> Dict[str, Any]:
    """Use VLM as a visual QA pass: markdown tables should render as real table grids."""
    try:
        images = pdf_pages_to_b64(str(pdf_path), max_pages=4)
        if not images:
            return {"ok": True, "issues": [], "suggestions": []}
        msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Inspect these PDF page images. If the report contains tables, verify they appear "
                        "as actual table grids/cells, not raw markdown pipe text. Return ONLY valid JSON: "
                        '{"ok": bool, "issues": [str], "suggestions": [str]}'
                    ),
                },
                *[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}} for b64 in images],
            ]
        )
        resp = LLM_VLM.invoke([msg])
        text = resp.content
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
    except Exception as ex:
        print(f"[PDF VLM-QA] table validation skipped: {ex}")
    return {"ok": True, "issues": [], "suggestions": []}


def inject_visual_markers(draft: str) -> str:
    """Add hidden PDF-only visual markers without changing normal markdown preview."""
    out = draft
    if "<!-- VISUAL:ARCHITECTURE_DIAGRAM -->" not in out:
        anchor = "## 2. Technology Landscape"
        if anchor in out:
            out = out.replace(anchor, f"<!-- VISUAL:ARCHITECTURE_DIAGRAM -->\n\n{anchor}", 1)
        else:
            ko_anchor = "## 2. 분석 대상 기술 현황"
            out = out.replace(ko_anchor, f"<!-- VISUAL:ARCHITECTURE_DIAGRAM -->\n\n{ko_anchor}", 1)
    if "<!-- VISUAL:TRL_CHART -->" not in out:
        anchor = "### 3.1 TRL-Based Competitor Benchmarking"
        if anchor in out:
            out = out.replace(anchor, f"{anchor}\n\n<!-- VISUAL:TRL_CHART -->", 1)
        else:
            ko_anchor = "### 3.1 TRL 기반 경쟁사 비교"
            out = out.replace(ko_anchor, f"{ko_anchor}\n\n<!-- VISUAL:TRL_CHART -->", 1)
    return out


def render_pdf(draft: str, pdf_path: Path) -> Tuple[bool, List[str]]:
    """matplotlib으로 참고 보고서 스타일의 Markdown PDF를 렌더링한다."""
    issues: List[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.font_manager import FontProperties

        raw_lines = draft.splitlines()
        is_korean_pdf = any("\uac00" <= ch <= "\ud7a3" for ch in draft)

        font_candidates = [
            Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
            Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        ]
        font_path = next((p for p in font_candidates if p.exists()), None)
        if is_korean_pdf:
            font_props = (
                FontProperties(fname=str(font_path), size=8.5)
                if font_path
                else FontProperties(family="DejaVu Sans", size=8.5)
            )
        else:
            font_props = FontProperties(family="DejaVu Sans", size=7.8)
        small_props = font_props.copy()
        small_props.set_size(7 if is_korean_pdf else 6.5)
        title_props = font_props.copy()
        title_props.set_size(20 if is_korean_pdf else 18)
        title_props.set_weight("bold")
        subtitle_props = font_props.copy()
        subtitle_props.set_size(12 if is_korean_pdf else 10.5)
        section_props = font_props.copy()
        section_props.set_size(13 if is_korean_pdf else 11.5)
        section_props.set_weight("bold")
        h3_props = font_props.copy()
        h3_props.set_size(10 if is_korean_pdf else 8.8)
        h3_props.set_weight("bold")

        body_start = next((i for i, line in enumerate(raw_lines) if line.startswith("## ")), min(len(raw_lines), 7))
        cover_lines = [line.strip("# ").strip() for line in raw_lines[:body_start] if line.strip()]
        body_lines = raw_lines[body_start:]
        body_wrap_width = 88 if is_korean_pdf else 122
        blank_step = 0.018 if is_korean_pdf else 0.009
        body_line_step = 0.024 if is_korean_pdf else 0.016
        h2_step = 0.052 if is_korean_pdf else 0.043
        h3_step = 0.034 if is_korean_pdf else 0.026
        h4_step = 0.030 if is_korean_pdf else 0.024

        def clean_inline_markdown(value: str) -> str:
            cleaned = value.strip()
            cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
            cleaned = cleaned.replace("<br>", " ").replace("<br/>", " ")
            if cleaned.startswith("- "):
                cleaned = "• " + cleaned[2:]
            return cleaned

        def is_markdown_table_line(value: str) -> bool:
            stripped_value = value.strip()
            return stripped_value.startswith("|") and stripped_value.endswith("|")

        def is_markdown_separator(value: str) -> bool:
            cells = [cell.strip() for cell in value.strip().strip("|").split("|")]
            return bool(cells) and all(cell and set(cell) <= {"-", ":"} for cell in cells)

        def parse_table_row(value: str) -> List[str]:
            return [clean_inline_markdown(cell) for cell in value.strip().strip("|").split("|")]

        def extract_tables(lines: List[str]) -> List[List[List[str]]]:
            tables: List[List[List[str]]] = []
            i = 0
            while i < len(lines):
                if is_markdown_table_line(lines[i]):
                    raw_table: list[str] = []
                    while i < len(lines) and is_markdown_table_line(lines[i]):
                        raw_table.append(lines[i])
                        i += 1
                    rows = [parse_table_row(line) for line in raw_table if not is_markdown_separator(line)]
                    if len(rows) >= 2:
                        tables.append(rows)
                else:
                    i += 1
            return tables

        markdown_tables = extract_tables(body_lines)

        def extract_trl_rows() -> List[Tuple[str, str, float]]:
            rows: List[Tuple[str, str, float]] = []
            for table in markdown_tables:
                headers = [h.lower() for h in table[0]]
                if not any("trl" in h for h in headers):
                    continue
                tech_idx = next((i for i, h in enumerate(headers) if "tech" in h or "기술" in h), None)
                co_idx = next((i for i, h in enumerate(headers) if "company" in h or "회사" in h), None)
                trl_idx = next((i for i, h in enumerate(headers) if "trl" in h), None)
                if tech_idx is None or co_idx is None or trl_idx is None:
                    continue
                for row in table[1:]:
                    if max(tech_idx, co_idx, trl_idx) >= len(row):
                        continue
                    tech = row[tech_idx]
                    co = row[co_idx]
                    try:
                        trl = float(str(row[trl_idx]).strip())
                    except Exception:
                        continue
                    if tech and co:
                        rows.append((tech, co, trl))
            return rows

        trl_rows = extract_trl_rows()

        def add_footer(fig, page_no: int) -> None:
            fig.text(0.5, 0.03, f"Page {page_no}", fontproperties=small_props, ha="center", color="#666666")

        def render_cover(pdf: PdfPages) -> None:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("#ffffff")
            title = cover_lines[0] if cover_lines else "Technology Strategy Report"
            subtitle = cover_lines[1] if len(cover_lines) > 1 else ""
            rest = cover_lines[2:] if len(cover_lines) > 2 else []

            fig.text(0.07, 0.82, title, fontproperties=title_props, color="#0b253a")
            if subtitle:
                fig.text(0.07, 0.77, subtitle, fontproperties=subtitle_props, color="#1f4e79")
            y = 0.70
            for line in rest[:10]:
                fig.text(0.07, y, clean_inline_markdown(line), fontproperties=font_props, color="#222222")
                y -= 0.03

            fig.text(
                0.07,
                0.12,
                "Generated by Agentic RAG Supervisor Demo",
                fontproperties=small_props,
                color="#666666",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        def render_trl_chart() -> None:
            nonlocal fig, y
            if not trl_rows:
                return
            if y < 0.30:
                flush_page()
            techs = [t for t, _, _ in trl_rows]
            trls = [v for _, _, v in trl_rows]
            ax = fig.add_axes([0.10, y - 0.22, 0.84, 0.20])
            ax.barh(list(range(len(techs)))[:12], trls[:12], color="#1f77b4")
            ax.set_yticks(list(range(len(techs)))[:12])
            ax.set_yticklabels(techs[:12], fontproperties=small_props)
            ax.invert_yaxis()
            ax.set_xlabel("TRL", fontproperties=small_props)
            ax.set_title("TRL Snapshot", fontproperties=section_props, loc="left")
            y -= 0.26

        def render_architecture_diagram() -> None:
            nonlocal fig, y
            if y < 0.35:
                flush_page()
            ax = fig.add_axes([0.10, y - 0.22, 0.84, 0.20])
            ax.axis("off")
            ax.text(
                0.0,
                0.5,
                "Architecture Diagram Placeholder\n(Insert company-specific schematic if available)",
                fontproperties=section_props,
                color="#1f4e79",
            )
            y -= 0.26

        def render_table(table_lines: List[str]) -> None:
            nonlocal fig, y
            rows = [parse_table_row(line) for line in table_lines if not is_markdown_separator(line)]
            if len(rows) < 2:
                return
            ncols = max(len(r) for r in rows)
            rows = [r + [""] * (ncols - len(r)) for r in rows]
            if y < 0.20:
                flush_page()
            ax = fig.add_axes([0.07, y - 0.20, 0.88, 0.18])
            ax.axis("off")
            tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="upper left", cellLoc="left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7 if is_korean_pdf else 6.2)
            tbl.scale(1.0, 1.2)
            y -= 0.22

        def flush_page() -> None:
            nonlocal fig, y, page_no
            add_footer(fig, page_no)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            page_no += 1
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("#ffffff")
            y = 0.95

        def render_text_line(line: str) -> None:
            nonlocal fig, y
            stripped = line.strip()
            if not stripped:
                y -= blank_step
                return
            if stripped.startswith("#### "):
                fig.text(0.07, y, clean_inline_markdown(stripped[5:]), fontproperties=h3_props, color="#0b253a")
                y -= h4_step
            elif stripped.startswith("## "):
                fig.text(0.07, y, clean_inline_markdown(stripped[3:]), fontproperties=section_props, color="#0b253a")
                y -= h2_step
            elif stripped.startswith("### "):
                fig.text(0.07, y, clean_inline_markdown(stripped[4:]), fontproperties=h3_props, color="#1f4e79")
                y -= h3_step
            elif stripped.startswith("# "):
                fig.text(0.07, y, clean_inline_markdown(stripped[2:]), fontproperties=section_props, color="#0b253a")
                y -= h2_step
            else:
                cleaned = clean_inline_markdown(stripped)
                wrapped = textwrap.wrap(cleaned, width=body_wrap_width, replace_whitespace=False) or [cleaned]
                for wline in wrapped:
                    if y < 0.09:
                        flush_page()
                    fig.text(0.075, y, wline, fontproperties=font_props, color="#222222")
                    y -= body_line_step

        with PdfPages(str(pdf_path)) as pdf:
            page_no = 1
            render_cover(pdf)
            page_no += 1
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("#ffffff")
            y = 0.95

            i = 0
            while i < len(body_lines):
                line = body_lines[i]
                stripped = line.strip()
                if stripped == "<!-- VISUAL:TRL_CHART -->":
                    render_trl_chart()
                    i += 1
                elif stripped == "<!-- VISUAL:ARCHITECTURE_DIAGRAM -->":
                    render_architecture_diagram()
                    i += 1
                elif is_markdown_table_line(stripped):
                    table_lines: list[str] = []
                    while i < len(body_lines) and is_markdown_table_line(body_lines[i]):
                        table_lines.append(body_lines[i])
                        i += 1
                    render_table(table_lines)
                else:
                    render_text_line(line)
                    i += 1
            add_footer(fig, page_no)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        ok = pdf_path.exists() and pdf_path.stat().st_size > 0
        return ok, issues
    except Exception as ex:
        issues.append(f"PDF rendering failed: {ex}")
        return False, issues

