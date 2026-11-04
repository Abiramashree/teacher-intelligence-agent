"""
report.py
PDF report generation for tutoring insights (ReportLab).

Split out of the agent so report formatting can be tested and reused on its own.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

LIST_FIELDS = [
    "topics",
    "misconceptions",
    "knowledge_gaps",
    "questions_to_ask",
    "suggested_activities",
    "next_best_prompts",
    "actionable_feedback",
    "key_quotes",
]


def save_insights_pdf(
    insights: Dict[str, Any],
    retrieved_chunks: Optional[List[str]] = None,
    output_path: str = "tutoring_insights.pdf",
) -> str:
    """Render a TutoringInsights dict to a PDF and return the absolute path."""
    retrieved_chunks = retrieved_chunks or []
    styles = getSampleStyleSheet()

    story: List[Any] = [
        Paragraph("<b>Tutoring Session Insights Report</b>", styles["Title"]),
        Paragraph(datetime.now().strftime("%B %d, %Y %H:%M"), styles["Normal"]),
        Spacer(1, 12),
        Paragraph("<b>Summary</b>", styles["Heading2"]),
        Paragraph(insights.get("summary", "No summary."), styles["BodyText"]),
        Spacer(1, 12),
    ]

    core_data = [
        ["Student Understanding", str(insights.get("student_understanding", ""))],
        ["Difficulty Rating", str(insights.get("difficulty_rating", ""))],
        ["Bloom's Level", str(insights.get("blooms_level", ""))],
        ["Sentiment", str(insights.get("sentiment", ""))],
        ["Confidence", f"{float(insights.get('confidence', 0.0)):.2f}"],
    ]
    table = Table(core_data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )
    story += [Paragraph("<b>Session Metrics</b>", styles["Heading2"]), table, Spacer(1, 12)]

    for field in LIST_FIELDS:
        values = insights.get(field, [])
        if values:
            story.append(Paragraph(f"<b>{field.replace('_', ' ').title()}</b>", styles["Heading3"]))
            for item in values:
                story.append(Paragraph(f"- {item}", styles["BodyText"]))
            story.append(Spacer(1, 6))

    if retrieved_chunks:
        story += [Spacer(1, 12), Paragraph("<b>Retrieved Transcript Chunks</b>", styles["Heading2"])]
        for i, chunk in enumerate(retrieved_chunks[:5], 1):
            snippet = chunk[:500] + ("..." if len(chunk) > 500 else "")
            story += [Paragraph(f"[{i}] {snippet}", styles["Code"]), Spacer(1, 4)]

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(out_path), pagesize=A4).build(story)
    return str(out_path)
