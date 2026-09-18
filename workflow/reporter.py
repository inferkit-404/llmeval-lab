"""Report generator for evaluation results."""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from rich.console import Console


console = Console()


@dataclass
class ReportSection:
    """A section of a report."""
    title: str
    content: str
    level: int = 2  # Markdown heading level


class ReportGenerator:
    """Generate evaluation reports in various formats."""

    def __init__(self, output_dir: str = "./reports"):
        """Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(
        self,
        sections: List[ReportSection],
        title: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a Markdown report.

        Args:
            sections: List of report sections
            title: Report title
            output_path: Optional output path

        Returns:
            Markdown content as string
        """
        lines = [
            f"{'#' * 1} {title}",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n---\n",
        ]

        for section in sections:
            lines.append(f"{'#' * section.level} {section.title}")
            lines.append(f"\n{section.content}\n")

        content = "\n".join(lines)

        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    def generate_json_report(
        self,
        data: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a JSON report.

        Args:
            data: Report data
            output_path: Optional output path

        Returns:
            JSON content as string
        """
        data["generated_at"] = datetime.now().isoformat()
        content = json.dumps(data, indent=2, ensure_ascii=False)

        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    def generate_summary_report(
        self,
        eval_results: List[Any],
        model_name: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a summary report from evaluation results.

        Args:
            eval_results: List of EvalResult objects
            model_name: Name of evaluated model
            output_path: Optional output path

        Returns:
            Report content as string
        """
        # Group results by benchmark
        by_benchmark: Dict[str, List[Any]] = {}
        for result in eval_results:
            if result.benchmark_name not in by_benchmark:
                by_benchmark[result.benchmark_name] = []
            by_benchmark[result.benchmark_name].append(result)

        sections = [
            ReportSection(
                title="Overview",
                content=f"**Model**: {model_name}\n\n**Total Benchmarks**: {len(by_benchmark)}\n\n**Total Metrics**: {len(eval_results)}",
            ),
        ]

        # Benchmark details
        benchmark_content = []
        for bench_name, results in by_benchmark.items():
            bench_lines = [f"### {bench_name}"]
            for r in results:
                value_str = f"{r.value:.4f}"
                if r.std:
                    value_str += f" ± {r.std:.4f}"
                bench_lines.append(f"- **{r.metric}**: {value_str} (n={r.num_samples})")
            benchmark_content.append("\n".join(bench_lines))

        sections.append(ReportSection(
            title="Benchmark Results",
            content="\n\n".join(benchmark_content),
            level=2,
        ))

        # Errors if any
        errors = [r for r in eval_results if hasattr(r, 'error') and r.error]
        if errors:
            error_content = "\n".join(f"- {e.error}" for e in errors)
            sections.append(ReportSection(
                title="Errors",
                content=error_content,
                level=2,
            ))

        return self.generate_markdown_report(
            sections,
            f"Evaluation Report: {model_name}",
            output_path,
        )

    def print_report(self, content: str) -> None:
        """Print report to console.

        Args:
            content: Report content
        """
        console.print(content)
