"""Result analysis and reporting module."""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np


@dataclass
class AnalysisReport:
    """Comprehensive analysis report."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_name: str = ""
    overall_score: float = 0.0
    benchmark_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    radar_data: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "overall_score": self.overall_score,
            "benchmark_scores": self.benchmark_scores,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "radar_data": self.radar_data,
        }


class ResultAnalyzer:
    """Analyze evaluation results and generate reports."""

    # Benchmark categories and their typical scores
    BENCHMARK_CATEGORIES = {
        "reasoning": ["gsm8k", "math", "bbh"],
        "knowledge": ["mmlu", "triviaqa", "natural_questions"],
        "commonsense": ["hellaswag", "piqa", "arc"],
        "truthfulness": ["truthfulqa"],
        "code": ["humaneval", "mbpp"],
        "multimodal": ["mmmu", "seedbench", "scienceqa"],
    }

    def __init__(self, output_dir: str = "./eval_results"):
        """Initialize result analyzer.

        Args:
            output_dir: Directory to save analysis reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        results: List[Any],
        model_name: str,
    ) -> AnalysisReport:
        """Analyze evaluation results.

        Args:
            results: List of EvalResult objects
            model_name: Name of the evaluated model

        Returns:
            AnalysisReport with insights
        """
        # Group results by benchmark
        benchmark_scores: Dict[str, Dict[str, float]] = {}
        for result in results:
            if result.benchmark_name not in benchmark_scores:
                benchmark_scores[result.benchmark_name] = {}
            benchmark_scores[result.benchmark_name][result.metric] = result.value

        # Calculate overall score (weighted average)
        overall_score = self._calculate_overall_score(benchmark_scores)

        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(benchmark_scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(benchmark_scores, weaknesses)

        # Calculate radar data
        radar_data = self._calculate_radar_data(benchmark_scores)

        report = AnalysisReport(
            model_name=model_name,
            overall_score=overall_score,
            benchmark_scores=benchmark_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            radar_data=radar_data,
        )

        return report

    def _calculate_overall_score(
        self,
        benchmark_scores: Dict[str, Dict[str, float]],
    ) -> float:
        """Calculate overall score from benchmark results."""
        all_scores = []
        for bench_name, metrics in benchmark_scores.items():
            for metric, value in metrics.items():
                if metric == "accuracy":
                    all_scores.append(value)

        if not all_scores:
            return 0.0

        return np.mean(all_scores) * 100

    def _identify_strengths_weaknesses(
        self,
        benchmark_scores: Dict[str, Dict[str, float]],
    ) -> Tuple[List[str], List[str]]:
        """Identify model strengths and weaknesses."""
        strengths = []
        weaknesses = []

        # Get all accuracy scores
        scores_with_names = []
        for bench_name, metrics in benchmark_scores.items():
            if "accuracy" in metrics:
                scores_with_names.append((bench_name, metrics["accuracy"]))

        if not scores_with_names:
            return strengths, weaknesses

        # Sort by score
        scores_with_names.sort(key=lambda x: x[1], reverse=True)

        # Top 3 are strengths (if score > 0.6)
        for bench_name, score in scores_with_names[:3]:
            if score > 0.6:
                strengths.append(f"{bench_name}: {score:.2%}")

        # Bottom 3 are weaknesses (if score < 0.5)
        for bench_name, score in scores_with_names[-3:]:
            if score < 0.5:
                weaknesses.append(f"{bench_name}: {score:.2%}")

        return strengths, weaknesses

    def _generate_recommendations(
        self,
        benchmark_scores: Dict[str, Dict[str, float]],
        weaknesses: List[str],
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        for weakness in weaknesses:
            bench_name = weakness.split(":")[0].strip()

            if bench_name in self.BENCHMARK_CATEGORIES["reasoning"]:
                recommendations.append(
                    "Consider additional chain-of-thought training to improve reasoning能力"
                )
            elif bench_name in self.BENCHMARK_CATEGORIES["knowledge"]:
                recommendations.append(
                    "Consider knowledge distillation or retrieval augmentation to improve knowledge tasks"
                )
            elif bench_name in self.BENCHMARK_CATEGORIES["multimodal"]:
                recommendations.append(
                    "Consider vision-language joint training to improve multimodal understanding"
                )
            elif bench_name == "truthfulqa":
                recommendations.append(
                    "Consider RLHF or constitutional AI training to improve truthfulness"
                )

        if not recommendations:
            recommendations.append("Model performance is balanced across categories")

        return recommendations

    def _calculate_radar_data(
        self,
        benchmark_scores: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Calculate radar chart data for capability visualization."""
        radar_data = {}

        for category, benchmarks in self.BENCHMARK_CATEGORIES.items():
            category_scores = []
            for bench in benchmarks:
                if bench in benchmark_scores and "accuracy" in benchmark_scores[bench]:
                    category_scores.append(benchmark_scores[bench]["accuracy"])

            if category_scores:
                radar_data[category] = np.mean(category_scores)
            else:
                radar_data[category] = 0.0

        return radar_data

    def generate_html_report(
        self,
        report: AnalysisReport,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate HTML visualization report.

        Args:
            report: AnalysisReport to visualize
            output_path: Optional path to save HTML

        Returns:
            HTML content as string
        """
        radar_json = json.dumps(report.radar_data)
        benchmark_json = json.dumps(report.benchmark_scores)

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLMEval-Lab Report: {report.model_name}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .score-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .overall-score {{
            font-size: 4em;
            font-weight: bold;
            color: #667eea;
            text-align: center;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h3 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .strength {{
            color: #22c55e;
        }}
        .weakness {{
            color: #ef4444;
        }}
        .recommendation {{
            color: #3b82f6;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin: 10px 0;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LLMEval-Lab Analysis Report</h1>
        <p>Model: <strong>{report.model_name}</strong></p>
        <p class="timestamp">Generated: {report.timestamp}</p>
    </div>

    <div class="score-card">
        <h2 style="text-align: center; margin-top: 0;">Overall Score</h2>
        <div class="overall-score">{report.overall_score:.1f}%</div>
    </div>

    <div class="grid">
        <div class="card">
            <h3 class="strength">✅ Strengths</h3>
            <ul>
                {"".join(f"<li class='strength'>{s}</li>" for s in report.strengths)}
            </ul>
        </div>
        <div class="card">
            <h3 class="weakness">⚠️ Weaknesses</h3>
            <ul>
                {"".join(f"<li class='weakness'>{w}</li>" for w in report.weaknesses)}
            </ul>
        </div>
    </div>

    <div class="card" style="margin-top: 20px;">
        <h3 class="recommendation">💡 Recommendations</h3>
        <ul>
            {"".join(f"<li class='recommendation'>{r}</li>" for r in report.recommendations)}
        </ul>
    </div>

    <div class="card" style="margin-top: 20px;">
        <h3>Radar Chart</h3>
        <div id="radar-chart"></div>
    </div>

    <div class="card" style="margin-top: 20px;">
        <h3>Benchmark Details</h3>
        <div id="benchmark-chart"></div>
    </div>

    <script>
        // Radar Chart
        const radarData = {radar_json};
        const categories = Object.keys(radarData);
        const values = categories.map(k => radarData[k] * 100);
        values.push(values[0]); // Close the polygon
        const theta = [...categories, categories[0]];

        Plotly.newPlot('radar-chart', [{{
            type: 'scatterpolar',
            r: values,
            theta: theta,
            fill: 'toself',
            fillcolor: 'rgba(102, 126, 234, 0.3)',
            line: {{
                color: '#667eea',
                width: 2
            }}
        }}], {{
            polar: {{
                radialaxis: {{
                    visible: true,
                    range: [0, 100]
                }}
            }},
            showlegend: false
        }});

        // Benchmark Bar Chart
        const benchmarks = {benchmark_json};
        const benchNames = Object.keys(benchmarks);
        const benchValues = benchNames.map(name => benchmarks[name].accuracy * 100 || 0);

        Plotly.newPlot('benchmark-chart', [{{
            type: 'bar',
            x: benchNames,
            y: benchValues,
            marker: {{
                color: benchValues.map(v => v >= 60 ? '#22c55e' : v >= 40 ? '#f59e0b' : '#ef4444')
            }}
        }}], {{
            xaxis: {{ title: 'Benchmark' }},
            yaxis: {{ title: 'Accuracy (%)', range: [0, 100] }},
            margin: {{ t: 20 }}
        }});
    </script>
</body>
</html>
        """

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        return html_content

    def save_report(
        self,
        report: AnalysisReport,
        format: str = "json",
    ) -> Path:
        """Save report to file.

        Args:
            report: AnalysisReport to save
            format: Output format (json, html)

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = report.model_name.replace("/", "_")

        if format == "html":
            output_path = self.output_dir / f"{model_safe}_{timestamp}.html"
            self.generate_html_report(report, str(output_path))
        else:
            output_path = self.output_dir / f"{model_safe}_{timestamp}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        return output_path
