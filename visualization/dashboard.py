"""Dashboard for visualizing evaluation results."""

from typing import Dict, List, Any, Optional
from pathlib import Path

from .charts import ChartGenerator


class Dashboard:
    """Create interactive dashboards for evaluation results."""

    def __init__(self, output_dir: str = "./dashboard"):
        """Initialize dashboard.

        Args:
            output_dir: Directory to save dashboard files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chart_generator = ChartGenerator(output_dir)

    def create_model_comparison_dashboard(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: Optional[str] = None,
    ) -> str:
        """Create a model comparison dashboard.

        Args:
            results: Dict mapping model names to their benchmark scores
            output_path: Optional output path

        Returns:
            HTML content as string
        """
        models = list(results.keys())
        benchmarks = list(results[models[0]].keys()) if models else []

        # Generate charts
        radar_html = self.chart_generator.create_radar_chart(
            {m: results[m] for m in models[:2]},  # Compare up to 2 models
            title="Model Capability Comparison",
        )

        bar_html = self.chart_generator.create_benchmark_bar_chart(
            results,
            title="Benchmark Scores by Model",
        )

        # Build dashboard HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
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
        h2 {{ margin-top: 0; color: #333; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Model Comparison Dashboard</h1>
        <p>Comparing {len(models)} models across {len(benchmarks)} benchmarks</p>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Radar Chart</h2>
            {radar_html}
        </div>
        <div class="card">
            <h2>Benchmark Comparison</h2>
            {bar_html}
        </div>
    </div>
</body>
</html>
        """

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

        return html

    def create_checkpoint_tracking_dashboard(
        self,
        history: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> str:
        """Create a checkpoint tracking dashboard.

        Args:
            history: List of checkpoint evaluation records
            output_path: Optional output path

        Returns:
            HTML content as string
        """
        chart_html = self.chart_generator.create_line_chart(
            data=history,
            x_field="step",
            y_fields=["accuracy"] if history and "accuracy" in history[0] else [],
            title="Checkpoint Capability Tracking",
        )

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Checkpoint Tracking Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h2 {{ margin-top: 0; color: #333; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Checkpoint Tracking Dashboard</h1>
        <p>Monitoring model capability changes over {len(history)} checkpoints</p>
    </div>

    <div class="card">
        <h2>Capability Evolution</h2>
        {chart_html}
    </div>
</body>
</html>
        """

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

        return html
