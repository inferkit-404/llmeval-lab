"""Chart generation utilities for visualization."""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class ChartGenerator:
    """Generate various charts for evaluation results."""

    def __init__(self, output_dir: str = "./charts"):
        """Initialize chart generator.

        Args:
            output_dir: Directory to save chart HTML files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_radar_chart(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Radar Chart",
        output_path: Optional[str] = None,
    ) -> str:
        """Create a radar chart for capability comparison.

        Args:
            data: Dict mapping labels to value dicts
            title: Chart title
            output_path: Optional output path

        Returns:
            HTML content with Plotly chart
        """
        # Extract categories and values
        labels = list(data.keys())
        if not labels:
            return "<p>No data available</p>"

        first_label = labels[0]
        categories = list(data[first_label].keys())
        values_by_model = {}

        for label, values in data.items():
            values_by_model[label] = [values.get(c, 0) for c in categories]

        traces = []
        colors = ["#667eea", "#f59e0b", "#22c55e", "#ef4444", "#8b5cf6"]

        for i, (label, values) in enumerate(values_by_model.items()):
            values_plot = values + [values[0]]  # Close the polygon
            theta_plot = categories + [categories[0]]

            traces.append({
                "type": "scatterpolar",
                "r": values_plot,
                "theta": theta_plot,
                "fill": "toself" if i == 0 else "none",
                "name": label,
                "line": {"color": colors[i % len(colors)], "width": 2},
            })

        layout = {
            "title": {"text": title},
            "polar": {
                "radialaxis": {"visible": True, "range": [0, 1]}
            },
            "showlegend": True,
        }

        chart_html = f'<div id="radar-chart"></div><script>'
        chart_html += f'Plotly.newPlot("radar-chart", {json.dumps(traces)}, {json.dumps(layout)});'
        chart_html += '</script>'

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(chart_html)

        return chart_html

    def create_benchmark_bar_chart(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Benchmark Scores",
        output_path: Optional[str] = None,
    ) -> str:
        """Create a bar chart comparing benchmark scores.

        Args:
            data: Dict mapping model names to benchmark scores
            title: Chart title
            output_path: Optional output path

        Returns:
            HTML content with Plotly chart
        """
        benchmarks = list(next(iter(data.values())).keys()) if data else []
        models = list(data.keys())

        traces = []
        colors = ["#667eea", "#f59e0b", "#22c55e", "#ef4444", "#8b5cf6"]

        for i, model in enumerate(models):
            values = [data[model].get(b, 0) for b in benchmarks]
            traces.append({
                "type": "bar",
                "x": benchmarks,
                "y": [v * 100 for v in values],
                "name": model,
                "marker": {"color": colors[i % len(colors)]},
            })

        layout = {
            "title": {"text": title},
            "xaxis": {"title": "Benchmark"},
            "yaxis": {"title": "Accuracy (%)", "range": [0, 100]},
            "barmode": "group",
            "margin": {"t": 60},
        }

        chart_html = f'<div id="bar-chart"></div><script>'
        chart_html += f'Plotly.newPlot("bar-chart", {json.dumps(traces)}, {json.dumps(layout)});'
        chart_html += '</script>'

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(chart_html)

        return chart_html

    def create_line_chart(
        self,
        data: List[Dict[str, Any]],
        x_field: str,
        y_fields: List[str],
        title: str = "Line Chart",
        output_path: Optional[str] = None,
    ) -> str:
        """Create a line chart for tracking changes over time.

        Args:
            data: List of data points
            x_field: Field to use for x-axis
            y_fields: Fields to plot on y-axis
            title: Chart title
            output_path: Optional output path

        Returns:
            HTML content with Plotly chart
        """
        if not data or not y_fields:
            return "<p>No data available</p>"

        traces = []
        colors = ["#667eea", "#f59e0b", "#22c55e", "#ef4444", "#8b5cf6"]

        for i, y_field in enumerate(y_fields):
            x_values = [d.get(x_field, 0) for d in data]
            y_values = [d.get(y_field, 0) for d in data]

            traces.append({
                "type": "scatter",
                "x": x_values,
                "y": y_values,
                "mode": "lines+markers",
                "name": y_field,
                "line": {"color": colors[i % len(colors)], "width": 2},
            })

        layout = {
            "title": {"text": title},
            "xaxis": {"title": x_field},
            "yaxis": {"title": "Value"},
            "margin": {"t": 60},
        }

        chart_html = f'<div id="line-chart"></div><script>'
        chart_html += f'Plotly.newPlot("line-chart", {json.dumps(traces)}, {json.dumps(layout)});'
        chart_html += '</script>'

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(chart_html)

        return chart_html

    def create_heatmap(
        self,
        data: List[List[float]],
        x_labels: List[str],
        y_labels: List[str],
        title: str = "Heatmap",
        output_path: Optional[str] = None,
    ) -> str:
        """Create a heatmap chart.

        Args:
            data: 2D array of values
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis
            title: Chart title
            output_path: Optional output path

        Returns:
            HTML content with Plotly chart
        """
        trace = {
            "type": "heatmap",
            "z": data,
            "x": x_labels,
            "y": y_labels,
            "colorscale": "Viridis",
        }

        layout = {
            "title": {"text": title},
            "margin": {"t": 60, "l": 100},
        }

        chart_html = f'<div id="heatmap-chart"></div><script>'
        chart_html += f'Plotly.newPlot("heatmap-chart", {json.dumps([trace])}, {json.dumps(layout)});'
        chart_html += '</script>'

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(chart_html)

        return chart_html
