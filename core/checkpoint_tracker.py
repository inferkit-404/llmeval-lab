"""Checkpoint tracking system for monitoring model capability changes."""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from config import CheckpointTrackerConfig


console = Console()


@dataclass
class CheckpointRecord:
    """Record of a single checkpoint evaluation."""
    checkpoint_path: str
    step: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "step": self.step,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointRecord":
        return cls(
            checkpoint_path=data["checkpoint_path"],
            step=data.get("step"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {}),
        )


class CheckpointTracker:
    """Track model capabilities across checkpoints."""

    def __init__(self, config: Optional[CheckpointTrackerConfig] = None):
        """Initialize checkpoint tracker.

        Args:
            config: Tracker configuration
        """
        self.config = config or CheckpointTrackerConfig()
        self.storage_path = Path(self.config.storage_path)
        self.records: Dict[str, List[CheckpointRecord]] = defaultdict(list)
        self._load_existing_records()

    def _load_existing_records(self) -> None:
        """Load existing records from storage."""
        if not self.storage_path.exists():
            self.storage_path.mkdir(parents=True, exist_ok=True)
            return

        records_file = self.storage_path / "checkpoint_records.json"
        if records_file.exists():
            try:
                with open(records_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for model_name, records_list in data.items():
                        self.records[model_name] = [
                            CheckpointRecord.from_dict(r) for r in records_list
                        ]
                console.print(f"[green]Loaded {len(self.records)} existing checkpoint records[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load existing records: {e}[/yellow]")

    def _save_records(self) -> None:
        """Save records to storage."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        records_file = self.storage_path / "checkpoint_records.json"

        data = {
            model_name: [r.to_dict() for r in records]
            for model_name, records in self.records.items()
        }

        with open(records_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def record(
        self,
        model_name: str,
        checkpoint_path: str,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CheckpointRecord:
        """Record evaluation results for a checkpoint.

        Args:
            model_name: Name of the model
            checkpoint_path: Path to the checkpoint
            metrics: Dictionary of metric names and values
            step: Optional training step
            metadata: Optional additional metadata

        Returns:
            The created CheckpointRecord
        """
        record = CheckpointRecord(
            checkpoint_path=checkpoint_path,
            step=step,
            metrics=metrics,
            metadata=metadata or {},
        )

        self.records[model_name].append(record)
        self._save_records()

        console.print(f"[green]Recorded checkpoint: {checkpoint_path}[/green]")
        return record

    def get_history(
        self,
        model_name: str,
        metric: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get evaluation history for a model.

        Args:
            model_name: Name of the model
            metric: Optional specific metric to retrieve

        Returns:
            List of historical records
        """
        records = self.records.get(model_name, [])

        if metric:
            return [
                {
                    "checkpoint_path": r.checkpoint_path,
                    "step": r.step,
                    "timestamp": r.timestamp,
                    "value": r.metrics.get(metric),
                }
                for r in records
                if metric in r.metrics
            ]

        return [r.to_dict() for r in records]

    def analyze_capability_change(
        self,
        model_name: str,
        metric: str,
    ) -> Dict[str, Any]:
        """Analyze capability changes for a metric.

        Args:
            model_name: Name of the model
            metric: Metric to analyze

        Returns:
            Analysis result with trend and alerts
        """
        history = self.get_history(model_name, metric)

        if len(history) < 2:
            return {
                "trend": "insufficient_data",
                "values": [h["value"] for h in history if h["value"] is not None],
                "alerts": [],
            }

        values = [h["value"] for h in history if h["value"] is not None]

        # Calculate trend
        first_value = values[0]
        last_value = values[-1]
        change = last_value - first_value
        change_pct = (change / first_value * 100) if first_value != 0 else 0

        # Detect anomalies
        alerts = []
        if len(values) >= 3:
            # Check for sudden drops
            for i in range(1, len(values)):
                drop = values[i-1] - values[i]
                drop_pct = (drop / values[i-1] * 100) if values[i-1] != 0 else 0
                if drop_pct > self.config.alert_threshold * 100:
                    alerts.append({
                        "type": "degradation",
                        "step": i,
                        "previous_value": values[i-1],
                        "current_value": values[i],
                        "drop_pct": drop_pct,
                    })

        # Determine trend
        if change_pct > 5:
            trend = "improving"
        elif change_pct < -5:
            trend = "degrading"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "first_value": first_value,
            "last_value": last_value,
            "change": change,
            "change_pct": change_pct,
            "values": values,
            "alerts": alerts,
            "model_name": model_name,
            "metric": metric,
        }

    def print_capability_report(self, model_name: str) -> None:
        """Print a capability analysis report for a model."""
        table = Table(title=f"Capability Report: {model_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("First", style="green", justify="right")
        table.add_column("Last", style="green", justify="right")
        table.add_column("Change", style="yellow", justify="right")
        table.add_column("Trend", style="magenta")

        # Get all metrics
        all_metrics = set()
        for record in self.records.get(model_name, []):
            all_metrics.update(record.metrics.keys())

        for metric in sorted(all_metrics):
            analysis = self.analyze_capability_change(model_name, metric)
            change_str = f"{analysis['change_pct']:+.1f}%"

            trend_emoji = {
                "improving": "📈",
                "degrading": "📉",
                "stable": "➡️",
                "insufficient_data": "❓",
            }.get(analysis["trend"], "")

            table.add_row(
                metric,
                f"{analysis['first_value']:.4f}" if analysis['first_value'] else "N/A",
                f"{analysis['last_value']:.4f}" if analysis['last_value'] else "N/A",
                change_str,
                f"{trend_emoji} {analysis['trend']}",
            )

        console.print(table)

        # Print alerts if any
        for metric in sorted(all_metrics):
            analysis = self.analyze_capability_change(model_name, metric)
            if analysis["alerts"]:
                console.print(f"[red]⚠️ Alert: {metric} degradation detected![/red]")
                for alert in analysis["alerts"]:
                    console.print(
                        f"  Step {alert['step']}: {alert['previous_value']:.4f} → {alert['current_value']:.4f} "
                        f"(-{alert['drop_pct']:.1f}%)"
                    )
