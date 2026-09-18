"""Execution monitor for tracking task execution."""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.spinner import Spinner


console = Console()


@dataclass
class ExecutionRecord:
    """Record of a single execution."""
    task_id: str
    task_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None


class ExecutionMonitor:
    """Monitor task execution in real-time."""

    def __init__(self, update_interval: float = 1.0):
        """Initialize execution monitor.

        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.executions: Dict[str, List[ExecutionRecord]] = defaultdict(list)
        self.active_executions: Dict[str, ExecutionRecord] = {}
        self._running = False

    def start_execution(self, task_id: str, task_name: str) -> ExecutionRecord:
        """Start tracking an execution.

        Args:
            task_id: Task identifier
            task_name: Task display name

        Returns:
            ExecutionRecord
        """
        record = ExecutionRecord(
            task_id=task_id,
            task_name=task_name,
            start_time=time.time(),
        )

        self.active_executions[task_id] = record
        return record

    def end_execution(
        self,
        task_id: str,
        status: str = "completed",
        error: Optional[str] = None,
    ) -> None:
        """End tracking an execution.

        Args:
            task_id: Task identifier
            status: Final status
            error: Optional error message
        """
        if task_id in self.active_executions:
            record = self.active_executions[task_id]
            record.end_time = time.time()
            record.status = status
            record.error = error

            self.executions[record.task_id].append(record)
            del self.active_executions[task_id]

    def update_metrics(self, task_id: str, metrics: Dict[str, Any]) -> None:
        """Update execution metrics.

        Args:
            task_id: Task identifier
            metrics: Metrics dictionary
        """
        if task_id in self.active_executions:
            self.active_executions[task_id].metrics.update(metrics)

    def get_active_count(self) -> int:
        """Get number of active executions."""
        return len(self.active_executions)

    def get_total_executions(self, task_id: Optional[str] = None) -> int:
        """Get total number of executions."""
        if task_id:
            return len(self.executions.get(task_id, []))
        return sum(len(v) for v in self.executions.values())

    def get_success_rate(self, task_id: Optional[str] = None) -> float:
        """Calculate success rate.

        Args:
            task_id: Optional task ID to filter by

        Returns:
            Success rate (0.0 to 1.0)
        """
        if task_id:
            records = self.executions.get(task_id, [])
        else:
            records = [r for records in self.executions.values() for r in records]

        if not records:
            return 0.0

        successful = sum(1 for r in records if r.status == "completed")
        return successful / len(records)

    def get_average_duration(self, task_id: Optional[str] = None) -> float:
        """Calculate average execution duration.

        Args:
            task_id: Optional task ID to filter by

        Returns:
            Average duration in seconds
        """
        if task_id:
            records = self.executions.get(task_id, [])
        else:
            records = [r for records in self.executions.values() for r in records]

        completed = [r for r in records if r.duration is not None]
        if not completed:
            return 0.0

        return sum(r.duration for r in completed) / len(completed)

    def print_summary(self) -> None:
        """Print execution summary."""
        table = Table(title="Execution Summary")
        table.add_column("Task", style="cyan")
        table.add_column("Total", style="white", justify="right")
        table.add_column("Active", style="yellow", justify="right")
        table.add_column("Success Rate", style="green", justify="right")
        table.add_column("Avg Duration", style="magenta", justify="right")

        for task_id, records in self.executions.items():
            active = 1 if task_id in self.active_executions else 0
            total = len(records)
            successful = sum(1 for r in records if r.status == "completed")
            success_rate = successful / total if total > 0 else 0.0
            avg_duration = sum(r.duration for r in records if r.duration) / len(records) if records else 0.0

            task_name = records[0].task_name if records else task_id

            table.add_row(
                task_name,
                str(total),
                str(active),
                f"{success_rate:.1%}",
                f"{avg_duration:.1f}s",
            )

        console.print(table)

    def start_live_display(self) -> None:
        """Start live display of execution status."""
        self._running = True

        with Live(self._generate_table(), console=console, refresh_per_second=1) as live:
            while self._running:
                live.update(self._generate_table())
                time.sleep(self.update_interval)

    def stop_live_display(self) -> None:
        """Stop live display."""
        self._running = False

    def _generate_table(self) -> Table:
        """Generate current status table."""
        table = Table(title=f"Active Executions ({self.get_active_count()} running)")
        table.add_column("Task", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Duration", style="yellow")
        table.add_column("Metrics", style="green")

        for record in self.active_executions.values():
            duration = time.time() - record.start_time
            metrics_str = ", ".join(f"{k}={v}" for k, v in record.metrics.items())

            table.add_row(
                record.task_name,
                f"[cyan]{record.status}[/cyan]",
                f"{duration:.1f}s",
                metrics_str[:50],
            )

        if not self.active_executions:
            table.add_row("[dim]No active executions[/dim]", "", "", "")

        return table
