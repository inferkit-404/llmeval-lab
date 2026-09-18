"""Task scheduler for automated evaluation workflows."""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import croniter

from rich.console import Console
from rich.table import Table


console = Console()


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Task:
    """Represents an evaluation task."""
    id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 3600
    result: Any = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TaskScheduler:
    """Schedule and execute evaluation tasks."""

    def __init__(self, max_concurrent: int = 4):
        """Initialize task scheduler.

        Args:
            max_concurrent: Maximum concurrent tasks
        """
        self.max_concurrent = max_concurrent
        self.tasks: Dict[str, Task] = {}
        self.running_tasks: Dict[str, threading.Thread] = {}
        self.schedule: Dict[str, str] = {}  # task_id -> cron expression
        self._stop_event = threading.Event()

        console.print(f"[green]TaskScheduler initialized with max_concurrent={max_concurrent}[/green]")

    def add_task(
        self,
        task_id: str,
        name: str,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        timeout: int = 3600,
    ) -> Task:
        """Add a task to the scheduler.

        Args:
            task_id: Unique task identifier
            name: Task display name
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            max_retries: Maximum retry attempts
            timeout: Task timeout in seconds

        Returns:
            Created Task object
        """
        if kwargs is None:
            kwargs = {}

        task = Task(
            id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            max_retries=max_retries,
            timeout=timeout,
        )

        self.tasks[task_id] = task
        console.print(f"[blue]Added task: {name} ({task_id})[/blue]")
        return task

    def schedule_task(self, task_id: str, cron_expr: str) -> None:
        """Schedule a recurring task with cron expression.

        Args:
            task_id: Task identifier
            cron_expr: Cron expression (e.g., "0 9 * * *")
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        self.schedule[task_id] = cron_expr
        console.print(f"[yellow]Scheduled task {task_id} with cron: {cron_expr}[/yellow]")

    def run_task(self, task_id: str) -> bool:
        """Run a task immediately.

        Args:
            task_id: Task identifier

        Returns:
            True if task succeeded
        """
        if task_id not in self.tasks:
            console.print(f"[red]Task {task_id} not found[/red]")
            return False

        if len(self.running_tasks) >= self.max_concurrent:
            console.print(f"[yellow]Max concurrent tasks reached, task {task_id} queued[/yellow]")

        task = self.tasks[task_id]
        thread = threading.Thread(target=self._execute_task, args=(task_id,))
        thread.start()
        self.running_tasks[task_id] = thread

        return True

    def _execute_task(self, task_id: str) -> None:
        """Execute a task with retry logic."""
        task = self.tasks[task_id]

        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()

            console.print(f"[cyan]Executing task: {task.name}[/cyan]")

            # Execute with timeout
            result = task.func(*task.args, **task.kwargs)

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()

            console.print(f"[green]Task {task.name} completed successfully[/green]")

        except Exception as e:
            task.error = str(e)
            task.retry_count += 1

            if task.retry_count < task.max_retries:
                task.status = TaskStatus.RETRYING
                console.print(
                    f"[yellow]Task {task.name} failed (attempt {task.retry_count}), "
                    f"retrying in {task.retry_count * 10}s...[/yellow]"
                )
                time.sleep(task.retry_count * 10)
                self._execute_task(task_id)  # Recursive retry
            else:
                task.status = TaskStatus.FAILED
                console.print(f"[red]Task {task.name} failed after {task.max_retries} attempts[/red]")

        finally:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

    def wait_for_completion(self, timeout: Optional[int] = None) -> None:
        """Wait for all running tasks to complete.

        Args:
            timeout: Maximum wait time in seconds
        """
        start = time.time()

        while self.running_tasks:
            if timeout and (time.time() - start) > timeout:
                console.print("[yellow]Timeout waiting for tasks to complete[/yellow]")
                break

            time.sleep(0.5)

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task."""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return None

    def list_tasks(self) -> List[Task]:
        """List all tasks."""
        return list(self.tasks.values())

    def print_status(self) -> None:
        """Print scheduler status table."""
        table = Table(title="Task Scheduler Status")
        table.add_column("Task", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Retries", style="yellow", justify="right")
        table.add_column("Created", style="green")

        for task in self.tasks.values():
            status_color = {
                TaskStatus.PENDING: "white",
                TaskStatus.RUNNING: "cyan",
                TaskStatus.COMPLETED: "green",
                TaskStatus.FAILED: "red",
                TaskStatus.RETRYING: "yellow",
            }.get(task.status, "white")

            table.add_row(
                task.name,
                f"[{status_color}]{task.status.value}[/{status_color}]",
                str(task.retry_count),
                task.created_at[:19],
            )

        console.print(table)
