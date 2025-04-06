import threading
import queue
import time
from typing import Dict, Any, Optional, Callable, List, Tuple

from inferno.utils.logger import get_logger

logger = get_logger(__name__)


class Task:
    """
    Represents a task in the queue.
    """
    def __init__(self, task_id: str, task_type: str, params: Dict[str, Any], callback: Optional[Callable] = None):
        """
        Initialize a task.

        Args:
            task_id: Unique identifier for the task
            task_type: Type of task (e.g., 'completion', 'chat_completion')
            params: Parameters for the task
            callback: Optional callback function to call when the task is complete
        """
        self.task_id = task_id
        self.task_type = task_type
        self.params = params
        self.callback = callback
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.status = "pending"  # pending, running, completed, failed

    def start(self):
        """
        Mark the task as started.
        """
        self.started_at = time.time()
        self.status = "running"

    def complete(self, result: Any):
        """
        Mark the task as completed.

        Args:
            result: Result of the task
        """
        self.completed_at = time.time()
        self.result = result
        self.status = "completed"

        # Call the callback if provided
        if self.callback:
            try:
                self.callback(self)
            except Exception as e:
                logger.error(f"Error in task callback: {e}")

    def fail(self, error: str):
        """
        Mark the task as failed.

        Args:
            error: Error message
        """
        self.completed_at = time.time()
        self.error = error
        self.status = "failed"

        # Call the callback if provided
        if self.callback:
            try:
                self.callback(self)
            except Exception as e:
                logger.error(f"Error in task callback: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary.

        Returns:
            Dictionary representation of the task
        """
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error
        }


class TaskQueue:
    """
    Queue for managing and processing tasks.
    """
    def __init__(self, max_workers: int = 4, max_queue_size: int = 100):
        """
        Initialize the task queue.

        Args:
            max_workers: Maximum number of worker threads
            max_queue_size: Maximum size of the queue
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.tasks: Dict[str, Task] = {}
        self.workers: List[threading.Thread] = []
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self.running = False

        # Task handlers for different task types
        self.handlers: Dict[str, Callable] = {}

    def start(self):
        """
        Start the task queue workers.
        """
        with self.lock:
            if self.running:
                return

            self.running = True

            # Start worker threads
            for i in range(self.max_workers):
                worker = threading.Thread(target=self._worker_loop, name=f"TaskWorker-{i}")
                worker.daemon = True  # Daemon threads exit when the main thread exits
                worker.start()
                self.workers.append(worker)

            logger.info(f"Started {self.max_workers} task queue workers")

    def stop(self):
        """
        Stop the task queue workers.
        """
        with self.lock:
            if not self.running:
                return

            self.running = False

            # Clear the queue
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    break

            # Wait for workers to exit
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=1.0)

            self.workers.clear()
            logger.info("Stopped task queue workers")

    def register_handler(self, task_type: str, handler: Callable):
        """
        Register a handler for a task type.

        Args:
            task_type: Type of task
            handler: Handler function for the task
        """
        with self.lock:
            self.handlers[task_type] = handler
            logger.info(f"Registered handler for task type '{task_type}'")

    def add_task(self, task: Task) -> bool:
        """
        Add a task to the queue.

        Args:
            task: Task to add

        Returns:
            True if the task was added, False otherwise
        """
        with self.lock:
            # Start the workers if not already running
            if not self.running:
                self.start()

            # Check if the queue is full
            if self.queue.full():
                logger.warning(f"Task queue is full, rejecting task {task.task_id}")
                return False

            # Add the task to the dictionary
            self.tasks[task.task_id] = task

            # Add the task to the queue
            try:
                self.queue.put(task.task_id, block=False)
                logger.info(f"Added task {task.task_id} to queue")
                return True
            except queue.Full:
                # Remove the task from the dictionary
                del self.tasks[task.task_id]
                logger.warning(f"Task queue is full, rejecting task {task.task_id}")
                return False

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.

        Args:
            task_id: ID of the task to get

        Returns:
            Task object or None if not found
        """
        with self.lock:
            return self.tasks.get(task_id)

    def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all tasks, optionally filtered by status.

        Args:
            status: Optional status to filter by

        Returns:
            List of task dictionaries
        """
        with self.lock:
            if status:
                return [task.to_dict() for task in self.tasks.values() if task.status == status]
            else:
                return [task.to_dict() for task in self.tasks.values()]

    def _worker_loop(self):
        """
        Worker thread loop for processing tasks.
        """
        while self.running:
            try:
                # Get a task from the queue
                task_id = self.queue.get(block=True, timeout=1.0)

                # Get the task from the dictionary
                with self.lock:
                    task = self.tasks.get(task_id)

                if task is None:
                    logger.warning(f"Task {task_id} not found in dictionary")
                    self.queue.task_done()
                    continue

                # Mark the task as started
                task.start()

                # Process the task
                try:
                    # Get the handler for this task type
                    handler = self.handlers.get(task.task_type)

                    if handler is None:
                        raise ValueError(f"No handler registered for task type '{task.task_type}'")

                    # Call the handler
                    result = handler(**task.params)

                    # Mark the task as completed
                    task.complete(result)

                except Exception as e:
                    # Mark the task as failed
                    logger.error(f"Error processing task {task_id}: {e}")
                    task.fail(str(e))

                # Mark the task as done in the queue
                self.queue.task_done()

            except queue.Empty:
                # Queue timeout, just continue
                pass
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")

        logger.info(f"Worker {threading.current_thread().name} exiting")