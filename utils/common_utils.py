import sys
import time
from pathlib import Path

from omegaconf import DictConfig

import envs
from envs.base_env import BaseEnv


def snakecase_to_pascalcase(s: str) -> str:
    components = s.split("_")
    return "".join(word.capitalize() for word in components)


def make_envs(config: DictConfig) -> BaseEnv:
    """Create environment based on task backend."""
    backend = config.task.get("backend")
    TaskSuite = getattr(envs, backend + "_env")
    return TaskSuite.make_envs(config)


def print_info(*message):
    print("\033[96m", *message, "\033[0m")


class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.time_total = 0.0

    def on(self):
        assert self.start_time is None, "Timer {} is already turned on!".format(self.name)
        self.start_time = time.time()

    def off(self):
        assert self.start_time is not None, "Timer {} not started yet!".format(self.name)
        self.time_total += time.time() - self.start_time
        self.start_time = None

    def report(self):
        print_info("Time report [{}]: {:.2f} seconds".format(self.name, self.time_total))

    def clear(self):
        self.start_time = None
        self.time_total = 0.0


class TimeReport:
    def __init__(self):
        self.timers = {}

    def add_timer(self, name):
        assert name not in self.timers, "Timer {} already exists!".format(name)
        self.timers[name] = Timer(name=name)

    def start_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].on()

    def end_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].off()

    def report(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
        else:
            print_info("------------Time Report------------")
            for timer_name in self.timers.keys():
                self.timers[timer_name].report()
            print_info("-----------------------------------")

    def clear_timer(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].clear()
        else:
            for timer_name in self.timers.keys():
                self.timers[timer_name].clear()

    def pop_timer(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
            del self.timers[name]
        else:
            self.report()
            self.timers = {}


class TeeOutput:
    """A class that redirects stdout/stderr to both console and a log file."""

    def __init__(self, log_file_path: Path, stream):
        self.log_file = open(log_file_path, "a", buffering=1)  # Line buffered
        self.original_stream = stream

    def write(self, message):
        """Write to both console and log file."""
        self.original_stream.write(message)
        # Check if log file is still open before writing
        if self.log_file and not self.log_file.closed:
            try:
                self.log_file.write(message)
                self.log_file.flush()
            except (ValueError, OSError):
                # File is closed or other I/O error, ignore
                pass

    def flush(self):
        """Flush both streams."""
        self.original_stream.flush()
        if self.log_file and not self.log_file.closed:
            try:
                self.log_file.flush()
            except (ValueError, OSError):
                # File is closed or other I/O error, ignore
                pass

    def close(self):
        """Close the log file."""
        if self.log_file:
            self.log_file.close()

    def __getattr__(self, name):
        """Delegate other attributes to the original stream."""
        return getattr(self.original_stream, name)


class TeeStdoutStderr:
    """Context manager that redirects both stdout and stderr to log file.

    Also configures Python's warnings and logging modules to ensure all output
    (including warnings and errors) is captured in the log file.
    """

    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path
        self.tee_stdout = None
        self.tee_stderr = None
        self.original_stdout = None
        self.original_stderr = None
        self.original_excepthook = None
        self.original_showwarning = None
        self.original_logging_handler = None

    def __enter__(self):
        import logging
        import warnings

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.tee_stdout = TeeOutput(self.log_file_path, sys.stdout)
        self.tee_stderr = TeeOutput(self.log_file_path, sys.stderr)
        sys.stdout = self.tee_stdout
        sys.stderr = self.tee_stderr

        # Configure sys.excepthook to capture full tracebacks
        self.original_excepthook = sys.excepthook

        def excepthook_redirected(exc_type, exc_value, exc_traceback):
            """Redirect exceptions to stderr (which is already redirected to log file)."""
            # Import traceback here to avoid circular imports
            import traceback

            # Print full traceback to stderr (which is redirected to log file)
            # This ensures the traceback is captured even if the program exits
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr, limit=None, chain=True)
            # Flush to ensure it's written immediately to the log file
            sys.stderr.flush()
            # Also call the original excepthook for consistency (though traceback is already printed)
            if self.original_excepthook is not None:
                self.original_excepthook(exc_type, exc_value, exc_traceback)

        sys.excepthook = excepthook_redirected

        # Configure warnings to use stderr (which is now redirected)
        self.original_showwarning = warnings.showwarning

        def showwarning_redirected(message, category, filename, lineno, file=None, line=None):
            """Redirect warnings to stderr (which is already redirected to log file)."""
            if file is None:
                file = sys.stderr
            self.original_showwarning(message, category, filename, lineno, file=file, line=line)

        warnings.showwarning = showwarning_redirected

        # Configure logging to also write to stderr (which is redirected)
        # Get the root logger and ensure it has a handler that writes to stderr
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            root_logger.addHandler(handler)
            root_logger.setLevel(logging.WARNING)  # Capture WARNING and above by default
            self.original_logging_handler = handler

        # Configure Genesis logger specifically to ensure gs.warn() output is captured
        # Genesis logger might have propagate=False, so we need to add a handler directly
        genesis_logger = logging.getLogger("genesis")
        # Check if Genesis logger already has a StreamHandler to stderr
        has_genesis_stderr = any(
            isinstance(h, logging.StreamHandler) and h.stream == sys.stderr for h in genesis_logger.handlers
        )
        if not has_genesis_stderr:
            genesis_handler = logging.StreamHandler(sys.stderr)
            genesis_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            genesis_logger.addHandler(genesis_handler)
            genesis_logger.setLevel(logging.WARNING)
        # Ensure propagation is enabled so it also goes to root logger
        genesis_logger.propagate = True

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import logging
        import traceback
        import warnings

        # If an exception occurred, print traceback before restoring streams
        if exc_type is not None:
            # Print traceback to stderr (which is redirected to log file)
            traceback.print_exception(exc_type, exc_val, exc_tb, file=sys.stderr, limit=None, chain=True)
            sys.stderr.flush()

        # Restore sys.excepthook
        if hasattr(self, "original_excepthook") and self.original_excepthook is not None:
            sys.excepthook = self.original_excepthook

        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Restore original warnings.showwarning
        if self.original_showwarning is not None:
            warnings.showwarning = self.original_showwarning

        # Restore logging if we modified it
        if self.original_logging_handler is not None:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.original_logging_handler)

        # Don't close the log files - keep them open so they can receive writes
        # during program exit (e.g., from atexit handlers)
        # The files will be automatically closed when Python exits
