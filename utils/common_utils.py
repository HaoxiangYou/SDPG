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
    """Context manager that redirects both stdout and stderr to log file."""

    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path
        self.tee_stdout = None
        self.tee_stderr = None
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.tee_stdout = TeeOutput(self.log_file_path, sys.stdout)
        self.tee_stderr = TeeOutput(self.log_file_path, sys.stderr)
        sys.stdout = self.tee_stdout
        sys.stderr = self.tee_stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        # Don't close the log files - keep them open so they can receive writes
        # during program exit (e.g., from atexit handlers)
        # The files will be automatically closed when Python exits
