import sys
import time
from queue import Empty, Queue
from threading import Thread

_BAR_SIZE = 20
_KILOBYTE = 1024
_FINISHED_BAR = "#"
_REMAINING_BAR = "-"

_UNKNOWN_SIZE = "?"
_STR_MEGABYTE = " MB"

_HOURS_OF_ELAPSED = "%d:%02d:%02d"
_MINUTES_OF_ELAPSED = "%02d:%02d"

_RATE_FORMAT = "%5.2f"
_PERCENTAGE_FORMAT = "%3d%%"
_HUMANINZED_FORMAT = "%0.2f"

_DISPLAY_FORMAT = "|%s| %s/%s %s [elapsed: %s left: %s, %s MB/sec]"

_REFRESH_CHAR = "\r"


class Progress(Thread):
    """
    A class that extends Thread to display progress information.

    This class is used to display progress information on the screen at regular
    intervals. It is initialized with an interval and a standard output stream.
    The `set_meta` method is used to set metadata for the object before
    uploading, and the `run` method is used to display the progress information.

    Attributes:
        interval (int): The time interval to be displayed on the screen.
        stdout (file-like object): The standard output stream where progress
                                   information is written.
    """

    def __init__(self, interval=1, stdout=sys.stdout):
        Thread.__init__(self)
        self.daemon = True
        self.total_length = 0
        self.interval = interval
        self.object_name = None

        self.last_printed_len = 0
        self.current_size = 0

        self.display_queue = Queue()
        self.initial_time = time.time()
        self.stdout = stdout
        self.start()

    def set_meta(self, total_length, object_name):
        """
        Metadata settings for the object. This method called before uploading
        object
        :param total_length: Total length of object.
        :param object_name: Object name to be showed.
        """
        self.total_length = total_length
        self.object_name = object_name
        self.prefix = self.object_name + ": " if self.object_name else ""

    def run(self):
        displayed_time = 0
        while True:
            try:
                # display every interval secs
                task = self.display_queue.get(timeout=self.interval)
            except Empty:
                elapsed_time = time.time() - self.initial_time
                if elapsed_time > displayed_time:
                    displayed_time = elapsed_time
                self.print_status(
                    current_size=self.current_size,
                    total_length=self.total_length,
                    displayed_time=displayed_time,
                    prefix=self.prefix,
                )
                continue

            current_size, total_length = task
            displayed_time = time.time() - self.initial_time
            self.print_status(
                current_size=current_size,
                total_length=total_length,
                displayed_time=displayed_time,
                prefix=self.prefix,
            )
            self.display_queue.task_done()
            if current_size == total_length:
                # once we have done uploading everything return
                self.done_progress()
                return

    def update(self, size):
        if not isinstance(size, int):
            raise ValueError(
                "{} type can not be displayed. "
                "Please change it to Int.".format(type(size))
            )

        self.current_size += size
        self.display_queue.put((self.current_size, self.total_length))

    def done_progress(self):
        self.total_length = 0
        self.object_name = None
        self.last_printed_len = 0
        self.current_size = 0

    def print_status(self, current_size, total_length, displayed_time, prefix):
        formatted_str = prefix + format_string(
            current_size, total_length, displayed_time
        )
        self.stdout.write(
            _REFRESH_CHAR
            + formatted_str
            + " " * max(self.last_printed_len - len(formatted_str), 0)
        )
        self.stdout.flush()
        self.last_printed_len = len(formatted_str)


def seconds_to_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, m = divmod(minutes, 60)
    if hours:
        return _HOURS_OF_ELAPSED % (hours, m, seconds)
    else:
        return _MINUTES_OF_ELAPSED % (m, seconds)


def format_string(current_size, total_length, elapsed_time):
    n_to_mb = current_size / _KILOBYTE / _KILOBYTE
    elapsed_str = seconds_to_time(elapsed_time)

    rate = _RATE_FORMAT % (n_to_mb / elapsed_time) if elapsed_time else _UNKNOWN_SIZE
    frac = float(current_size) / total_length
    bar_length = int(frac * _BAR_SIZE)
    bar = _FINISHED_BAR * bar_length + _REMAINING_BAR * (_BAR_SIZE - bar_length)
    percentage = _PERCENTAGE_FORMAT % (frac * 100)
    left_str = (
        seconds_to_time(elapsed_time / current_size * (total_length - current_size))
        if current_size
        else _UNKNOWN_SIZE
    )

    humanized_total = (
        _HUMANINZED_FORMAT % (total_length / _KILOBYTE / _KILOBYTE) + _STR_MEGABYTE
    )
    humanized_n = _HUMANINZED_FORMAT % n_to_mb + _STR_MEGABYTE

    return _DISPLAY_FORMAT % (
        bar,
        humanized_n,
        humanized_total,
        percentage,
        elapsed_str,
        left_str,
        rate,
    )
