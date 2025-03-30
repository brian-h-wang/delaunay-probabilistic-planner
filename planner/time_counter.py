from collections import defaultdict
import time
import numpy as np


class TimeCounter:
    """Class for counting execution time of different steps of the simulation pipeline.

    Usage
    -----
    tc = TimeCounter()
    while (simulation.is_running()):
        with tc.count('construct_map'):
            construct_map()
        with tc.count('update_plan'):
            update_plan()
    tc.write_npz("time_counts.npz")

    This will record the time taken for each construct_map and update_plan step.
    The runtimes for these steps are each saved to an individual array,
    and all arrays are written to time_counts.npz

    """

    def __init__(self):
        self.timings = defaultdict(list)

    class _Timer:
        def __init__(self, key, timings):
            self.key = key
            self.timings = timings

        def __enter__(self):
            self.start_time = time.time()

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.start_time
            self.timings[self.key].append(elapsed)

    def count(self, key):
        return self._Timer(key, self.timings)

    def write_npz(self, filename):
        """Write recorded timings to a .npz file"""
        npz_data = {key: np.array(times) for key, times in self.timings.items()}
        np.savez(filename, **npz_data)


class StubTimeCounter(TimeCounter):
    """Stubbed out version that does nothing, for backwards compatibility."""

    class _StubTimer:
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def __init__(self):
        # Optional: still call super().__init__ if you want to keep the API consistent
        super().__init__()
        self.timings.clear()

    def count(self, key):
        return self._StubTimer()

    def write_npz(self, filename):
        pass
