from time import perf_counter
from monai import transforms
import numpy as np

__all__ = ["class_timer"]


def class_timer(cls, verbose=False):
    class WrappedClass(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.name = cls.__name__
            self.run_times = []
            self.run_counts = 0
            self.verbose = verbose

        def __call__(self, *args, **kwargs):
            self.run_counts += 1

            t_1 = perf_counter()
            output = super().__call__(*args, **kwargs)
            t_2 = perf_counter()

            if self.verbose:
                print(f"{self.name}: {t_2 - t_1:.5f}s")

            self.run_times.append(t_2 - t_1)
            return output

        def get_stats(self):
            if self.run_counts == 0:
                return np.nan, np.nan

            return np.mean(self.run_times), np.std(self.run_times)

        def print_stats(self):
            avg, std = self.get_stats()
            print(f"{self.name}: Runs: {self.run_counts}, Avg: {avg:.5f}s, Std: {std:.5f}s")

    return WrappedClass

class Identity(transforms.MapTransform):
    def __init__(self, keys: list[str], allow_missing_keys=False) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: dict) -> dict:
        for key in self.keys:
            if key in data:
                pass
            else:
                raise ValueError(f"Key '{key}' is not in data")
        return data
