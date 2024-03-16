from time import perf_counter, sleep
from monai import transforms
import numpy as np

__all__ = ["transform_timer"]

def transform_timer(cls):
    class WrappedClass(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cls_name = cls.__name__

        def __call__(self, *args, **kwargs):
            t_1 = perf_counter()
            output = super().__call__(*args, **kwargs)
            t_2 = perf_counter()
            print(f"{self.cls_name}: {t_2 - t_1:.5f}s")
            return output

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


def main():
    data = {
        "image": np.random.randn(3,3,3),
        "label": np.random.randint(0, 2, (3,3,3))
    }

    print(data)
    IdentityTimed = transform_timer(Identity)
    transform = IdentityTimed(keys=["image", "label"])
    data = transform(data)
    print(data)


if __name__ == '__main__':
    main()
