from dataclasses import dataclass
from typing import Any, TypeVar, Type, cast


T = TypeVar("T")

from typing import TypeAlias

ResizeType = None | float | list[int | None] | tuple[int | None, int | None] # no resize | ratio | [height, width]


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def to_float(x: Any) -> float:
    assert isinstance(x, (int, float))
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()

def from_resize(x: Any) -> ResizeType:
    # resize ratio
    if isinstance(x, (float, int)) and not isinstance(x, bool):
        assert x > 0, "Resize ratio must be > 0"
        return float(x)
    # [height, width] parsing, either can be null
    if isinstance(x, (list, tuple)):
        assert len(x) == 2, "Resize must have two elements (height, width)"
        def _check_dim(val):
            assert val is None or (isinstance(val, int) and not isinstance(val, bool) and val > 0), \
                f"Resize dimension must be a positive int or null, got {x!r}"
            return val
        height, width = _check_dim(x[0]), _check_dim(x[1])
        assert height is not None or width is not None, \
            "Resize must have at least one non-null dimension"
        return [height, width]
    # null = no resizing
    assert x is None, f"Resize must be None, a int/float, or (height, width), got {x!r}"
    return None

def to_resize(x: ResizeType):
    # same as from_resize but do not convert a possible int to float
    if isinstance(x, (float, int)):
        return x
    else:
        return from_resize(x)


@dataclass
class Data:
    dataset_name: str
    batch_size: int
    vocab_name: str
    num_workers: int
    resize: ResizeType
    transcript_format: str

    @staticmethod
    def from_dict(obj: Any) -> 'Data':
        assert isinstance(obj, dict)
        dataset_name = from_str(obj.get("dataset_name"))
        batch_size = from_int(obj.get("batch_size"))
        vocab_name = from_str(obj.get("vocab_name"))
        num_workers = from_int(obj.get("num_workers"))
        assert "resize" in obj or "reduce_ratio" in obj, \
            'Config field "resize" is required (use null for no resize, or legacy "reduce_ratio")'
        resize = from_resize(obj.get("resize", obj.get("reduce_ratio")))
        transcript_format = from_str(obj.get("transcript_format"))
        return Data(dataset_name, batch_size, vocab_name, num_workers, resize, transcript_format)

    def to_dict(self) -> dict:
        result: dict = {}
        result["dataset_name"] = from_str(self.dataset_name)
        result["batch_size"] = from_int(self.batch_size)
        result["vocab_name"] = from_str(self.vocab_name)
        result["num_workers"] = from_int(self.num_workers)
        result["resize"] = to_resize(self.resize)
        result["transcript_format"] = from_str(self.transcript_format)
        return result


@dataclass
class ExperimentConfig:
    data: Data

    @staticmethod
    def from_dict(obj: Any) -> 'ExperimentConfig':
        assert isinstance(obj, dict)
        data = Data.from_dict(obj.get("data"))
        return ExperimentConfig(data)

    def to_dict(self) -> dict:
        result: dict = {}
        result["data"] = to_class(Data, self.data)
        return result


def experiment_config_from_dict(s: Any) -> ExperimentConfig:
    return ExperimentConfig.from_dict(s)


def experiment_config_to_dict(x: ExperimentConfig) -> Any:
    return to_class(ExperimentConfig, x)