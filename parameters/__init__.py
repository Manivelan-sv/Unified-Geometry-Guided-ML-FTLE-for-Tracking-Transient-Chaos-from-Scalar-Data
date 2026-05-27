"""
Dynamic parameter loader for the Poincaré / FTLE analysis pipeline.

Public API
----------
load_dataset_params(data_id)  → module
    Import/reload a single parameters/data_N.py.

discover_all_params(params_dir=None)  → {data_id: module}
    Find EVERY data_N.py in the parameters/ directory and load all of them.
    Returns an OrderedDict sorted by the numeric suffix.

Usage
-----
    from parameters import discover_all_params
    for data_id, P in discover_all_params().items():
        WINDOW = P.T_WINDOW_STEPS
        ...
"""
import importlib
import re
from pathlib import Path
from collections import OrderedDict

_PATTERN = re.compile(r'^data_\d+$')


def load_dataset_params(data_id):
    """Import and reload parameters/<data_id>.py.  Returns the module."""
    if isinstance(data_id, Path):
        data_id = data_id.name
    data_id = str(data_id)
    if not _PATTERN.match(data_id):
        raise ValueError(
            f"data_id must match ^data_\\d+$ (e.g. 'data_2', 'data_4'), "
            f"got {data_id!r}."
        )
    mod = importlib.import_module(f"parameters.{data_id}")
    importlib.reload(mod)
    return mod


def discover_all_params(params_dir=None):
    """
    Scan the parameters/ package directory for every ``data_N.py`` file,
    import each one, and return an ordered mapping.

    Parameters
    ----------
    params_dir : str or Path, optional
        Directory to scan.  Defaults to the ``parameters/`` package folder
        (i.e. the directory containing this ``__init__.py``).

    Returns
    -------
    OrderedDict[str, module]
        Keys are data-ids (``"data_2"``, ``"data_4"`` …) sorted by their
        numeric suffix; values are the freshly-reloaded parameter modules.

    Example
    -------
    >>> from parameters import discover_all_params
    >>> for data_id, P in discover_all_params().items():
    ...     print(data_id, P.T_WINDOW_STEPS)
    """
    if params_dir is None:
        params_dir = Path(__file__).parent

    stems = sorted(
        [p.stem for p in Path(params_dir).iterdir()
         if p.suffix == ".py" and _PATTERN.match(p.stem)],
        key=lambda s: int(s.split("_")[-1]),
    )

    if not stems:
        print(f"⚠  No data_N.py files found in {params_dir}")
        return OrderedDict()

    result = OrderedDict()
    print(f"Discovered {len(stems)} parameter file(s) in {params_dir}:")
    for data_id in stems:
        mod = importlib.import_module(f"parameters.{data_id}")
        importlib.reload(mod)
        result[data_id] = mod
        print(f"  ✓  {data_id}.py  →  OUT_DIR={getattr(mod,'OUT_DIR','?')}")

    return result
