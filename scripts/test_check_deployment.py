import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_deployment


def test_normalize_url():
    assert check_deployment.normalize_url(
        "https://backend.onrender.com/"
    ) == "https://backend.onrender.com"


def test_parse_args_requires_urls():
    args = check_deployment.parse_args([
        "--backend",
        "https://backend.onrender.com",
        "--frontend",
        "https://frontend.onrender.com/",
    ])
    assert args.backend == "https://backend.onrender.com"
    assert args.frontend == "https://frontend.onrender.com"


def test_normalize_url_rejects_path():
    with pytest.raises(check_deployment.argparse.ArgumentTypeError):
        check_deployment.normalize_url("https://backend.onrender.com/app")
