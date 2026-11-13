"""Build package metadata from the project README."""

from pathlib import Path

from setuptools import setup


readme = Path(__file__).with_name("README.md").read_text(encoding="utf-8")
repository = "https://github.com/zhoumuci/SpaRMap/blob/main"
links = {
    "docs/assets/case1.svg": (
        "https://raw.githubusercontent.com/zhoumuci/SpaRMap/main/docs/assets/case1.svg"
    ),
    "pyproject.toml": f"{repository}/pyproject.toml",
    "docs/tutorial/tutorial.md": f"{repository}/docs/tutorial/tutorial.md",
    "LICENSE": f"{repository}/LICENSE",
}
for relative, absolute in links.items():
    readme = readme.replace(f"]({relative})", f"]({absolute})")

setup(long_description=readme, long_description_content_type="text/markdown")
