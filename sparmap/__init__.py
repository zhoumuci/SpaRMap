"""SpaRMap: spatial representation learning and gene-set mapping."""

__all__ = ["sparmap"]


def __getattr__(name):
    if name == "sparmap":
        from .runner import sparmap

        globals()[name] = sparmap
        return sparmap
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
