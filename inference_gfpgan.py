#!/usr/bin/env python
"""Backward-compatible entry point for the packaged GFPGAN CLI."""

from gfpgan.cli import imread, imwrite, main

__all__ = ['imread', 'imwrite', 'main']


if __name__ == '__main__':
    raise SystemExit(main())
