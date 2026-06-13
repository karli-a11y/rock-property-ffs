"""
adapters/ — Plugin-style database ingestion layer (P1.1)

Each adapter wraps one source database and exposes a consistent interface:
    adapter.load() -> (DataFrame, mapping_log_DataFrame)

To add a new database:
1. Create adapters/<key>.py implementing BaseAdapter
2. Register the database in config/databases.yaml
3. No modification of the scientific core is required.
"""

from .base import BaseAdapter

__all__ = ["BaseAdapter"]
