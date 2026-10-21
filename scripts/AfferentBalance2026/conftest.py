"""Exclude immutable historical tests and generated runs from live collection."""
collect_ignore = ["archive", "report"]
collect_ignore_glob = ["experiments/*/results"]
