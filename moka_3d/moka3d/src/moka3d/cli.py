#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:37:45 2026

@author: cosimo
"""

from __future__ import annotations

from pathlib import Path
import shutil
import typer

from .config import load_config, validate_config
from .pipeline import run_pipeline
from .defaults import DEFAULT_CONFIG_YAML

app = typer.Typer(help="MOKA3D command-line interface")


@app.command("init-config")
def init_config(output: Path = typer.Argument(..., help="Path where the example YAML config will be written")):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
    typer.echo(f"Example config written to: {output}")


@app.command("validate")
def validate(config_path: Path = typer.Argument(..., exists=True, help="YAML configuration file")):
    cfg = load_config(config_path)
    validate_config(cfg)
    typer.echo("Configuration is valid.")


@app.command("run")
def run(config_path: Path = typer.Argument(..., exists=True, help="YAML configuration file")):
    cfg = load_config(config_path)
    run_pipeline(cfg, config_path=config_path)
    typer.echo("-------------- Run completed successfully --------------")


def main():
    app()


if __name__ == "__main__":
    main()