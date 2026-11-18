"""Point d'entrée en ligne de commande : `qarniya --help`.

Les commandes (train, evaluate, predict, serve) seront ajoutées aux étapes suivantes.
"""

import typer

from qarniya import __version__

app = typer.Typer(help="Qarniya AI — dépistage précoce du kératocône.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Qarniya AI — dépistage précoce du kératocône."""


@app.command()
def version() -> None:
    """Affiche la version installée."""
    typer.echo(f"qarniya {__version__}")


if __name__ == "__main__":
    app()
