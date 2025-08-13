import click
from anyio import Path
from dotenv import load_dotenv
from fabric import Connection

from sal.const import PROBE_DATA_INPUT_ROOT, PROBE_OUTPUT_ROOT
from sal.utils.env import get_env_or_throw

load_dotenv()


@click.group()
def cli():
    """Remote job management tool."""
    pass


@cli.command("probe")
@click.option(
    "--host",
    required=True,
    help="Remote host to sync with",
)
@click.option(
    "--direction",
    type=click.Choice(["push", "pull"]),
    required=True,
    help="Direction of sync: 'push' to upload probe_data to remote, 'pull' to download output from remote",
)
def probe(direction: str, host: str) -> None:
    """Sync probe data and output between local and remote systems."""

    remote_root = Path(get_env_or_throw("REMOTE_ROOT"))
    user = get_env_or_throw("USERNAME")

    # Build connection string
    conn_host = f"{user}@{host}" if user else host

    try:
        with Connection(host) as conn:
            if direction == "push":
                # Sync local probe_data to remote
                local_path = PROBE_DATA_INPUT_ROOT
                remote_path = Path(remote_root) / PROBE_DATA_INPUT_ROOT.name

                conn.run(f"mkdir -p {remote_path}")

                click.echo(
                    f"Syncing probe data from {local_path} to {conn_host}:{remote_path}"
                )

                conn.local(
                    f"rsync -avz --progress {local_path}/ {conn_host}:{remote_path}/"
                )

            elif direction == "pull":
                # Sync remote output to local
                remote_path = f"{remote_root}/output/probes/"
                local_path = str(PROBE_OUTPUT_ROOT) + "/"

                click.echo(
                    f"Syncing output from {conn_host}:{remote_path} to {local_path}"
                )

                conn.local(
                    f"rsync -avz --progress {conn_host}:{remote_path} {local_path}"
                )

    except Exception as e:
        raise click.ClickException(f"Sync failed: {e}")

    click.secho("✔ Sync completed successfully", fg="green")


if __name__ == "__main__":
    cli()
