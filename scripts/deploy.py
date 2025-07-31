import logging
import subprocess
from dataclasses import dataclass, field
from io import StringIO
from time import sleep
from typing import Optional, Set

import click
from dotenv import load_dotenv
from fabric import Connection
from rich.console import Console

from sal.utils.env import get_env_or_throw

load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger("fabric").setLevel(logging.WARNING)
logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.getLogger("invoke").setLevel(logging.WARNING)

console = Console()


@dataclass
class RemoteConfig:
    username: str
    hostname: str
    remote_root: str
    uv_path: str = "/homes/km1124/.local/bin/uv"
    # remotes_to_exclude: Set[str] = field(
    #     default_factory=lambda: {
    #         "gpuvm21",
    #         "gpuvm22",
    #     }
    # )
    remotes_to_exclude: Set[str] = field(default_factory=lambda: set())

    def get_remotes_to_exclude(self) -> str:
        return ",".join(self.remotes_to_exclude)


@dataclass
class SlurmJobConfig:
    nodes: int = 1
    ntasks: int = 1
    partition: str = "gpgpuB"  # Default Tesla A30 24GB
    job_name: str = "qtts"
    output: str = "./logs/slurm-%j.log"
    cpus_per_task: int = 1
    gres: str = "gpu:1"


@dataclass
class RunConfig:
    "Config params passed to a run instance"

    wandb_api_key: str
    github_token: str
    commit_hash: str


@dataclass
class DeployConfig:
    run_config: RunConfig
    remote_config: RemoteConfig
    slurm_config: SlurmJobConfig = field(default_factory=SlurmJobConfig)


def get_base_config() -> tuple[RemoteConfig, RunConfig]:
    """Get base configuration from environment variables."""
    with console.status("[yellow]Validating env variables...", spinner="dots"):
        remote_config = RemoteConfig(
            username=get_env_or_throw("USERNAME"),
            hostname=get_env_or_throw("HOSTNAME"),
            remote_root=get_env_or_throw("REMOTE_ROOT"),
        )
        run_config = RunConfig(
            wandb_api_key=get_env_or_throw("WANDB_API_KEY"),
            github_token=get_env_or_throw("GITHUB_TOKEN"),
            commit_hash="",  # Will be set by commands that need it
        )
    console.print("[green]✔ Env variables validated")
    return remote_config, run_config


@click.group()
def cli():
    """Remote job management tool."""
    pass


PARTITION_MAP = {
    "A40": "gpgpu",  # Tesla A40 48GB (7)
    "A30": "gpgpuB",  # Tesla A30 24GB (20)
    "T4": "gpgpuC",  # Tesla T4 16GB GPUs (16)
    "T4D": "gpgpuD",  # Tesla T4 16GB GPUs (26)
    "A100": "AMD7-A100-T",  # Tesla A100 80GB (28)
    "A16": "a16gpu",  # Tesla A16 16GB GPUs (28)
}


@cli.command()
@click.option(
    "--partition",
    default="A100",
    help="SLURM partition to use (default: A100)",
    type=click.Choice(PARTITION_MAP.keys()),
)
@click.option("--commit-hash", required=False, help="Hash of commit to submit as a job")
@click.option("--tail/--no-tail", default=True, help="Whether to tail the output log")
def submit(
    commit_hash: str,
    partition: str,
    tail: bool,
):
    """Submit a new SLURM job."""
    remote_config, run_config = get_base_config()
    run_config.commit_hash = commit_hash if commit_hash else _get_latest_commit_hash()

    _prompt_commit_info(run_config.commit_hash)

    slurm_config = SlurmJobConfig(partition=PARTITION_MAP[partition])

    config = DeployConfig(
        run_config=run_config,
        remote_config=remote_config,
        slurm_config=slurm_config,
    )

    submit_job(config, tail_output=tail)


def _prompt_commit_info(commit_hash: str) -> None:
    commit_info = (
        subprocess.check_output(
            [
                "git",
                "log",
                "-1",
                "--pretty=format:%cd - %s",
                "--date=format:%Y-%m-%d %H:%M:%S",
                commit_hash,
            ],
            stderr=subprocess.STDOUT,
        )
        .strip()
        .decode("utf-8")
    )

    console.print(f"Commit info: [green]{commit_info}[/green]")

    user_confirmation = input("Do you want to continue with this commit? (yes/no): ")
    if user_confirmation.lower() in ["no", "n"]:
        raise SystemExit("Operation cancelled by the user.")


def _get_latest_commit_hash() -> str:
    console.print(
        "No commit hash was provided. The latest commit hash will be used by default."
    )
    try:
        latest_commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
            )
            .strip()
            .decode("utf-8")
        )
    except subprocess.CalledProcessError as e:
        console.print(
            f"Error retrieving the latest commit hash: {e.output.decode('utf-8')}"
        )
        raise SystemExit("Failed to retrieve the latest commit hash.")

    return latest_commit_hash


@cli.command()
@click.option("--job-id", help="Job ID to cancel")
def cancel(job_id: Optional[str]):
    """Cancel a SLURM job."""
    remote_config, _ = get_base_config()

    if not job_id:
        job_id = click.prompt("Enter the job ID")

    cancel_job(job_id, remote_config)


@cli.command("push-files")
def push_files_cmd():
    """Push files to remote server."""
    remote_config, _ = get_base_config()
    c = Connection(remote_config.hostname)
    push_files(c, remote_config)


@cli.command("configure-environment")
def configure_environment_cmd():
    """Configure the remote environment."""
    remote_config, _ = get_base_config()
    c = Connection(remote_config.hostname)
    push_files(c, remote_config)
    configure_environment(c, remote_config)


def push_files(connection, config: RemoteConfig) -> None:
    paths_to_transfer = [
        "src",
        "experiments",
        "job.sh",
        ".env",
        "pyproject.toml",
    ]
    with console.status("[yellow]Pushing files to remote...", spinner="dots"):
        connection.local(
            f"rsync -avz --exclude='__pycache__' --exclude='*.pyc' {' '.join(paths_to_transfer)} km1124@{config.hostname}:{config.remote_root}",
            hide=True,
        )
    console.print(f"[green]✔ Pushed files to remote: {config.hostname}")


def configure_environment(connection, config: RemoteConfig) -> None:
    with console.status("[yellow]Configuring environment...", spinner="dots"):
        with connection.cd(config.remote_root):
            connection.run(f"{config.uv_path} sync --group deploy")
            connection.run(f'{config.uv_path} pip install -e "."')
    console.print("[green]✔ Configured environment")


def write_jobscript(connection, config: DeployConfig) -> None:
    sc = config.slurm_config
    with console.status("[yellow]Writing job script to remote...", spinner="dots"):
        job_script_content = f"""#!/bin/bash
#SBATCH --nodes={sc.nodes}
#SBATCH --ntasks={sc.ntasks}
#SBATCH --partition={sc.partition}
#SBATCH --job-name={sc.job_name}
#SBATCH --output={sc.output}
#SBATCH --cpus-per-task={sc.cpus_per_task}
#SBATCH --gres={sc.gres}

# Exit on any error
set -e

echo "Launching jobscript"
source /vol/cuda/12.0.0/setup.sh
source .venv/bin/activate

echo "Fetching commit with hash $COMMIT_HASH"
git reset --hard HEAD

git remote set-url origin https://$GITHUB_TOKEN@github.com/KuSi833/search-and-learn.git
# Fetch with error handling
echo "Fetching latest changes..."
if ! git fetch; then
    echo "ERROR: Failed to fetch from remote repository"
    echo "This might be due to SSH key issues or network problems"
    exit 1
fi

# Checkout specific commit with error handling
echo "Checking out commit $COMMIT_HASH..."
if ! git checkout $COMMIT_HASH; then
    echo "ERROR: Failed to checkout commit $COMMIT_HASH"
    echo "Commit may not exist or may not be fetched"
    exit 1
fi

echo "Successfully checked out commit $COMMIT_HASH"

export UV_PYTHON_INSTALL_DIR="/vol/bitbucket/km1124/.cache/python"
export HF_HOME="/vol/bitbucket/km1124/.cache/huggingface"
export UV_CACHE_DIR="/vol/bitbucket/km1124/.cache/uv"

# Set VLLM profiling and logging configuration
export VLLM_TORCH_PROFILER_DIR="./trace"
# export VLLM_LOGGING_LEVEL="DEBUG"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export NCCL_P2P_DISABLE=1

echo "Running main file"
python experiments/qwen_math.py
"""
        connection.put(
            StringIO(job_script_content), f"{config.remote_config.remote_root}/job.sh"
        )
    console.print(
        f"[green]✔ Job script with commit hash {config.run_config.commit_hash} written to remote"
    )


def submit_job(config: DeployConfig, tail_output=True):
    with console.status("[yellow]Connecting to remote...", spinner="dots"):
        c = Connection(config.remote_config.hostname)
    console.print(f"︎[green]✔︎ Connected to {config.remote_config.hostname}")

    write_jobscript(c, config)

    with c.cd(config.remote_config.remote_root):
        with console.status("[yellow]Submitting job to SLURM...", spinner="dots"):
            console.print(
                f" - [blue]Excluded hosts: {config.remote_config.get_remotes_to_exclude()}"
            )
            result = c.run(
                f"sbatch --exclude={config.remote_config.get_remotes_to_exclude()} "
                "--export="
                f"WANDB_API_KEY='{config.run_config.wandb_api_key}',"
                f"GITHUB_TOKEN='{config.run_config.github_token}',"
                f"COMMIT_HASH='{config.run_config.commit_hash}' "
                "job.sh"
            )
            job_id = result.stdout.strip().split()[-1]
            console.print(f"[green]✔︎ Job submitted with ID: {job_id}")

        if tail_output:
            log_file = f"./logs/slurm-{job_id}.log"
            with console.status("[yellow]Waiting for job to start...", spinner="dots"):
                while True:
                    result = c.run(
                        f"test -f {log_file} && echo 'exists' || echo 'not found'",
                        hide=True,
                    )
                    if "exists" in result.stdout:
                        break
                    sleep(2)
            console.print("[green]✔ Job started. Tailing log file...")
            c.run(f"tail -f {log_file}")


def cancel_job(job_id, config: RemoteConfig):
    logger = logging.getLogger(__name__)
    c = Connection(config.hostname)

    logger.info(f"Cancelling job with ID: {job_id}")
    with c.cd(config.remote_root):
        c.run(f"scancel {job_id}")
        console.print(f"[green]Job {job_id} cancellation request sent")


if __name__ == "__main__":
    cli()
