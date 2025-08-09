import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from time import sleep
from typing import Optional, Set

import click
from anyio import Path
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
class RunConfig:
    "Config params passed to a run instance"

    wandb_api_key: str
    github_token: str
    commit_hash: str
    file_path: str = "experiments/qwen_math.py"  # path to file to run from project root
    # file_path: str = "experiments/expand.py"


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
class PbsJobConfig:
    ncpus: int = 4
    memory: str = "24gb"
    gpu_type: str = "L40S"  # Default to L40S GPU type
    walltime: str = "05:00:00"  # HH:MM:SS format
    queue: str = "v1_gpu72"
    job_name: str = "qtts"
    output: str = "./logs/log.log"


@dataclass
class DeployConfig:
    run_config: RunConfig
    remote_config: RemoteConfig
    slurm_config: SlurmJobConfig = field(default_factory=SlurmJobConfig)
    pbs_config: PbsJobConfig = field(default_factory=PbsJobConfig)


def get_slurm_job_script(config: DeployConfig) -> str:
    """Generate complete SLURM job script."""
    return f"""#!/bin/bash
#SBATCH --nodes={config.slurm_config.nodes}
#SBATCH --ntasks={config.slurm_config.ntasks}
#SBATCH --partition={config.slurm_config.partition}
#SBATCH --job-name={config.slurm_config.job_name}
#SBATCH --output={config.slurm_config.output}
#SBATCH --cpus-per-task={config.slurm_config.cpus_per_task}
#SBATCH --gres={config.slurm_config.gres}

# Exit on any error
set -e

echo "Launching SLURM jobscript"
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
# python {config.run_config.file_path}
python experiments/expand.py
"""


def get_pbs_job_script(config: DeployConfig) -> str:
    """Generate complete PBS job script."""
    # Build select statement for 1 node, 1 GPU
    select_statement = f"select=1:ncpus={config.pbs_config.ncpus}:mem={config.pbs_config.memory}:ngpus=1"
    if config.pbs_config.gpu_type != "any":
        select_statement += f":gpu_type={config.pbs_config.gpu_type}"

    return f"""#!/bin/bash
#PBS -l {select_statement}
#PBS -l walltime={config.pbs_config.walltime}
#PBS -q {config.pbs_config.queue}
#PBS -N {config.pbs_config.job_name}
#PBS -o {config.pbs_config.output}
#PBS -j oe

# Change to submission directory
cd $PBS_O_WORKDIR

# Exit on any error
set -e

echo "Launching PBS jobscript"
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

export CUDA_VISIBLE_DEVICES=0
# Set VLLM profiling and logging configuration
export VLLM_TORCH_PROFILER_DIR="./trace"
# export VLLM_LOGGING_LEVEL="DEBUG"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export NCCL_P2P_DISABLE=1

echo "Running main file"
python {config.run_config.file_path}
"""


def write_slurm_jobscript(connection, config: DeployConfig) -> None:
    with console.status(
        "[yellow]Writing SLURM job script to remote...", spinner="dots"
    ):
        job_script_content = get_slurm_job_script(config)
        connection.put(
            StringIO(job_script_content),
            f"{config.remote_config.remote_root}/slurm_job.sh",
        )
    console.print(
        f"[green]✔ SLURM job script with commit hash {config.run_config.commit_hash} written to remote"
    )


def write_pbs_jobscript(connection, config: DeployConfig) -> None:
    with console.status("[yellow]Writing PBS job script to remote...", spinner="dots"):
        job_script_content = get_pbs_job_script(config)
        connection.put(
            StringIO(job_script_content),
            f"{config.remote_config.remote_root}/pbs_job.sh",
        )
    console.print(
        f"[green]✔ PBS job script with commit hash {config.run_config.commit_hash} written to remote"
    )


def get_hostname_remote_config(hostname: str) -> RemoteConfig:
    with console.status(
        "[yellow]Validating SLURM remote env variables...", spinner="dots"
    ):
        remote_config = RemoteConfig(
            username=get_env_or_throw("USERNAME"),
            hostname=hostname,
            remote_root=get_env_or_throw("REMOTE_ROOT"),
        )
    console.print("[green]✔ SLURM env variables validated")
    return remote_config


def get_slurm_remote_config() -> RemoteConfig:
    """Get SLURM configuration from environment variables."""
    with console.status(
        "[yellow]Validating SLURM remote env variables...", spinner="dots"
    ):
        remote_config = RemoteConfig(
            username=get_env_or_throw("USERNAME"),
            hostname=get_env_or_throw("HOSTNAME"),
            remote_root=get_env_or_throw("REMOTE_ROOT"),
        )
    console.print("[green]✔ SLURM env variables validated")
    return remote_config


def get_pbs_remote_config() -> RemoteConfig:
    """Get PBS configuration from environment variables."""
    with console.status(
        "[yellow]Validating PBS remote env variables...", spinner="dots"
    ):
        remote_config = RemoteConfig(
            username=get_env_or_throw("USERNAME"),
            hostname=get_env_or_throw("HOSTNAME_PBS"),
            remote_root=get_env_or_throw("REMOTE_ROOT_PBS"),
        )
    console.print("[green]✔ PBS env variables validated")
    return remote_config


def get_run_config(commit_hash: Optional[str] = None) -> RunConfig:
    """Get run configuration from environment variables."""
    with console.status("[yellow]Validating run env variables...", spinner="dots"):
        run_config = RunConfig(
            wandb_api_key=get_env_or_throw("WANDB_API_KEY"),
            github_token=get_env_or_throw("GITHUB_TOKEN"),
            commit_hash=commit_hash if commit_hash else _get_latest_commit_hash(),
        )
    console.print("[green]✔ Env variables validated")

    _prompt_commit_info(run_config.commit_hash)

    return run_config


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


@cli.command("submit-slurm")
@click.option(
    "--partition",
    default="A100",
    help="SLURM partition to use (default: A100)",
    type=click.Choice(PARTITION_MAP.keys()),
)
@click.option("--commit-hash", required=False, help="Hash of commit to submit as a job")
@click.option("--tail/--no-tail", default=True, help="Whether to tail the output log")
def submit_slurm(
    commit_hash: str,
    partition: str,
    tail: bool,
):
    """Submit a new job to SLURM scheduler."""
    remote_config = get_slurm_remote_config()
    run_config = get_run_config(commit_hash)

    slurm_config = SlurmJobConfig(partition=PARTITION_MAP[partition])
    config = DeployConfig(
        run_config=run_config,
        remote_config=remote_config,
        slurm_config=slurm_config,
    )
    submit_slurm_job(config, tail_output=tail)


HOSTNAMES = set(
    [
        "gpuvm21",
        "gpuvm22",
        "merlin",
        "linnet",
    ]
)


@cli.command("submit-remote")
@click.option(
    "--hostname",
    required=True,
    help="Hostname of remote to run the script on",
    type=click.Choice(HOSTNAMES),
)
@click.option("--commit-hash", required=False, help="Hash of commit to submit as a job")
@click.option(
    "--script-path",
    required=True,
    help="Relative path to python executable from project root",
)
@click.option("--tail/--no-tail", default=True, help="Whether to tail the output log")
def submit_remote(
    hostname: str,
    commit_hash: str,
    script_path: str,
    tail: bool,
):
    """Run a script on a remote scheduler."""
    remote_config = get_hostname_remote_config(hostname)
    run_config = get_run_config(commit_hash)

    config = DeployConfig(
        run_config=run_config,
        remote_config=remote_config,
    )
    run_on_remote(config, script_path, tail)


@cli.command("setup-remote")
@click.option(
    "--hostname",
    required=True,
    help="Hostname of remote to run the script on",
    type=click.Choice(HOSTNAMES),
)
@click.option("--commit-hash", required=False, help="Hash of commit to submit as a job")
def setup_remote(
    hostname: str,
    commit_hash: Optional[str],
):
    """Set up remote with repo"""
    remote_config = get_hostname_remote_config(hostname)
    github_token = get_env_or_throw("GITHUB_TOKEN")

    commit_hash = commit_hash if commit_hash else _get_latest_commit_hash()
    _prompt_commit_info(commit_hash)

    with console.status("[yellow]Connecting to remote...", spinner="dots"):
        c = Connection(remote_config.hostname)
    console.print(f"︎[green]✔︎ Connected to {remote_config.hostname}")

    remote_root = Path(remote_config.remote_root)
    with console.status("[yellow]Setting up remote directory...", spinner="dots"):
        # Create directory if it doesn't exist
        c.run(f"mkdir -p {remote_root}", hide=True)

    with console.status("[yellow]Cloning repository...", spinner="dots"):
        # Check if repository already exists
        result = c.run(f"test -d {remote_root}/.git", warn=True, hide=True)
        if result.failed:
            # Repository doesn't exist, clone it
            c.run(
                f"git clone git@github.com:KuSi833/search-and-learn.git {remote_root}"
            )
            console.print(f"[green]✔ Repository cloned to {remote_root}")
        else:
            console.print(f"[green]✔ Repository already exists at {remote_root}")

    with c.cd(remote_root):
        _checkout_commit(c, github_token, commit_hash)

        with c.prefix(
            'export UV_PYTHON_INSTALL_DIR="/vol/bitbucket/km1124/.cache/python"; '
            'export HF_HOME="/vol/bitbucket/km1124/.cache/huggingface"; '
            'export UV_CACHE_DIR="/vol/bitbucket/km1124/.cache/uv"; '
            'export UV_LINK_MODE="copy"'
        ):
            # BUG: I think this will never give back control even when the setup is done
            # It has to be cancelled, and when run again will proceed
            with console.status("[yellow]Setting up .venv with uv...", spinner="dots"):
                c.run(f"{remote_config.uv_path} sync --group deploy", hide=False)
            console.print("[green]✔ Virtual environment setup completed")

    with console.status("[yellow]Syncing models directory...", spinner="dots"):
        c.run(
            f"rsync -avP /vol/bitbucket/km1124/search-and-learn/models/ {remote_root}/models/"
        )
    console.print("[green]✔ Models directory sync completed")


def _checkout_commit(
    connection: Connection, github_token: str, commit_hash: str
) -> bool:
    with console.status("[yellow]Fetching and checking out commit...", spinner="dots"):
        """Checkout specific commit on remote. Returns True if successful, False otherwise."""
        connection.run("git reset --hard HEAD", hide=True)
        connection.run(
            f"git remote set-url origin https://{github_token}@github.com/KuSi833/search-and-learn.git",
            hide=True,
        )

        # Fetch with error handling
        fetch_result = connection.run("git fetch", warn=True, hide=True)
        if fetch_result.failed:
            console.print("[red]ERROR: Failed to fetch from remote repository")
            console.print(
                "[red]This might be due to SSH key issues or network problems"
            )
            return False

        # Checkout specific commit with error handling
        checkout_result = connection.run(
            f"git checkout {commit_hash}", warn=True, hide=True
        )
        if checkout_result.failed:
            console.print(f"[red]ERROR: Failed to checkout commit {commit_hash}")
            console.print("[red]Commit may not exist or may not be fetched")
            return False

        console.print(f"[green]✔︎ Successfully checked out commit {commit_hash}")
        return True


def run_on_remote(config: DeployConfig, executable: str, tail: bool) -> None:
    with console.status("[yellow]Connecting to remote...", spinner="dots"):
        c = Connection(config.remote_config.hostname)
    console.print(f"︎[green]✔︎ Connected to {config.remote_config.hostname}")

    log_file = f"./logs/{config.remote_config.hostname}_run.log"

    with c.cd(config.remote_config.remote_root):
        if not _checkout_commit(
            c, config.run_config.github_token, config.run_config.commit_hash
        ):
            return

        with console.status("[yellow]Running executable on remote...", spinner="dots"):
            c.run(
                f"source ~/.bashrc && "
                f"export HF_HOME='/vol/bitbucket/km1124/.cache/huggingface' && "
                f"nohup env WANDB_API_KEY='{config.run_config.wandb_api_key}' "
                f"GITHUB_TOKEN='{config.run_config.github_token}' "
                f"./.venv/bin/python {executable} > {log_file} 2>&1 &",
                asynchronous=True,
                hide=True,
            )
            console.print("[green]✔︎ Executable started in background...")

        if tail:
            console.print(f"[yellow]Tailing log file {log_file}...")
            c.run(f"tail -f {log_file}")


@cli.command("submit-pbs")
@click.option("--commit-hash", required=False, help="Hash of commit to submit as a job")
@click.option("--tail/--no-tail", default=True, help="Whether to tail the output log")
@click.option(
    "--gpu-type",
    default="L40S",
    help="GPU type to request (default: L40S). Leave empty for any GPU type.",
)
def submit_pbs(
    commit_hash: str,
    tail: bool,
    gpu_type: str,
):
    """Submit a new job to PBS scheduler."""
    remote_config = get_pbs_remote_config()
    run_config = get_run_config(commit_hash)

    pbs_config = PbsJobConfig(gpu_type=gpu_type)
    config = DeployConfig(
        run_config=run_config,
        remote_config=remote_config,
        pbs_config=pbs_config,
    )
    submit_pbs_job(config, tail_output=tail)


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


# @cli.command()
# @click.option("--job-id", help="Job ID to cancel")
# def cancel(job_id: Optional[str]):
#     """Cancel a SLURM job."""
#     remote_config = get_remote_config()

#     if not job_id:
#         job_id = click.prompt("Enter the job ID")

#     cancel_job(job_id, remote_config)


# @cli.command("push-files")
# def push_files_cmd():
#     """Push files to remote server."""
#     remote_config = get_remote_config()
#     c = Connection(remote_config.hostname)
#     push_files(c, remote_config)


# @cli.command("configure-environment")
# def configure_environment_cmd():
#     """Configure the remote environment."""
#     remote_config = get_remote_config()
#     c = Connection(remote_config.hostname)
#     push_files(c, remote_config)
#     configure_environment(c, remote_config)


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


def submit_slurm_job(config: DeployConfig, tail_output=True):
    with console.status("[yellow]Connecting to remote...", spinner="dots"):
        c = Connection(config.remote_config.hostname)
    console.print(f"︎[green]✔︎ Connected to {config.remote_config.hostname}")

    # Write the SLURM job script
    write_slurm_jobscript(c, config)

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
                "slurm_job.sh"
            )
            job_id = result.stdout.strip().split()[-1]
            console.print(f"[green]✔︎ SLURM job submitted with ID: {job_id}")
            log_file = f"./logs/slurm-{job_id}.log"

        if tail_output:
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


def submit_pbs_job(config: DeployConfig, tail_output=True):
    # Get password from pass command
    password = subprocess.check_output(["pass", "show", "imperial"], text=True).strip()

    with console.status("[yellow]Connecting to remote...", spinner="dots"):
        gateway = Connection("imperial", user="km1124")
        c = Connection(
            user=config.remote_config.username,
            host=config.remote_config.hostname,
            gateway=gateway,
            connect_kwargs={
                "password": password,
                "look_for_keys": False,
                "allow_agent": False,
            },
        )
    console.print(f"︎[green]✔︎ Connected to cx3")

    # Write the PBS job script
    write_pbs_jobscript(c, config)

    with c.cd(config.remote_config.remote_root):
        with console.status("[yellow]Submitting job to PBS...", spinner="dots"):
            result = c.run(
                f"/opt/pbs/bin/qsub "
                f"-v WANDB_API_KEY='{config.run_config.wandb_api_key}',"
                f"GITHUB_TOKEN='{config.run_config.github_token}',"
                f"COMMIT_HASH='{config.run_config.commit_hash}' "
                "pbs_job.sh"
            )
            job_id = result.stdout.strip()
            console.print(f"[green]✔︎ PBS job submitted with ID: {job_id}")
            log_file = f"./logs/pbs-{job_id}.log"

        if tail_output:
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
