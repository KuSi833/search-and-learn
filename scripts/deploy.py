import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum
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
class RunConfig:
    "Config params passed to a run instance"

    wandb_api_key: str
    github_token: str
    commit_hash: str
    file_path: str = "experiments/qwen_math.py"  # path to file to run from project root


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
    gpu_type: str = "L40S"  # Empty string means any GPU type
    walltime: str = "01:00:00"  # HH:MM:SS format
    queue: str = "v1_gpu72"
    job_name: str = "qtts"
    output: str = "./logs/pbs-%j.log"


@dataclass
class DeployConfig:
    run_config: RunConfig
    remote_config: RemoteConfig
    slurm_config: SlurmJobConfig = field(default_factory=SlurmJobConfig)
    pbs_config: PbsJobConfig = field(default_factory=PbsJobConfig)


class Scheduler(str, Enum):
    SLURM = "slurm"
    PBS = "pbs"


def get_common_job_script_body(config: DeployConfig) -> str:
    """Generate the common body of job scripts shared between SLURM and PBS."""
    return f"""
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
python {config.run_config.file_path}
"""


def get_slurm_job_header(config: SlurmJobConfig) -> str:
    """Generate SLURM-specific job header."""
    return f"""#!/bin/bash
#SBATCH --nodes={config.nodes}
#SBATCH --ntasks={config.ntasks}
#SBATCH --partition={config.partition}
#SBATCH --job-name={config.job_name}
#SBATCH --output={config.output}
#SBATCH --cpus-per-task={config.cpus_per_task}
#SBATCH --gres={config.gres}
"""


def get_pbs_job_header(config: PbsJobConfig) -> str:
    """Generate PBS-specific job header."""
    # Build select statement for 1 node, 1 GPU
    select_statement = f"select=1:ncpus={config.ncpus}:mem={config.memory}:ngpus=1"
    if config.gpu_type:
        select_statement += f":gpu_type={config.gpu_type}"

    return f"""#!/bin/bash
#PBS -l {select_statement}
#PBS -l walltime={config.walltime}
#PBS -q {config.queue}
#PBS -N {config.job_name}
#PBS -o {config.output}
#PBS -j oe

# Change to submission directory
cd $PBS_O_WORKDIR
"""


def write_slurm_jobscript(connection, config: DeployConfig) -> None:
    with console.status(
        "[yellow]Writing SLURM job script to remote...", spinner="dots"
    ):
        job_script_content = get_slurm_job_header(
            config.slurm_config
        ) + get_common_job_script_body(config)
        connection.put(
            StringIO(job_script_content), f"{config.remote_config.remote_root}/job.sh"
        )
    console.print(
        f"[green]✔ SLURM job script with commit hash {config.run_config.commit_hash} written to remote"
    )


def write_pbs_jobscript(connection, config: DeployConfig) -> None:
    with console.status("[yellow]Writing PBS job script to remote...", spinner="dots"):
        job_script_content = get_pbs_job_header(
            config.pbs_config
        ) + get_common_job_script_body(config)
        connection.put(
            StringIO(job_script_content), f"{config.remote_config.remote_root}/job.sh"
        )
    console.print(
        f"[green]✔ PBS job script with commit hash {config.run_config.commit_hash} written to remote"
    )


def get_remote_config(scheduler: Scheduler) -> RemoteConfig:
    """Get base configuration from environment variables."""
    with console.status("[yellow]Validating remote env variables...", spinner="dots"):
        match scheduler:
            case Scheduler.SLURM:
                remote_config = RemoteConfig(
                    username=get_env_or_throw("USERNAME"),
                    hostname=get_env_or_throw("HOSTNAME"),
                    remote_root=get_env_or_throw("REMOTE_ROOT"),
                )
            case Scheduler.PBS:
                remote_config = RemoteConfig(
                    username=get_env_or_throw("USERNAME"),
                    hostname=get_env_or_throw("HOSTNAME"),
                    remote_root=get_env_or_throw("REMOTE_ROOT_PBS"),
                )
    console.print("[green]✔ Env variables validated")
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


@cli.command()
@click.option(
    "--partition",
    default="A100",
    help="SLURM partition to use (default: A100)",
    type=click.Choice(PARTITION_MAP.keys()),
)
@click.option("--commit-hash", required=False, help="Hash of commit to submit as a job")
@click.option("--tail/--no-tail", default=True, help="Whether to tail the output log")
@click.option(
    "--scheduler",
    default="slurm",
    help="Job scheduler to use (default: slurm)",
    type=click.Choice([s.value for s in Scheduler]),
)
def submit(
    commit_hash: str,
    partition: str,
    tail: bool,
    scheduler_str: str,
):
    """Submit a new job to SLURM or PBS scheduler."""
    scheduler = Scheduler(scheduler_str)
    remote_config = get_remote_config(scheduler)
    run_config = get_run_config(commit_hash)

    match scheduler:
        case Scheduler.SLURM:
            slurm_config = SlurmJobConfig(partition=PARTITION_MAP[partition])
            config = DeployConfig(
                run_config=run_config,
                remote_config=remote_config,
                slurm_config=slurm_config,
            )
            submit_slurm_job(config, tail_output=tail)
        case Scheduler.PBS:
            pbs_config = PbsJobConfig()
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
                "job.sh"
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
    with console.status("[yellow]Connecting to remote...", spinner="dots"):
        c = Connection(config.remote_config.hostname)
    console.print(f"︎[green]✔︎ Connected to {config.remote_config.hostname}")

    # Write the PBS job script
    write_pbs_jobscript(c, config)

    with c.cd(config.remote_config.remote_root):
        with console.status("[yellow]Submitting job to PBS...", spinner="dots"):
            result = c.run(
                f"qsub "
                f"-v WANDB_API_KEY='{config.run_config.wandb_api_key}',"
                f"GITHUB_TOKEN='{config.run_config.github_token}',"
                f"COMMIT_HASH='{config.run_config.commit_hash}' "
                "job.sh"
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
