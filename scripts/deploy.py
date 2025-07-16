import argparse
import logging
from dataclasses import dataclass, field
from time import sleep
from typing import Set
from io import StringIO

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
    remotes_to_exclude: Set[str] = field(
        default_factory=lambda: {
            "gpuvm21",
            "gpuvm22",
        }
    )

    def get_remotes_to_exclude(self) -> str:
        return ",".join(self.remotes_to_exclude)


@dataclass
class SlurmJobConfig:
    nodes: int = 1
    ntasks: int = 1
    partition: str = "gpgpuB"
    job_name: str = "qtts"
    output: str = "./logs/slurm-%j.log"
    cpus_per_task: int = 1
    gres: str = "gpu:1"


@dataclass
class RunConfig:
    "Config params passed to a run instance"

    wandb_api_key: str
    commit_hash: str


@dataclass
class DeployConfig:
    run_config: RunConfig
    remote_config: RemoteConfig
    slurm_config: SlurmJobConfig = field(default_factory=SlurmJobConfig)


def main():
    args = parse_args()

    if args.commit_hash is None:
        raise RuntimeError("Missing commit hash!")

    with console.status("[yellow]Validating env variables...", spinner="dots"):
        config = DeployConfig(
            remote_config=RemoteConfig(
                username=get_env_or_throw("USERNAME"),
                hostname=get_env_or_throw("HOSTNAME"),
                remote_root=get_env_or_throw("REMOTE_ROOT"),
            ),
            run_config=RunConfig(
                wandb_api_key=get_env_or_throw("WANDB_API_KEY"),
                commit_hash=args.commit_hash,
            ),
        )
    console.print("[green]✔ Env variables validated")

    if args.action == "submit":
        submit_job(config)
    elif args.action == "push_files":
        c = Connection(config.remote_config.hostname)
        push_files(c, config.remote_config)
    elif args.action == "cancel":
        if not args.job_id:
            job_id = input("Enter the job id: ")
        else:
            job_id = args.job_id
        cancel_job(job_id, config.remote_config)
    elif args.action == "configure_environment":
        c = Connection(config.remote_config.hostname)
        push_files(c, config.remote_config)
        configure_environment(c, config.remote_config)

    # if args.action == "run_remote":
    #     if args.hostname is not None:
    #         cfg.remote.hostname = args.hostname
    #     run_remote(cfg, args.run_name)
    # elif args.action == "pull_files":
    #     c = Connection(cfg.remote.hostname)
    #     pull_files(cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="Remote job management")
    parser.add_argument(
        "--action",
        choices=[
            "submit",
            "cancel",
            "run_remote",
            "push_files",
            "pull_files",
            "configure_environment",
        ],
        required=True,
        help="Action to perform: submit a new job or cancel an existing job",
    )
    parser.add_argument("--run_name", type=str, help="Override the experiment run name")
    parser.add_argument("--hostname", type=str, help="Remote hostname")
    parser.add_argument(
        "--job-id", type=str, help="Job ID to cancel (required for cancel action)"
    )
    parser.add_argument(
        "--commit-hash", type=str, help="Hash of commit to submit as a job"
    )
    return parser.parse_args()


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

source /vol/cuda/12.0.0/setup.sh

source .venv/bin/activate

export UV_PYTHON_INSTALL_DIR="/vol/bitbucket/km1124/.cache/python"
export HF_HOME="/vol/bitbucket/km1124/.cache/huggingface"
export UV_CACHE_DIR="/vol/bitbucket/km1124/.cache/uv"

# Set VLLM profiling and logging configuration
export VLLM_TORCH_PROFILER_DIR="./trace/"
export VLLM_LOGGING_LEVEL="DEBUG"

python experiments/qwen_math.py
"""
        connection.put(
            StringIO(job_script_content), f"{config.remote_config.remote_root}/job.sh"
        )
    console.print("[green]✔ Job script written to remote")


def checkout_commit(connection, config: DeployConfig) -> None:
    commit_hash = config.run_config.commit_hash
    with console.status(
        f"[yellow]Checking out commit with hash: {commit_hash}", spinner="dots"
    ):
        with connection.cd(config.remote_config.remote_root):
            connection.run(f"git fetch && git checkout {commit_hash}")
    console.print(f"[green]Checked out commit with hash: {commit_hash}")


def submit_job(config: DeployConfig, tail_output=True):
    with console.status("[yellow]Connecting to remote...", spinner="dots"):
        c = Connection(config.remote_config.hostname)
    console.print(f"︎[green]✔︎ Connected to {config.remote_config.hostname}")

    write_jobscript(c, config)
    checkout_commit(c, config)

    with c.cd(config.remote_config.remote_root):
        with console.status("[yellow]Submitting job to SLURM...", spinner="dots"):
            console.print(
                f" - [blue]Excluded hosts: {config.remote_config.get_remotes_to_exclude()}"
            )
            result = c.run(
                f"sbatch --exclude={config.remote_config.get_remotes_to_exclude()} "
                "--export="
                f"WANDB_API_KEY='{config.run_config.wandb_api_key}',"
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


# def pull_files(config: DictConfig):
#     logger = logging.getLogger(__name__)
#     c = Connection(config.remote.hostname)

#     logger.info("Copying logs from remote to local...")
#     c.local(
#         f"rsync -avz km1124@{config.remote.hostname}:{config.remote.content_dir}/res/ {ROOT / 'res'}"
#     )
#     logger.info("Res copied successfully")


if __name__ == "__main__":
    main()
