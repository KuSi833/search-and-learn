from time import sleep
import argparse
import logging
import os
from dotenv import load_dotenv
from fabric import Connection
from rich.console import Console
from rich.status import Status
from rich.panel import Panel
from rich.text import Text

from dataclasses import dataclass

load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger("fabric").setLevel(logging.WARNING)
logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.getLogger("invoke").setLevel(logging.WARNING)

console = Console()


@dataclass
class DeployConfig:
    username: str
    hostname: str
    remote_root: str


def main():
    args = parse_args()

    with console.status("[yellow]Validating environment...", spinner="dots"):
        username = os.getenv("USERNAME")
        if username is None:
            raise ValueError("USERNAME environment variable is not set")
        hostname = os.getenv("HOSTNAME")
        if hostname is None:
            raise ValueError("HOSTNAME environment variable is not set")
        remote_root = os.getenv("REMOTE_ROOT")
        if remote_root is None:
            raise ValueError("REMOTE_ROOT environment variable is not set")

        config = DeployConfig(username, hostname, remote_root)
    console.print("[green]✔ Environment validated")

    if args.action == "submit":
        submit_job(config)
    elif args.action == "push_files":
        c = Connection(config.hostname)
        push_files(c, config)
    elif args.action == "cancel":
        if not args.job_id:
            job_id = input("Enter the job id: ")
        else:
            job_id = args.job_id
        cancel_job(job_id, config)
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
        choices=["submit", "cancel", "run_remote", "push_files", "pull_files"],
        required=True,
        help="Action to perform: submit a new job or cancel an existing job",
    )
    parser.add_argument("--run_name", type=str, help="Override the experiment run name")
    parser.add_argument("--hostname", type=str, help="Remote hostname")
    parser.add_argument(
        "--job-id", type=str, help="Job ID to cancel (required for cancel action)"
    )
    return parser.parse_args()


def push_files(connection, config: DeployConfig) -> None:
    paths_to_transfer = [
        "src",
        "experiments",
        "job.sh",
        ".env",
    ]
    with console.status("[yellow]Pushing files to remote...", spinner="dots"):
        connection.local(
            f"rsync -avz --exclude='__pycache__' --exclude='*.pyc' {' '.join(paths_to_transfer)} km1124@{config.hostname}:{config.remote_root}"
        )
    console.print(f"[green]✔ Pushed files to remote: {config.hostname}")


def submit_job(config: DeployConfig, tail_output=True):
    with console.status("[yellow]Connecting to remote...", spinner="dots"):
        c = Connection(config.hostname)
    console.print("︎[green]✔︎ Connected to ", config.hostname)

    push_files(c, config)

    with c.cd(config.remote_root):
        with console.status("[yellow]Submitting job to SLURM...", spinner="dots"):
            result = c.run("sbatch job.sh")
            job_id = result.stdout.strip().split()[-1]
            console.print(f"[green]✔︎ Job submitted with ID: {job_id}")

        if tail_output:
            log_file = f"./logs/slurm-{job_id}.log"
            with console.status("[yellow]Waiting for job to start...", spinner="dots"):
                # Check periodically if the log file exists
                while True:
                    result = c.run(
                        f"test -f {log_file} && echo 'exists' || echo 'not found'",
                        hide=True,
                    )
                    if "exists" in result.stdout:
                        break
                    sleep(2)
            console.print("[green]✔ Log file found, tailing output")
            c.run(f"tail -f {log_file}")


# def run_remote(config: DictConfig, run_name):
#     c = Connection(config.remote.hostname)

#     push_files(c, config)

#     with c.cd(config.remote.content_dir):
#         logger.info("Configuring virtual environment...")
#         c.run(f"{config.remote.uv_path} sync")

#         logger.info(f"Running job on {config.remote.hostname}")
#         c.run(f".venv/bin/python main.py --run_name {run_name}")


def cancel_job(job_id, config: DeployConfig):
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
