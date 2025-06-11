import json
import subprocess
import time
from pathlib import Path


def get_all_uuids() -> list[str]:
    return (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        .strip()
        .splitlines()
    )


def json_dumps(obj: dict) -> str:
    return json.dumps(obj, indent=4, sort_keys=True)


def setup_docker_config():
    daemon_config_file = Path("/etc/docker/daemon.json")
    daemon_config = json.loads(daemon_config_file.read_text())
    print("Previous daemon config:\n", json_dumps(daemon_config))

    daemon_config["node-generic-resources"] = [
        "gpu=" + uuid for uuid in get_all_uuids()
    ]

    print("New daemon config:\n", json_dumps(daemon_config))
    daemon_config_file.write_text(json_dumps(daemon_config))


def setup_nvidia_docker_config():
    nvidia_container_config_file = Path("/etc/nvidia-container-runtime/config.toml")
    nvidia_container_config = nvidia_container_config_file.read_text()
    line = '#swarm-resource = "DOCKER_RESOURCE_GPU"'
    if line in nvidia_container_config:
        print("uncommenting", line)
        nvidia_container_config = nvidia_container_config.replace(
            line, line.removeprefix("#")
        )
        print("New nvidia-container config:\n", nvidia_container_config)
        nvidia_container_config_file.write_text(nvidia_container_config)
    else:
        print(
            "Line not found in nvidia-container config, did you already uncomment it?"
        )


def restarting_docker():
    print("Restarting docker")
    subprocess.check_call(["systemctl", "restart", "docker"])
    time.sleep(1)


def checking_nvidia_docker():
    print("Checking nvidia-docker")
    subprocess.check_call(
        [
            "docker",
            "run",
            "--rm",
            "--runtime=nvidia",
            "--gpus",
            "all",
            "ubuntu",
            "nvidia-smi",
        ]
    )


def main():
    setup_docker_config()
    setup_nvidia_docker_config()
    restarting_docker()
    checking_nvidia_docker()


if __name__ == "__main__":
    main()
