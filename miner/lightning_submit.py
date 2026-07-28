"""
Submits the pearl-miner Docker image as a Lightning AI Job and streams its
logs live, the same way `modal run modal_app.py` did.

Requires:
    pip install -U lightning-sdk
    export LIGHTNING_USER_ID=ebe743f2-5f7e-41e6-a8dc-179dc88a615b
    export LIGHTNING_API_KEY=a039a54d-d1ec-43cf-8fbf-62be094beda6

Before running this:
    docker build -t <YOUR_DOCKERHUB_USER>/pearl-miner:latest -f Dockerfile .
    docker push <YOUR_DOCKERHUB_USER>/pearl-miner:latest
"""

import sys
import time

from lightning_sdk import Job, Machine

# --- EDIT THESE THREE VALUES ---
TEAMSPACE = "your-username/your-teamspace"     # from lightning.ai URL, e.g. "shireff/personal"
DOCKER_IMAGE = "your-dockerhub-user/pearl-miner:latest"
MACHINE = Machine.L4                            # same GPU as the Modal version
# --------------------------------

JOB_NAME = "pearl-miner"


def main():
    print(f"Submitting job '{JOB_NAME}' on {MACHINE} using image {DOCKER_IMAGE} ...")

    job = Job.run(
        name=JOB_NAME,
        teamspace=TEAMSPACE,
        image=DOCKER_IMAGE,
        machine=MACHINE,
        command="python3.12 /root/pearl/run_mining.py",
        interruptible=False,   # set True if you want cheaper but preemptible compute
    )

    print(f"Job submitted. Status: {job.status}")
    print("Streaming logs (Ctrl+C stops watching, the job keeps running remotely)...\n")

    try:
        # job.logs is a live-following iterator over the job's stdout/stderr
        for line in job.logs:
            print(line, end="" if line.endswith("\n") else "\n")
    except KeyboardInterrupt:
        print("\nStopped watching logs locally. The job is still running on Lightning.")
        print(f"Reattach anytime with:\n"
              f"  from lightning_sdk import Job\n"
              f"  job = Job('{JOB_NAME}', teamspace='{TEAMSPACE}')\n"
              f"  for line in job.logs: print(line)")
        sys.exit(0)

    print(f"\nJob finished with status: {job.status}")


if __name__ == "__main__":
    main()