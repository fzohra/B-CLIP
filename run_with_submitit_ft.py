"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
from pathlib import Path

import main_clip_ft as main_slip
import submitit


def parse_args():
    parser = main_slip.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for Beta-CLIP pre-training", parents=[parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=8, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=2800, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="slurm", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--gpu_type", default="v100", type=str, help="GPU name.")
    parser.add_argument("--job_name", default="clip", type=str, help="Job name.")
    return parser.parse_args()


def get_shared_folder() -> Path:
    if Path("/ibex/project/c2262/SLIP/.ckpts").is_dir():
        p = Path(f"/ibex/project/c2262/SLIP/.ckpts")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main_clip_ft as main_slip

        self._setup_gpu_args()
        main_slip.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    kwargs = {}
    kwargs['slurm_constraint'] = args.gpu_type

    executor.update_parameters(
        mem_gb=48 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=6,
        # cpus_per_gpu=6,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name=args.job_name)

    args.dist_url = get_init_file().as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
