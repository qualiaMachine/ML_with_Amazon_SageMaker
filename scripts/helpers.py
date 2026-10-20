"""Helper functions used throughout the lesson.

Learners clone the lesson repository once, in the Accessing S3 episode, into
``/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/`` and import this module with::

    import sys
    sys.path.insert(0, "/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts")
    import helpers
"""
import boto3
from datetime import datetime, timedelta
import os

def get_s3_bucket_size(bucket_name):
    """
    Calculate the total size of an S3 bucket in megabytes and gigabytes.

    Parameters:
        bucket_name (str): The name of the S3 bucket.

    Returns:
        dict: A dictionary containing the total size in megabytes and gigabytes.
    """
    s3 = boto3.client('s3')
    total_size_bytes = 0

    # List and sum the size of all objects in the bucket
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get('Contents', []):
            total_size_bytes += obj['Size']

    # Convert the total size to megabytes and gigabytes for readability
    total_size_mb = total_size_bytes / (1024 ** 2)
    total_size_gb = total_size_bytes / (1024 ** 3)

    return {
        'size_mb': total_size_mb,
        'size_gb': total_size_gb
    }

def calculate_s3_storage_cost(bucket_name):
    """
    Calculates the estimated monthly storage cost of an S3 bucket based on its total size.

    Parameters:
        bucket_name (str): The name of the S3 bucket.

    Returns:
        tuple: A tuple containing the estimated monthly storage cost in USD and the total storage size in gigabytes.
    """
    # Retrieve bucket size using the get_s3_bucket_size function
    size_info = get_s3_bucket_size(bucket_name)
    total_size_gb = size_info['size_gb']

    # AWS S3 Standard Storage pricing for US East (N. Virginia) region
    # Pricing tiers as of November 1, 2024
    first_50_tb_price_per_gb = 0.023  # per GB for the first 50 TB
    next_450_tb_price_per_gb = 0.022  # per GB for the next 450 TB
    over_500_tb_price_per_gb = 0.021  # per GB for storage over 500 TB

    # Calculate the cost based on the size
    if total_size_gb <= 50 * 1024:
        cost = total_size_gb * first_50_tb_price_per_gb
    elif total_size_gb <= 500 * 1024:
        cost = (50 * 1024 * first_50_tb_price_per_gb) + \
               ((total_size_gb - 50 * 1024) * next_450_tb_price_per_gb)
    else:
        cost = (50 * 1024 * first_50_tb_price_per_gb) + \
               (450 * 1024 * next_450_tb_price_per_gb) + \
               ((total_size_gb - 500 * 1024) * over_500_tb_price_per_gb)

    return cost, total_size_gb

def list_S3_objects(bucket_name):
    """
    Lists all objects in a specified S3 bucket.

    Parameters:
        bucket_name (str): The name of the S3 bucket.

    Returns:
        list: A list of object keys (paths) in the bucket, or an empty list if the bucket is empty.
    """
    s3 = boto3.client('s3')
    file_list = []

    # Use paginator to list all objects in the bucket
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get('Contents', []):
            file_list.append(obj['Key'])  # Add each object's key to the list

    # If the bucket is empty, return an empty list
    return file_list
    
def get_instance_cost(instance_type, days=1):
    """
    Fetches the cost for a specific instance type over a specified number of days.
    """
    client = boto3.client('ce', region_name='us-east-1')
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    response = client.get_cost_and_usage(
        TimePeriod={'Start': start.strftime('%Y-%m-%d'), 'End': end.strftime('%Y-%m-%d')},
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        Filter={
            'And': [
                {'Dimensions': {'Key': 'SERVICE', 'Values': ['Amazon Elastic Compute Cloud - Compute']}},
                {'Dimensions': {'Key': 'INSTANCE_TYPE', 'Values': [instance_type]}}
            ]
        }
    )
    total_cost = sum(float(day['Total']['UnblendedCost']['Amount']) for day in response['ResultsByTime'])
    return total_cost

def list_running_ec2_instances():
    """
    Lists all running EC2 instances in the account with instance ID and instance type.
    """
    ec2_client = boto3.client('ec2')
    response = ec2_client.describe_instances(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])
    instances = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instances.append({'InstanceId': instance['InstanceId'], 'InstanceType': instance['InstanceType']})
    return instances

def get_sagemaker_notebook_status(notebook_instance_name):
    """
    Fetches the status of a SageMaker notebook instance.
    """
    sagemaker_client = boto3.client('sagemaker')
    response = sagemaker_client.describe_notebook_instance(NotebookInstanceName=notebook_instance_name)
    return response['NotebookInstanceStatus']

def get_notebook_instance_info(notebook_instance_name, region=None):
    """
    Fetches the status and instance type of a specific SageMaker notebook instance.

    Parameters:
        notebook_instance_name (str): Name of the SageMaker notebook instance (not an .ipynb file).
        region (str or None): AWS region of the instance. Defaults to the session's region
            (e.g. ``boto3.Session().region_name``); never hard-code a region in lesson code.

    Returns:
        dict: {'Status': ..., 'InstanceType': ...}
    """
    if region is None:
        region = boto3.Session().region_name
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    response = sagemaker_client.describe_notebook_instance(NotebookInstanceName=notebook_instance_name)
    return {'Status': response['NotebookInstanceStatus'], 'InstanceType': response['InstanceType']}

def calculate_tuning_job_time(tuner, region=None):
    """
    Sums training and billable time across every training job launched by a SageMaker
    hyperparameter tuning job. Matches the call in the Hyperparameter-tuning episode:
    ``helpers.calculate_tuning_job_time(tuner, region)``.

    SageMaker bills per second: each training job's ``BillableTimeInSeconds`` (instance
    reserved for you, including container pull and data download) multiplied by its
    instance count. The tuning job itself is not billed.

    Parameters:
        tuner (sagemaker.tuner.HyperparameterTuner): A tuner whose ``.fit()`` has been called.
        region (str or None): AWS region. Defaults to the session's region.

    Returns:
        dict: instance type, max_jobs, max_parallel_jobs, number of completed jobs, and total
              training / billable seconds (billable already multiplied by instance count).
    """
    if region is None:
        region = boto3.Session().region_name
    sagemaker_client = boto3.client('sagemaker', region_name=region)

    tuning_job_name = tuner.latest_tuning_job.name
    tuning_job_desc = sagemaker_client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name)
    instance_type = tuning_job_desc['TrainingJobDefinition']['ResourceConfig']['InstanceType']
    limits = tuning_job_desc['HyperParameterTuningJobConfig']['ResourceLimits']

    # Page through all child training jobs of the tuning job
    paginator = sagemaker_client.get_paginator('list_training_jobs_for_hyper_parameter_tuning_job')
    job_names = []
    for page in paginator.paginate(HyperParameterTuningJobName=tuning_job_name, StatusEquals='Completed'):
        job_names.extend(j['TrainingJobName'] for j in page['TrainingJobSummaries'])

    total_training_seconds = 0
    total_billable_seconds = 0
    for job_name in job_names:
        job_desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        total_training_seconds += job_desc.get('TrainingTimeInSeconds', 0)
        total_billable_seconds += job_desc.get('BillableTimeInSeconds', 0) * job_desc['ResourceConfig']['InstanceCount']

    print(f"Tuning job: {tuning_job_name}")
    print(f"Instance Type: {instance_type}")
    print(f"Max Jobs: {limits['MaxNumberOfTrainingJobs']}")
    print(f"Max Parallel Jobs: {limits['MaxParallelTrainingJobs']}")
    print(f"Completed training jobs: {len(job_names)}")
    print(f"Total training time across all jobs: {total_training_seconds / 60:.1f} minutes")
    print(f"Total billable time across all jobs: {total_billable_seconds / 60:.1f} minutes (x instance count)")

    return {
        'TuningJobName': tuning_job_name,
        'InstanceType': instance_type,
        'MaxJobs': limits['MaxNumberOfTrainingJobs'],
        'MaxParallelJobs': limits['MaxParallelTrainingJobs'],
        'CompletedJobs': len(job_names),
        'TotalTrainingSeconds': total_training_seconds,
        'TotalBillableSeconds': total_billable_seconds,
    }

def get_current_costs(days=1):
    """
    Fetches the current total costs for the specified time period across all AWS services.
    """
    client = boto3.client('ce', region_name='us-east-1')
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    response = client.get_cost_and_usage(
        TimePeriod={'Start': start.strftime('%Y-%m-%d'), 'End': end.strftime('%Y-%m-%d')},
        Granularity='DAILY',
        Metrics=['UnblendedCost']
    )
    total_cost = sum(float(day['Total']['UnblendedCost']['Amount']) for day in response['ResultsByTime'])
    return total_cost


def convert_files(direction="notebook_to_python", directory="."):
    """Convert every notebook in ``directory`` to a .py script, or every .py script
    to a notebook, with jupytext (``pip install jupytext``).

    direction: "notebook_to_python" (default) or "python_to_notebook".
    Converts *every* matching file in the directory, so run it inside a folder that
    holds only the files you mean to convert.
    """
    import subprocess
    if direction == "notebook_to_python":
        src_ext, dst_ext, fmt = ".ipynb", ".py", "py"
    elif direction == "python_to_notebook":
        src_ext, dst_ext, fmt = ".py", ".ipynb", "notebook"
    else:
        raise ValueError("direction must be 'notebook_to_python' or 'python_to_notebook'")
    for name in sorted(os.listdir(directory)):
        if not name.endswith(src_ext):
            continue
        src = os.path.join(directory, name)
        dst = os.path.join(directory, name[: -len(src_ext)] + dst_ext)
        subprocess.run(["jupytext", "--to", fmt, src, "--output", dst], check=True)
        print(f"Converted {name} to {os.path.basename(dst)}")


if __name__ == "__main__":
    # Example: Get the cost of a specific EC2 instance type over the last day
    instance_type = 'g4dn.xlarge'  # Replace with your desired instance type
    cost = get_instance_cost(instance_type)
    print(f"Cost for {instance_type} over the last day: ${cost:.2f}")

    # Example: List all running EC2 instances with their IDs and types
    running_instances = list_running_ec2_instances()
    print("Running EC2 Instances:")
    for instance in running_instances:
        print(f"Instance ID: {instance['InstanceId']}, Instance Type: {instance['InstanceType']}")

    # Example: Check the status of a specific SageMaker notebook instance
    notebook_instance_name = 'YourNotebookInstanceName'  # Replace with your notebook instance name
    status = get_sagemaker_notebook_status(notebook_instance_name)
    print(f"SageMaker Notebook '{notebook_instance_name}' Status: {status}")

    # Example: Fetch status and instance type of a specific notebook instance
    notebook_info = get_notebook_instance_info(notebook_instance_name, boto3.Session().region_name)
    print(f"Notebook Instance Info for '{notebook_instance_name}':")
    print(f"Status: {notebook_info['Status']}, Instance Type: {notebook_info['InstanceType']}")

    # Example: Get the total AWS cost across all services over the last 7 days
    total_cost = get_current_costs(days=7)
    print(f"Total AWS costs over the last 7 days: ${total_cost:.2f}")


# ---------------------------------------------------------------------------
# Fine-tuning cost estimator (fine-tuning episode, "Back of the envelope")
# ---------------------------------------------------------------------------

# SageMaker training instances the lesson refers to: GPUs per box, GPU memory per GPU (GB),
# bf16 dense peak per GPU (TFLOPS, from NVIDIA's data sheets), and the us-east-1 on-demand
# price per instance-hour from the AWS price list of 2026-10-16. The price is for the whole box.
GPU_INSTANCES = {
    "ml.g4dn.xlarge":   {"gpus": 1, "gpu": "T4",        "gpu_mem_gb": 16, "tflops_bf16": 65,  "price": 0.74,  "note": "no bf16 support; modern LLM fine-tuning is slow or fails"},
    "ml.g6.xlarge":     {"gpus": 1, "gpu": "L4",        "gpu_mem_gb": 24, "tflops_bf16": 121, "price": 1.13},
    "ml.g5.xlarge":     {"gpus": 1, "gpu": "A10G",      "gpu_mem_gb": 24, "tflops_bf16": 70,  "price": 1.41},
    "ml.g5.2xlarge":    {"gpus": 1, "gpu": "A10G",      "gpu_mem_gb": 24, "tflops_bf16": 70,  "price": 1.52},
    "ml.g6e.xlarge":    {"gpus": 1, "gpu": "L40S",      "gpu_mem_gb": 48, "tflops_bf16": 362, "price": 2.61},
    "ml.g6e.2xlarge":   {"gpus": 1, "gpu": "L40S",      "gpu_mem_gb": 48, "tflops_bf16": 362, "price": 2.80},
    "ml.g5.12xlarge":   {"gpus": 4, "gpu": "A10G",      "gpu_mem_gb": 24, "tflops_bf16": 70,  "price": 7.09},
    "ml.g6e.12xlarge":  {"gpus": 4, "gpu": "L40S",      "gpu_mem_gb": 48, "tflops_bf16": 362, "price": 13.12},
    "ml.p4d.24xlarge":  {"gpus": 8, "gpu": "A100 40GB", "gpu_mem_gb": 40, "tflops_bf16": 312, "price": 25.25},
    "ml.p5.48xlarge":   {"gpus": 8, "gpu": "H100 80GB", "gpu_mem_gb": 80, "tflops_bf16": 989, "price": 63.30},
}

# Bytes of GPU memory per model parameter, from the fine-tuning episode's table.
BYTES_PER_PARAM = {"lora": 2.5, "full": 16.0}

# Multiplier on parameters x tokens: 6 for a full fine-tune (forward + backward through
# every weight), 4 for LoRA (no weight gradients for the frozen base).
FLOPS_PER_PARAM_TOKEN = {"lora": 4.0, "full": 6.0}


def estimate_finetune_cost(params_billions, examples, tokens_per_example, epochs=2, method="lora",
                           instance_type=None, runs=1, utilization=0.25, gradient_checkpointing=True,
                           activations_gb=4.0, startup_minutes=10, verbose=True):
    """
    Back-of-the-envelope cost of a fine-tuning run, using the two rules from the
    fine-tuning episode. Good to within about a factor of two, which is enough to tell
    a $5 run from a $500 one before anything is launched.

    Rule 1, memory: bytes per parameter x parameters + activations must fit the GPU memory
    (summed across GPUs on a multi-GPU box, where the model is sharded).
    Rule 2, time: FLOPs = k x parameters x tokens (k = 4 for LoRA, 6 for full), divided by
    the GPU's peak bf16 rate x utilization; +50% when gradient checkpointing recomputes
    the forward pass. Cost = instance-hours x price per instance-hour + startup.

    Parameters
    ----------
    params_billions : float   total parameters of the model you train (3, 7, 27, 70 ...).
                              For a vision-language model include the vision tower (CHURRO-3B ~3.75).
    examples : int            training examples (pages, documents, images, recordings ...)
    tokens_per_example : int  tokens the model sees per example. Text: words x 1.3. An image on a
                              Qwen-VL style model: pixels / 784 (about 1,300 for one megapixel) plus
                              the text. An EEG window or audio clip: ask the model card what one
                              example becomes after tokenisation.
    epochs : float            passes over the data
    method : "lora" | "full"
    instance_type : str       a key of helpers.GPU_INSTANCES; None picks the cheapest that fits
    runs : int                how many runs the project will need (a first run is never the last;
                              five to ten is realistic before any hyperparameter sweep)
    utilization : float       fraction of peak the GPU achieves; 0.25 unless you have measured
    gradient_checkpointing : bool   adds ~50% compute; needed on small GPUs
    activations_gb : float    headroom for activations; raise for long sequences or big batches
    startup_minutes : float   per-job fixed overhead (container pull, data download, model load)

    Returns a dict with the numbers and prints a report when verbose.
    """
    method = method.lower()
    if method not in BYTES_PER_PARAM:
        raise ValueError("method must be 'lora' or 'full'")
    params = params_billions * 1e9
    tokens = examples * tokens_per_example * epochs
    mem_gb = BYTES_PER_PARAM[method] * params / 1e9 + activations_gb
    flops = FLOPS_PER_PARAM_TOKEN[method] * params * tokens * (1.5 if gradient_checkpointing else 1.0)

    def fits(spec):
        return spec["gpus"] * spec["gpu_mem_gb"] >= mem_gb

    if instance_type is None:
        candidates = [k for k, v in GPU_INSTANCES.items() if fits(v) and "note" not in v]
        if not candidates:
            raise ValueError(f"needs ~{mem_gb:.0f} GB of GPU memory; nothing in GPU_INSTANCES holds it. "
                             "Use LoRA, a smaller model, or more than one box.")
        instance_type = min(candidates, key=lambda k: GPU_INSTANCES[k]["price"])
    spec = GPU_INSTANCES[instance_type]

    gpu_seconds = flops / (spec["tflops_bf16"] * 1e12 * utilization)
    gpu_hours = gpu_seconds / 3600
    instance_hours = gpu_hours / spec["gpus"] + startup_minutes / 60
    cost_per_run = instance_hours * spec["price"]
    max_run_minutes = max(15, round(2 * instance_hours * 60 / 5) * 5)
    result = {
        "method": method, "params_billions": params_billions, "tokens": tokens,
        "memory_gb_needed": round(mem_gb, 1), "instance_type": instance_type,
        "fits": fits(spec), "gpu_hours": round(gpu_hours, 2),
        "wall_clock_hours": round(instance_hours, 2), "cost_per_run": round(cost_per_run, 2),
        "runs": runs, "cost_total": round(cost_per_run * runs, 2),
        "max_run_seconds": int(max_run_minutes * 60),
    }
    if verbose:
        gpus = f'{spec["gpus"]} x {spec["gpu"]}'
        print(f"{method.upper()} fine-tune of a {params_billions:g}B model: {examples:,} examples x "
              f"{tokens_per_example:,} tokens x {epochs:g} passes = {tokens/1e6:,.1f}M tokens")
        print(f"memory:     ~{mem_gb:.0f} GB needed; {instance_type} has {spec['gpus'] * spec['gpu_mem_gb']} GB ({gpus})")
        if not fits(spec):
            fitting = [k for k, v in GPU_INSTANCES.items() if fits(v) and "note" not in v]
            cheapest = min(fitting, key=lambda k: GPU_INSTANCES[k]["price"]) if fitting else None
            print(f"\n*** DOES NOT FIT: {instance_type} cannot hold this job. It would start, bill ~{startup_minutes:g} min of startup "
                  f"(~${startup_minutes / 60 * spec['price']:.2f}), then die with a CUDA out-of-memory error. ***")
            if cheapest:
                c = GPU_INSTANCES[cheapest]
                print(f"    smallest instance that fits: {cheapest} ({c['gpus']} x {c['gpu']}, {c['gpus'] * c['gpu_mem_gb']} GB, ${c['price']}/h); "
                      f"rerun with instance_type=\"{cheapest}\"")
            else:
                print("    nothing in GPU_INSTANCES holds it: use LoRA, a smaller model, or more than one box")
            if method == "full":
                print("    or switch to method=\"lora\": about %.0f GB instead of %.0f" % (BYTES_PER_PARAM["lora"] * params / 1e9 + activations_gb, mem_gb))
            print("    the numbers below are what the run WOULD cost on this instance if it fit")
        print(f"compute:    {flops:.2e} FLOPs at {utilization:.0%} of the {spec['tflops_bf16']} TFLOPS peak "
              f"= {gpu_hours:.2f} GPU-hours")
        print(f"wall-clock: ~{instance_hours * 60:.0f} min including {startup_minutes:g} min startup")
        print(f"cost:       ~${cost_per_run:.2f} per run at ${spec['price']}/h; ~${cost_per_run * runs:.2f} for {runs} run(s)")
        print(f"set max_run={result['max_run_seconds']}  ({max_run_minutes:.0f} min, about twice the estimate)")
        if "note" in spec:
            print(f"warning:    {spec['gpu']}: {spec['note']}")
        if method == "full":
            print("warning:    full fine-tuning updates every weight; do not run one on the class account")
        if method == "lora" and params_billions > 7:
            print("warning:    LoRA above 7B is outside what the class account allows")
        if spec["gpus"] > 1:
            print("warning:    multi-GPU instance; not allowed on the class account, and more GPUs do not make the run cheaper")
    return result
