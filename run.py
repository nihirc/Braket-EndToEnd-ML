from braket.jobs.local.local_job import LocalQuantumJob
from braket.aws import AwsQuantumJob, AwsSession
from braket.jobs.image_uris import retrieve_image, Framework
from braket.jobs.config import InstanceConfig
import time
import boto3

region = "us-west-2"
sess = AwsSession(boto3.session.Session(region_name=region))
job_name = f"qml-local-job-{int(time.time())}"
q_device = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
script_path = "src/script.py"
s3_reduced_data_bucket = "s3://bucket/data/" # Dummy location
image_uri_tf = retrieve_image(Framework.PL_TENSORFLOW, region)

hyperparameters_local = {
    "n_qubits": 4,                
    "n_shots": 10,               
    "step": 0.001,               
    "batch_size": 32,              
    "num_epochs": 1,              
    "max_parallel": 50,           
    "use_local_simulator": True,  
}

## Run it locally as a job
job = LocalQuantumJob.create(
    device=q_device,
    source_module=script_path,
    image_uri=image_uri_tf,
    job_name=job_name,
    hyperparameters=hyperparameters_local,
    input_data=s3_reduced_data_bucket,
    aws_session=sess
)

# job_instance_type = "ml.m5.4xlarge"

# hyperparameters_quantum_job_local = {
#     "n_qubits": "4",               
#     "n_shots": "10",               
#     "step": "0.001",              
#     "batch_size": "32",              
#     "num_epochs": "2",             
#     "max_parallel": "50",           
#     "use_local_simulator": "False"    
# }

# quantum_jobs_local = []
# hyperparameters = hyperparameters_quantum_job_local.copy()
# hyperparameters['n_qubits'] = str(4)

# quantum_jobs_local.append({
#     'job_name': job_name,
#     'hyperparameters': hyperparameters,
# })

# ## Run it as a job on Braket. Provide quantum hardware ARN to run on a quantum device
# job = AwsQuantumJob.create(
#     device=q_device,
#     source_module=script_path,
#     job_name=job_name,
#     instance_config=InstanceConfig(instanceType=job_instance_type),
#     image_uri=image_uri_tf,
#     copy_checkpoints_from_job=None,
#     hyperparameters=hyperparameters,
#     input_data="s3://amazon-braket-us-west-2-782067938675/data/",
#     wait_until_complete=False
# )

print(job.name)