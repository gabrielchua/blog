# Deploying HuggingFace models on AWS SageMaker with 2 commands

You can have a HuggingFace model up and running on SageMaker in just a few lines of code. 

===

First, you need to import the necessary modules:

```python
from sagemaker import get_execution_role
from sagemaker.huggingface.model import HuggingFaceModel
```

Set up the necessary environment variables, including the model ID, instance type, and versions.

```python
ENDPOINT_NAME = "baai-bge-large-en-v1-5"

HF_ENV = {
    'HF_MODEL_ID':'BAAI/bge-large-en-v1.5',
    'HF_TASK':'feature-extraction'
}

INSTANCE_TYPE = "ml.m5.xlarge"

TRANSFORMER_VER = "4.26"

PY_VERSION = "py39"

PYTORCH_VERSION = "1.13"
```

Create a HuggingFaceModel model with the specified configurations.

Here we are using SageMaker's built-in container images with specific versions of python, pytorch and transformers. A full list of available images can be found [here](https://github.com/aws/deep-learning-containers/blob/master/available_images.md).

```python
huggingface_model = HuggingFaceModel(
   env=HF_ENV,
   role=get_execution_role(),
   transformers_version=TRANSFORMER_VER,
   pytorch_version=PYTORCH_VERSION,
   py_version=PY_VERSION,
)
```

Then use the `.deploy` method.

```python
predictor = huggingface_model.deploy(
    endpoint_name=ENDPOINT_NAME,
    initial_instance_count=1,
    instance_type=INSTANCE_TYPE
)
```

And that’s it! With just a few lines of code, your HuggingFace model is live on AWS SageMaker. It’s incredibly fast to get started and deploy.