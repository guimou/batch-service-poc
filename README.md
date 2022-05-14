# batch-service-poc

PoC for batch service development

## Repo description

* In the `development` folder there are two notebooks as they would be used directly by data scientists:
  * The standard one to train from local files and save the model locally.
  * The S3 one to read the data from S3 first, then write back to S3.
* At the root there is:
  * `train.py`, a cleaned version of the notebook code.
  * `requirements.txt`, Python packages needed by the notebooks or the Python file.
  * `Dockerfile` to build the container image that can be used to train the model.
  * `env-minimal-example.list`, an example of the minimal environment variables that need to be passed to the container to train the mode. Other variables are are available but have default values in the code (see L20-43 in `train.py`).

## Usage

Build container image:

```bash
podman build -t model-trainer:v0.0.1 .
```

Run container image (after editing/complementing  `env-minimal-example.list` with your values):

```bash
podman run --rm -it --network="host" --env-file ./env-minimal-example.list model-trainer:v0.0.1
```

NOTE: `--network="host"` is only needed if your S3 endpoint is on your host at 127.0.0.1/0.0.0.0/localhost. It can be removed if it's on another address accessible by the container.
