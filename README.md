# ML Model Deployment Engine

Production-grade ML model deployment and serving engine for managing model lifecycles across multiple environments.

## Features

- **Multi-Framework Support**: PyTorch, TensorFlow, ONNX, scikit-learn, LLM
- **Multi-Environment Deployment**: dev, staging, production, edge
- **Deployment Strategies**: Blue-green, canary, rolling updates
- **Auto-Scaling**: Scale deployments by replica count
- **Metrics & Monitoring**: Request tracking, latency monitoring, error rates
- **Health Checks**: Deployment health status monitoring

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Register a Model

```python
from src.model_deploy import ModelDeploymentEngine

engine = ModelDeploymentEngine()

artifact = engine.register_model(
    name="my_model",
    version="1.0.0",
    framework="pytorch",
    path="/path/to/model.pt",
    input_schema={"image": "tensor"},
    output_schema={"prediction": "float"}
)
```

### Deploy a Model

```bash
python src/model_deploy.py deploy model_123 production --replicas 3
```

### Get Metrics

```bash
python src/model_deploy.py metrics deployment_456
```

### List Deployments

```bash
python src/model_deploy.py list --env production --status active
```

## Database

Models and deployments are stored in SQLite at `~/.blackroad/model-deploy.db`.

## License

MIT
