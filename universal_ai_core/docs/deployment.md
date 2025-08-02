# Deployment Guide

## Overview

This guide covers deployment strategies for Universal AI Core across different environments, from development to enterprise production. The system supports containerized deployments, cloud platforms, and high-availability configurations adapted from Saraphis deployment patterns.

## Deployment Options

### 1. Local Development Deployment

#### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd universal_ai_core

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run development server
python -c "
from universal_ai_core import create_development_api
api = create_development_api()
print('Universal AI Core development server started')
input('Press Enter to stop...')
api.shutdown()
"
```

#### Configuration
```yaml
# config/development.yaml
core:
  max_workers: 4
  enable_monitoring: true
  debug_mode: true
  log_level: "DEBUG"

api:
  enable_caching: true
  cache_size: 1000
  enable_safety_checks: false
  rate_limit_requests_per_minute: 100

plugins:
  feature_extractors:
    molecular:
      enabled: true
      rdkit_enabled: false  # Optional for development
    cybersecurity:
      enabled: true
    financial:
      enabled: true

logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 2. Docker Deployment

#### Basic Docker Setup

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "from universal_ai_core import create_api; api = create_api(); health = api.get_health_status(); print('healthy' if health['overall'] == 'healthy' else 'unhealthy'); api.shutdown()"

# Start application
CMD ["python", "-m", "universal_ai_core.server"]
```

**Build and Run:**
```bash
# Build image
docker build -t universal-ai-core:latest .

# Run container
docker run -d \
  --name universal-ai-core \
  -p 8000:8000 \
  -e UNIVERSAL_AI_MAX_WORKERS=8 \
  -e UNIVERSAL_AI_ENABLE_MONITORING=true \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  universal-ai-core:latest
```

#### Docker Compose

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  universal-ai-core:
    build: .
    ports:
      - "8000:8000"
    environment:
      - UNIVERSAL_AI_MAX_WORKERS=8
      - UNIVERSAL_AI_ENABLE_MONITORING=true
      - UNIVERSAL_AI_LOG_LEVEL=INFO
      - UNIVERSAL_AI_CACHE_SIZE=10000
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - universal_ai_cache:/app/cache
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "from universal_ai_core import create_api; api = create_api(); health = api.get_health_status(); exit(0 if health['overall'] == 'healthy' else 1); api.shutdown()"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Optional: Redis for distributed caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  # Optional: PostgreSQL for persistent storage
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: universal_ai_core
      POSTGRES_USER: uac_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  universal_ai_cache:
  redis_data:
  postgres_data:
```

**Run with Docker Compose:**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f universal-ai-core

# Stop services
docker-compose down
```

### 3. Kubernetes Deployment

#### Basic Kubernetes Configuration

**namespace.yaml:**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: universal-ai-core
```

**configmap.yaml:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: universal-ai-core-config
  namespace: universal-ai-core
data:
  config.yaml: |
    core:
      max_workers: 8
      enable_monitoring: true
      debug_mode: false
      log_level: "INFO"
    
    api:
      enable_caching: true
      cache_size: 10000
      enable_safety_checks: true
      rate_limit_requests_per_minute: 1000
    
    plugins:
      feature_extractors:
        molecular:
          enabled: true
        cybersecurity:
          enabled: true
        financial:
          enabled: true
```

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: universal-ai-core
  namespace: universal-ai-core
  labels:
    app: universal-ai-core
spec:
  replicas: 3
  selector:
    matchLabels:
      app: universal-ai-core
  template:
    metadata:
      labels:
        app: universal-ai-core
    spec:
      containers:
      - name: universal-ai-core
        image: universal-ai-core:latest
        ports:
        - containerPort: 8000
        env:
        - name: UNIVERSAL_AI_MAX_WORKERS
          value: "8"
        - name: UNIVERSAL_AI_ENABLE_MONITORING
          value: "true"
        - name: UNIVERSAL_AI_LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: cache-volume
          mountPath: /app/cache
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config-volume
        configMap:
          name: universal-ai-core-config
      - name: cache-volume
        emptyDir:
          sizeLimit: 1Gi
```

**service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: universal-ai-core-service
  namespace: universal-ai-core
spec:
  selector:
    app: universal-ai-core
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

**ingress.yaml:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: universal-ai-core-ingress
  namespace: universal-ai-core
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.universal-ai-core.com
    secretName: universal-ai-core-tls
  rules:
  - host: api.universal-ai-core.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: universal-ai-core-service
            port:
              number: 80
```

**Deploy to Kubernetes:**
```bash
# Apply configurations
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get pods -n universal-ai-core
kubectl logs -f deployment/universal-ai-core -n universal-ai-core

# Scale deployment
kubectl scale deployment universal-ai-core --replicas=5 -n universal-ai-core
```

### 4. Cloud Platform Deployments

#### AWS ECS Deployment

**task-definition.json:**
```json
{
  "family": "universal-ai-core",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "universal-ai-core",
      "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/universal-ai-core:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "UNIVERSAL_AI_MAX_WORKERS",
          "value": "8"
        },
        {
          "name": "UNIVERSAL_AI_ENABLE_MONITORING",
          "value": "true"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/universal-ai-core",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "python -c \"from universal_ai_core import create_api; api = create_api(); health = api.get_health_status(); exit(0 if health['overall'] == 'healthy' else 1); api.shutdown()\""
        ],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

**Deploy to ECS:**
```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster universal-ai-core-cluster \
  --service-name universal-ai-core-service \
  --task-definition universal-ai-core:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-abcdef],assignPublicIp=ENABLED}"
```

#### Google Cloud Run Deployment

**cloudbuild.yaml:**
```yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/universal-ai-core:latest', '.']
  
  # Push the container image to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/universal-ai-core:latest']
  
  # Deploy container image to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
    - 'run'
    - 'deploy'
    - 'universal-ai-core'
    - '--image'
    - 'gcr.io/$PROJECT_ID/universal-ai-core:latest'
    - '--region'
    - 'us-central1'
    - '--platform'
    - 'managed'
    - '--allow-unauthenticated'
    - '--memory'
    - '2Gi'
    - '--cpu'
    - '2'
    - '--set-env-vars'
    - 'UNIVERSAL_AI_MAX_WORKERS=8,UNIVERSAL_AI_ENABLE_MONITORING=true'

images:
- gcr.io/$PROJECT_ID/universal-ai-core:latest
```

**Deploy:**
```bash
# Submit build
gcloud builds submit --config cloudbuild.yaml .

# Or direct deployment
gcloud run deploy universal-ai-core \
  --image gcr.io/PROJECT_ID/universal-ai-core:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars UNIVERSAL_AI_MAX_WORKERS=8,UNIVERSAL_AI_ENABLE_MONITORING=true
```

#### Azure Container Instances

**deployment-template.json:**
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "containerGroupName": {
      "type": "string",
      "defaultValue": "universal-ai-core-group"
    },
    "containerName": {
      "type": "string",
      "defaultValue": "universal-ai-core"
    },
    "image": {
      "type": "string",
      "defaultValue": "your-registry.azurecr.io/universal-ai-core:latest"
    }
  },
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2019-12-01",
      "name": "[parameters('containerGroupName')]",
      "location": "[resourceGroup().location]",
      "properties": {
        "containers": [
          {
            "name": "[parameters('containerName')]",
            "properties": {
              "image": "[parameters('image')]",
              "ports": [
                {
                  "port": 8000,
                  "protocol": "TCP"
                }
              ],
              "environmentVariables": [
                {
                  "name": "UNIVERSAL_AI_MAX_WORKERS",
                  "value": "8"
                },
                {
                  "name": "UNIVERSAL_AI_ENABLE_MONITORING",
                  "value": "true"
                }
              ],
              "resources": {
                "requests": {
                  "cpu": 2,
                  "memoryInGB": 4
                }
              }
            }
          }
        ],
        "osType": "Linux",
        "ipAddress": {
          "type": "Public",
          "ports": [
            {
              "port": 8000,
              "protocol": "TCP"
            }
          ]
        },
        "restartPolicy": "Always"
      }
    }
  ]
}
```

## Production Deployment Best Practices

### 1. Security Configuration

#### Environment Variables
```bash
# Production environment variables
export UNIVERSAL_AI_MAX_WORKERS=16
export UNIVERSAL_AI_ENABLE_MONITORING=true
export UNIVERSAL_AI_LOG_LEVEL=INFO
export UNIVERSAL_AI_CACHE_SIZE=50000
export UNIVERSAL_AI_ENABLE_SAFETY_CHECKS=true
export UNIVERSAL_AI_RATE_LIMIT_REQUESTS_PER_MINUTE=2000

# Security settings
export UNIVERSAL_AI_API_KEY=your-secure-api-key
export UNIVERSAL_AI_ALLOWED_ORIGINS=https://your-domain.com
export UNIVERSAL_AI_ENABLE_CORS=true
```

#### SSL/TLS Configuration
```yaml
# config/production.yaml
api:
  ssl_enabled: true
  ssl_cert_path: "/etc/ssl/certs/universal-ai-core.crt"
  ssl_key_path: "/etc/ssl/private/universal-ai-core.key"
  require_https: true
  cors_origins:
    - "https://your-domain.com"
    - "https://app.your-domain.com"
```

### 2. Monitoring and Logging

#### Structured Logging
```yaml
# config/production.yaml
logging:
  level: "INFO"
  format: "json"
  handlers:
    - type: "file"
      filename: "/var/log/universal-ai-core/app.log"
      max_bytes: 10485760  # 10MB
      backup_count: 5
    - type: "syslog"
      facility: "local0"
  fields:
    service: "universal-ai-core"
    version: "1.0.0"
    environment: "production"
```

#### Metrics and Monitoring
```python
# monitoring/prometheus_exporter.py
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from universal_ai_core import create_production_api

# Metrics
request_counter = Counter('universal_ai_requests_total', 'Total requests')
request_duration = Histogram('universal_ai_request_duration_seconds', 'Request duration')
active_workers = Gauge('universal_ai_active_workers', 'Active workers')
cache_hit_rate = Gauge('universal_ai_cache_hit_rate', 'Cache hit rate')

def start_monitoring():
    # Start Prometheus metrics server
    start_http_server(9090)
    
    # Create API with monitoring
    api = create_production_api()
    
    # Update metrics periodically
    import threading
    import time
    
    def update_metrics():
        while True:
            try:
                metrics = api.get_metrics()
                active_workers.set(metrics.get('active_workers', 0))
                cache_hit_rate.set(metrics.get('cache_hit_rate', 0))
            except Exception:
                pass
            time.sleep(30)
    
    thread = threading.Thread(target=update_metrics, daemon=True)
    thread.start()
    
    return api

if __name__ == "__main__":
    api = start_monitoring()
    print("Monitoring started on port 9090")
```

### 3. Database Integration

#### PostgreSQL Configuration
```python
# config/database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

DATABASE_CONFIG = {
    "postgresql": {
        "url": "postgresql://user:password@localhost/universal_ai_core",
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_recycle": 3600
    }
}

def create_database_engine():
    return create_engine(
        DATABASE_CONFIG["postgresql"]["url"],
        poolclass=QueuePool,
        pool_size=DATABASE_CONFIG["postgresql"]["pool_size"],
        max_overflow=DATABASE_CONFIG["postgresql"]["max_overflow"],
        pool_timeout=DATABASE_CONFIG["postgresql"]["pool_timeout"],
        pool_recycle=DATABASE_CONFIG["postgresql"]["pool_recycle"]
    )
```

#### Redis Integration
```python
# config/cache.py
import redis
from universal_ai_core.config import get_config

def create_redis_client():
    config = get_config()
    
    return redis.Redis(
        host=config.get("redis_host", "localhost"),
        port=config.get("redis_port", 6379),
        db=config.get("redis_db", 0),
        password=config.get("redis_password"),
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
        health_check_interval=30
    )
```

### 4. High Availability Setup

#### Load Balancer Configuration (NGINX)
```nginx
upstream universal_ai_core {
    least_conn;
    server 10.0.1.10:8000 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8000 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.universal-ai-core.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.universal-ai-core.com;
    
    ssl_certificate /etc/ssl/certs/universal-ai-core.crt;
    ssl_certificate_key /etc/ssl/private/universal-ai-core.key;
    
    location / {
        proxy_pass http://universal_ai_core;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        proxy_buffering on;
        proxy_buffer_size 8k;
        proxy_buffers 8 8k;
    }
    
    location /health {
        proxy_pass http://universal_ai_core/health;
        access_log off;
    }
}
```

#### Health Checks
```python
# health/health_checker.py
import asyncio
import aiohttp
from typing import Dict, List

class HealthChecker:
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
    
    async def check_endpoint(self, session: aiohttp.ClientSession, endpoint: str) -> Dict:
        try:
            async with session.get(f"{endpoint}/health", timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"endpoint": endpoint, "status": "healthy", "response": data}
                else:
                    return {"endpoint": endpoint, "status": "unhealthy", "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"endpoint": endpoint, "status": "unhealthy", "error": str(e)}
    
    async def check_all_endpoints(self) -> Dict:
        async with aiohttp.ClientSession() as session:
            tasks = [self.check_endpoint(session, endpoint) for endpoint in self.endpoints]
            results = await asyncio.gather(*tasks)
            
            healthy_count = sum(1 for r in results if r["status"] == "healthy")
            
            return {
                "overall_health": "healthy" if healthy_count > 0 else "unhealthy",
                "healthy_endpoints": healthy_count,
                "total_endpoints": len(self.endpoints),
                "endpoints": results
            }

# Usage
async def main():
    checker = HealthChecker([
        "http://10.0.1.10:8000",
        "http://10.0.1.11:8000",
        "http://10.0.1.12:8000"
    ])
    
    health_status = await checker.check_all_endpoints()
    print(health_status)

if __name__ == "__main__":
    asyncio.run(main())
```

## Deployment Automation

### 1. CI/CD Pipeline (GitHub Actions)

**.github/workflows/deploy.yml:**
```yaml
name: Deploy Universal AI Core

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run tests
      run: |
        pytest tests/ --cov=universal_ai_core --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ghcr.io/your-org/universal-ai-core:latest
          ghcr.io/your-org/universal-ai-core:${{ github.sha }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to Kubernetes
      uses: azure/k8s-deploy@v1
      with:
        manifests: |
          k8s/deployment.yaml
          k8s/service.yaml
        images: |
          ghcr.io/your-org/universal-ai-core:${{ github.sha }}
        kubectl-version: 'latest'
```

### 2. Infrastructure as Code (Terraform)

**main.tf:**
```hcl
provider "aws" {
  region = var.aws_region
}

# VPC Configuration
resource "aws_vpc" "universal_ai_core_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "universal-ai-core-vpc"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "universal_ai_core_cluster" {
  name = "universal-ai-core"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Application Load Balancer
resource "aws_lb" "universal_ai_core_alb" {
  name               = "universal-ai-core-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets           = aws_subnet.public_subnets[*].id

  enable_deletion_protection = false

  tags = {
    Name = "universal-ai-core-alb"
  }
}

# ECS Service
resource "aws_ecs_service" "universal_ai_core_service" {
  name            = "universal-ai-core"
  cluster         = aws_ecs_cluster.universal_ai_core_cluster.id
  task_definition = aws_ecs_task_definition.universal_ai_core_task.arn
  desired_count   = 3
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_sg.id]
    subnets         = aws_subnet.private_subnets[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.universal_ai_core_tg.arn
    container_name   = "universal-ai-core"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.universal_ai_core_listener]
}

variables.tf:
```hcl
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}
```

## Troubleshooting Deployment Issues

### Common Issues and Solutions

1. **Container Startup Failures**
   ```bash
   # Check container logs
   docker logs universal-ai-core
   
   # Debug container
   docker run -it universal-ai-core:latest /bin/bash
   ```

2. **Memory Issues**
   ```yaml
   # Increase memory limits in deployment
   resources:
     limits:
       memory: "4Gi"
     requests:
       memory: "2Gi"
   ```

3. **Plugin Loading Errors**
   ```bash
   # Check plugin availability
   python -c "
   from universal_ai_core import create_api
   api = create_api()
   plugins = api.core.plugin_manager.list_available_plugins()
   print('Available plugins:', plugins)
   "
   ```

4. **Database Connection Issues**
   ```python
   # Test database connection
   from universal_ai_core.config import get_config
   import psycopg2
   
   config = get_config()
   try:
       conn = psycopg2.connect(config['database_url'])
       print("Database connection successful")
   except Exception as e:
       print(f"Database connection failed: {e}")
   ```

## Deployment Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] Security review completed
- [ ] Performance benchmarks validated
- [ ] Configuration reviewed
- [ ] Backup procedures tested
- [ ] Monitoring setup verified

### Deployment
- [ ] Database migrations applied
- [ ] Configuration files updated
- [ ] Container images built and pushed
- [ ] Services deployed
- [ ] Load balancer configured
- [ ] SSL certificates installed

### Post-Deployment
- [ ] Health checks passing
- [ ] Monitoring alerts configured
- [ ] Performance metrics within range
- [ ] Security scans completed
- [ ] Documentation updated
- [ ] Team notified

## Conclusion

Universal AI Core provides flexible deployment options from development to enterprise production environments. The containerized architecture ensures consistency across platforms, while the monitoring and configuration systems provide enterprise-grade operational capabilities adapted from Saraphis deployment patterns.

For support with deployment issues, refer to the troubleshooting guide or contact the development team.