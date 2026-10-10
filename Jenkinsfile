
pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    
environment {
    AWS_REGION       = 'us-east-1'
    AWS_ACCOUNT_ID   = '692859921402'
    ECR_REPOSITORY   = 'pdf-bot'
    ECS_CLUSTER      = 'pdf-bot'
    ECS_SERVICE      = 'pdf-bot-service-rpegt7y1'
    TASK_FAMILY      = 'pdf-bot'
    CONTAINER_NAME   = 'Main'
}

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/Riteshadwani1982/Pdf.git'
            }
        }

        stage('Prepare Image Tag') {
            steps {
                script {
                    env.IMAGE_TAG = sh(
                        script: 'git rev-parse --short=12 HEAD',
                        returnStdout: true
                    ).trim()

                    env.ECR_URI =
                        "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"

                    env.IMAGE_URI = "${ECR_URI}:${IMAGE_TAG}"
                }

                echo "Image to build: ${IMAGE_URI}"
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    set -eu
                    docker build --pull -t "$IMAGE_URI" .
                '''
            }
        }

        stage('Login to ECR') {
            steps {
                sh '''
                    set -eu
                    aws ecr get-login-password --region "$AWS_REGION" |
                    docker login \
                        --username AWS \
                        --password-stdin \
                        "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
                '''
            }
        }

        stage('Push Image to ECR') {
            steps {
                sh '''
                    set -eu
                    docker push "$IMAGE_URI"
                '''
            }
        }

        stage('Register ECS Task Definition') {
            steps {
                sh '''
                    set -eu

                    aws ecs describe-task-definition \
                        --task-definition "$TASK_FAMILY" \
                        --region "$AWS_REGION" \
                        --query taskDefinition \
                        --output json > task-definition-current.json

                    python3 - <<'PY'
import json
import os

source = "task-definition-current.json"
with open(source, encoding="utf-8") as f:
    task = json.load(f)

for key in (
    "taskDefinitionArn", "revision", "status",
    "requiresAttributes", "compatibilities",
    "registeredAt", "registeredBy"
):
    task.pop(key, None)

container_name = os.environ["CONTAINER_NAME"]
image_uri = os.environ["IMAGE_URI"]

matches = [
    c for c in task["containerDefinitions"]
    if c["name"] == container_name
]

if len(matches) != 1:
    raise SystemExit(
        f"Expected exactly one container named {container_name}; "
        f"found {len(matches)}"
    )

matches[0]["image"] = image_uri

with open("task-definition-new.json", "w", encoding="utf-8") as f:
    json.dump(task, f)
PY

                    aws ecs register-task-definition \
                        --cli-input-json file://task-definition-new.json \
                        --region "$AWS_REGION" \
                        --query 'taskDefinition.taskDefinitionArn' \
                        --output text > registered-task-arn.txt

                    cat registered-task-arn.txt
                '''
            }
        }

        stage('Deploy to ECS') {
            steps {
                sh '''
                    set -eu

                    TASK_ARN="$(cat registered-task-arn.txt)"

                    aws ecs update-service \
                        --cluster "$ECS_CLUSTER" \
                        --service "$ECS_SERVICE" \
                        --task-definition "$TASK_ARN" \
                        --region "$AWS_REGION" \
                        --query 'service.serviceName' \
                        --output text

                    aws ecs wait services-stable \
                        --cluster "$ECS_CLUSTER" \
                        --services "$ECS_SERVICE" \
                        --region "$AWS_REGION"
                '''
            }
        }
    }

    post {
        success {
            echo 'Deployment completed successfully.'
        }
        failure {
            echo 'Pipeline failed. Review the stage logs before retrying.'
        }
        always {
            sh 'rm -f task-definition-current.json task-definition-new.json registered-task-arn.txt || true'
        }
    }
}
