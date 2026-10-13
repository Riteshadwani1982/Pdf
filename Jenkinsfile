
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
    }

    stages {
        stage('Select Deployment Environment') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'main') {
                        env.TARGET_ENV = 'prod'
                        env.ECS_SERVICE = 'pdf-bot-service-rpegt7y1'
                        env.TASK_FAMILY = 'pdf-bot'
                        env.CONTAINER_NAME = 'Main'
                    } else if (env.BRANCH_NAME == 'Development') {
                        env.TARGET_ENV = 'dev'
                        env.ECS_SERVICE = 'pdf-bot-dev-service'
                        env.TASK_FAMILY = 'pdf-bot-dev'
                        env.CONTAINER_NAME = 'Pdf-bot-dev'
                    } else {
                        error(
                            "Deployment is not configured for branch: " +
                            "${env.BRANCH_NAME}"
                        )
                    }

                    env.ECR_URI =
                        "${env.AWS_ACCOUNT_ID}.dkr.ecr." +
                        "${env.AWS_REGION}.amazonaws.com/" +
                        "${env.ECR_REPOSITORY}"

                    echo "Branch: ${env.BRANCH_NAME}"
                    echo "Environment: ${env.TARGET_ENV}"
                    echo "ECS Cluster: ${env.ECS_CLUSTER}"
                    echo "ECS Service: ${env.ECS_SERVICE}"
                    echo "Task Family: ${env.TASK_FAMILY}"
                }
            }
        }

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Prepare Image Tag') {
            steps {
                script {
                    def commitSha = sh(
                        script: 'git rev-parse --short=12 HEAD',
                        returnStdout: true
                    ).trim()

                    env.IMAGE_TAG = "${env.TARGET_ENV}-${commitSha}"

                    env.IMAGE_URI =
                        "${env.ECR_URI}:${env.IMAGE_TAG}"

                    echo "Image tag: ${env.IMAGE_TAG}"
                    echo "Image URI: ${env.IMAGE_URI}"
                }
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

                    aws ecr get-login-password \
                        --region "$AWS_REGION" |
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

with open("task-definition-current.json", encoding="utf-8") as f:
    task = json.load(f)

# Remove AWS-generated fields before registering a new revision.
for key in (
    "taskDefinitionArn",
    "revision",
    "status",
    "requiresAttributes",
    "compatibilities",
    "registeredAt",
    "registeredBy",
    "deregisteredAt"
):
    task.pop(key, None)

container_name = os.environ["CONTAINER_NAME"]
image_uri = os.environ["IMAGE_URI"]

containers = task.get("containerDefinitions", [])
matches = [c for c in containers if c["name"] == container_name]

# Fall back only if the task definition has exactly one container.
if not matches and len(containers) == 1:
    matches = containers

if len(matches) != 1:
    names = [c["name"] for c in containers]
    raise SystemExit(
        f"Cannot uniquely identify container '{container_name}'. "
        f"Available containers: {names}"
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

                    echo "Registered task definition:"
                    cat registered-task-arn.txt
                '''
            }
        }

        stage('Deploy to ECS') {
            steps {
                sh '''
                    set -eu

                    TASK_ARN="$(cat registered-task-arn.txt)"

                    echo "Deploying $IMAGE_URI"
                    echo "To service $ECS_SERVICE"
                    echo "Using task definition $TASK_ARN"

                    aws ecs update-service \
                        --cluster "$ECS_CLUSTER" \
                        --service "$ECS_SERVICE" \
                        --task-definition "$TASK_ARN" \
                        --region "$AWS_REGION"

                    aws ecs wait services-stable \
                        --cluster "$ECS_CLUSTER" \
                        --services "$ECS_SERVICE" \
                        --region "$AWS_REGION"

                    echo "ECS service is stable."
                '''
            }
        }
    }

    post {
        success {
            echo "Deployment to ${env.TARGET_ENV} completed successfully."
        }

        failure {
            echo "Pipeline failed for branch ${env.BRANCH_NAME}. Review the stage logs."
        }

        always {
            sh '''
                rm -f task-definition-current.json \
                      task-definition-new.json \
                      registered-task-arn.txt || true
            '''
        }
    }
}
