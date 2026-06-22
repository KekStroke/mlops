"""
Script to set up ClearML Serving endpoint for the spam classifier.

Run AFTER train.py has created and registered the model.

Usage:
    python deploy_serving.py --model-id <MODEL_ID>
    python deploy_serving.py --model-id <MODEL_ID> --queue default
"""

import argparse

from clearml import Task


TASK_NAME = "sms_spam_serving"
PROJECT_NAME = "Course MLOps"
PREPROCESS_FILE = "serving/preprocess.py"
ENDPOINT_NAME = "sms_spam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy ClearML Serving endpoint.")
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help='ClearML model ID (shown at end of train.py output)',
    )
    parser.add_argument(
        "--queue",
        type=str,
        default="default",
        help='Serve queue name (default: "default")',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    serving_task = Task.init(
        project_name=PROJECT_NAME,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.data_processing,
    )

    model_endpoints = serving_task.set_model_config(
        config=dict(
            endpoint_name=ENDPOINT_NAME,
            model_id=args.model_id,
            preprocess_package=PREPROCESS_FILE,
        ),
    )

    serving_task.execute_remotely(queue_name=args.queue)

    print(f"Serving endpoint '{ENDPOINT_NAME}' deployed.")
    print(f"Model ID: {args.model_id}")
    print(f"Endpoint URL: http://127.0.0.1:8085/serve/{ENDPOINT_NAME}")


if __name__ == "__main__":
    main()
