import boto3
import os
import pickle
import json
import numpy as np
import torch


def lambda_handler(event, context):
    print('event:', event)

    # return results formatted for AWS API Gateway
    result = {"statusCode": 200,
              "headers": {"Content-Type": "application/json"},
              "body": torch.tensor([1,2,3]).sum().item()}
    return result