#!/bin/bash bash

LAMBDA_DIR=$(cd `dirname $0` && pwd)
BASE_DIR=$(dirname $LAMBDA_DIR)
docker run -it --name lambda-container -v $BASE_DIR:/host -w /host --rm lambci/lambda:build-python3.6 bash lambda/build_pack.sh