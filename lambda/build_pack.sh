export LAMBDA_DIR=$(cd `dirname $0` && pwd)
export BASE_DIR=$(dirname $LAMBDA_DIR)
export LAMBDAPACK=$LAMBDA_DIR/lambdapack


virtualize_environment () {
    rm -rf $LAMBDA_DIR/env
    python3 -m venv $LAMBDA_DIR/env
    source $LAMBDA_DIR/env/bin/activate
}

install_packages () {
    pip3 install numpy
    pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
    pip3 install boto3
}

gather_packages () {
    rm -rf $LAMBDAPACK
    mkdir $LAMBDAPACK

    cd $LAMBDAPACK

    # Copy python packages from virtual environment
    cp -R $LAMBDA_DIR/env/lib/python3.6/site-packages/* .
    cp -R $LAMBDA_DIR/env/lib64/python3.6/site-packages/* .

    echo "Original size $(du -sh . | cut -f1)"
}

gather_modules () {
    cp $BASE_DIR/lambda_function.py .
    cp -R $BASE_DIR/datasets .
    cp -R $BASE_DIR/models .
    cp -R $BASE_DIR/generators .
    cp $BASE_DIR/dictionaries.py .
    cp $LAMBDA_DIR/test_lambda.py .
    # Upload parameters onto S3
}

test_lambda () {
    python3 test_lambda.py

}

minimize_pack () {
    rm -r ./botocore
    rm -r ./botocore-*
    find . -type d -name "tests" -exec rm -rf {} +
    find . -name "*.so" | xargs strip
    find . -name "*.so.*" | xargs strip
    rm -r ./pip
    rm -r ./pip-*
    rm -r ./wheel
    rm -r ./wheel-*
    rm ./easy_install.py
    find . -name \*.pyc -delete

    echo "Stripped size $(du -sh . | cut -f1)"
}

compress_pack () {
    zip -FS -r1 $LAMBDA_DIR/pack.zip * > /dev/null
    echo "Compressed size $(du -sh $LAMBDA_DIR/pack.zip | cut -f1)"
}

main () {
    virtualize_environment
    install_packages
    gather_packages
    gather_modules
    test_lambda
    minimize_pack
    compress_pack
}

main