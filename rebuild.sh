sudo pip uninstall -y protobuf
sudo pip uninstall -y tensorflow
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install /tmp/tensorflow_pkg/tensorflow-0.12.1-cp27-cp27mu-linux_x86_64.whl

