# GPUVerify

[![Build Status](https://travis-ci.org/mc-imperial/gpuverify.svg?branch=master)](https://travis-ci.org/mc-imperial/gpuverify)

GPUVerify is a static analyser for verifying race- and divergence-freedom of
GPU kernels written in OpenCL and CUDA.

The documentation is written using Sphinx and can be found in
```Documentation``` or
[online](http://multicore.doc.ic.ac.uk/tools/GPUVerify/docs/)

## Generating the Documentation

To generate the documentation run:
```
$ cd Documentation
$ make html
```
You can then view ```Documentation/_build/html/index.html``` in a browser.

## Building from source

### Linux

Package names assume you are in the Ubuntu familiy but should be similar in your favourite distribution.

1. Install packages `python python-psutil unzip mono-complete nuget`, where `python` refers to Python 2.
2. Download latest release zip and `unzip GPUVerifyLinux64.zip`; this sould give you a `./2018-03-22` folder
3. Clone git repository and `cd` into it
4. Run `nuget restore GPUVerify.sln`
5. Run `xbuild /p:Configuration=Release GPUVerify.sln`
6. Copy the files you find in the `./Binaries` directory over the ones you got in the release zip's `./bin` folder, e.g. `cp ./Binaries/* ../2018-03-22/bin/`
7. Change into release folder, e.g. `cd ../2018-03-22`
8. Verify build by running tests: `./gvtester.py --write-pickle run.pickle testsuite/`
