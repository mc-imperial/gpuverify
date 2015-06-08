FROM ubuntu:14.04
MAINTAINER Dan Liew <daniel.liew@imperial.ac.uk>

ENV BUGLE_REV=a214f90357120debe2de622484a8307932b6c6ee \
    LIBCLC_REV=237229 \
    LIBCLC_SVN_URL=http://llvm.org/svn/llvm-project/libclc/trunk \
    LLVM_VERSION=3.6 \
    CONTAINER_USER=gv

# Get keys, add repos and update apt-cache
RUN apt-get update && apt-get -y install wget && \
    wget -O - http://llvm.org/apt/llvm-snapshot.gpg.key|apt-key add - && \
    echo "deb http://llvm.org/apt/trusty/ llvm-toolchain-trusty-${LLVM_VERSION} main" > /etc/apt/sources.list.d/llvm.list && \
    apt-key adv --recv-keys --keyserver keyserver.ubuntu.com C504E590 && \
    echo 'deb http://ppa.launchpad.net/delcypher/gpuverify-smt/ubuntu trusty main' > /etc/apt/sources.list.d/smt.list && \
    wget -O - http://download.mono-project.com/repo/xamarin.gpg |apt-key add - && \
    echo "deb http://download.mono-project.com/repo/debian wheezy main" > /etc/apt/sources.list.d/mono-xamarin.list && \
    apt-get update

# Setup LLVM, Clang
RUN apt-get -y install llvm-${LLVM_VERSION} llvm-${LLVM_VERSION}-dev llvm-${LLVM_VERSION}-tools clang-${LLVM_VERSION} libclang-${LLVM_VERSION}-dev && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${LLVM_VERSION} 10 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${LLVM_VERSION} 10 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/clang 50 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 50 && \
    update-alternatives --install /usr/bin/opt opt /usr/bin/opt-${LLVM_VERSION} 50 && \
    update-alternatives --install /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-${LLVM_VERSION} 50

# Setup Mono
RUN apt-get -y install mono-xbuild \
                       libmono-microsoft-build-tasks-v4.0-4.0-cil \
                       mono-dmcs libmono-system-numerics4.0-cil \
                       libmono-system-windows4.0-cil \
                       libmono-corlib4.0-cil

# Setup Python
RUN apt-get -y --no-install-recommends install python python-dev python-pip && \
    ln -s /usr/bin/clang /usr/bin/x86_64-linux-gnu-gcc && \
    pip install psutil flask tornado pyyaml


# Setup Z3
RUN apt-get -y install z3=4.3.2-0~trusty1

# Install Other tools needed for build
RUN apt-get -y --no-install-recommends install cmake zlib1g-dev zlib1g git subversion make libedit-dev vim

# Add a non-root user
RUN useradd -m ${CONTAINER_USER}
USER ${CONTAINER_USER}
WORKDIR /home/${CONTAINER_USER}

# Build Bugle
RUN mkdir bugle && cd bugle && mkdir build && \
    git clone git://github.com/mc-imperial/bugle.git src && \
    cd src/ && git checkout ${BUGLE_REV} && cd ../ && \
    cd build && \
    cmake -DLLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config-${LLVM_VERSION} ../src && \
    make


# Libclc
RUN mkdir libclc && \
    cd libclc && \
    mkdir install && \
    svn co -r ${LIBCLC_REV} ${LIBCLC_SVN_URL} srcbuild && \
    cd srcbuild && \
    ./configure.py --with-llvm-config=/usr/bin/llvm-config-${LLVM_VERSION} --prefix=/home/${CONTAINER_USER}/libclc/install nvptx-- nvptx64-- && \
    make && \
    make install

# Put GPUVerify source code into the image and fix permissions
RUN mkdir gpuverify
ADD / /home/${CONTAINER_USER}/gpuverify/
ADD Docker/z3.gvfindtools.py /home/${CONTAINER_USER}/gpuverify/gvfindtools.py
USER root
RUN chown --recursive ${CONTAINER_USER}: /home/${CONTAINER_USER}/gpuverify
USER ${CONTAINER_USER}

# Build GPUVerify C# components
RUN cd gpuverify && \
    xbuild GPUVerify.sln && \
    ln -s /usr/bin/z3 Binaries/z3.exe

# Put GPUVerify in PATH
RUN echo 'PATH=/home/${CONTAINER_USER}/gpuverify:$PATH' >> /home/${CONTAINER_USER}/.bashrc
