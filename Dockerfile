FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y g++ cmake build-essential && \
    apt-get clean

WORKDIR /app

COPY ./tensor ./tensor
COPY ./sample ./sample
COPY ./CMakeLists.txt ./CMakeLists.txt

RUN mkdir build && \
    cd ./build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=../install && \
    cmake --build . --target=install --config=Release

CMD ["./install/bin/sample"]