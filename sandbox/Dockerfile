# The sandbox image for container mode: Python with numpy and CPU-only torch,
# nothing else. Recipes run in it as nobody, with no network and a read-only
# root filesystem; only the sandbox folder is mounted.
#
# Build it with `python -m core.sandbox build`, which passes the versions
# pinned in requirements.txt and this machine's Python version.
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

ARG NUMPY_VERSION
ARG TORCH_VERSION
RUN pip install --no-cache-dir "numpy==${NUMPY_VERSION}" \
 && pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu \
 && rm -rf /root/.cache

WORKDIR /box
