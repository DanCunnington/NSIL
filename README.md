# NSIL: Neuro-Symbolic Learning of Answer Set Programs from Raw Data
This repository is associated with the [paper](https://arxiv.org/abs/2205.12735) "Neuro-Symbolic Learning of Answer Set Programs from Raw Data", published at IJCAI 2023. Please consider [citing](#citation) if it is useful in your work. The technical appendix forming our supplementary material is available [here](./TechnicalAppendix.pdf).

## Intel x86 vs. Apple Silicon (Mac M1/M2)
If running on an intel machine, the recommended approach is using [Docker](#docker-installation). For Apple Silicon users, Docker can be used by adding the `--platform linux/amd64` flag to the build and run commands, however, the performance hit is significant. It is recommended to install [natively](#native-unix-installation).

## Docker installation
A docker container is provided for easy setup and installation so the only additional software required is a container runtime such as [Docker Desktop](https://www.docker.com/products/docker-desktop/), [Podman](https://podman.io/), or [minikube](https://minikube.sigs.k8s.io/docs/tutorials/docker_desktop_replacement/).

### Setup
The following commands assume the `docker` command is available on your system. If using a different container run-time, please replace accordingly. From the root directory:

1. `docker build -t nsil:ijcai_2023 .`
2. `docker run -d -p 8000:8000 -p 9990:9990 --name nsil_ijcai_2023 nsil:ijcai_2023`
3. Open [http://localhost:8000](http://localhost:8000) in your web browser to launch a markdown viewer containing detailed documentation that explains how to reproduce the experiment results.

### Running experiments
Once the docker container is running:

1. `docker exec -it nsil_ijcai_2023 /bin/bash`
2. `cd scripts`
3. Run experiments using the commands detailed in the [documentation](http://localhost:8000).

### Stopping the container and removing files
```
exit
docker stop nsil_ijcai_2023
docker rm nsil_ijcai_2023
docker rmi nsil:ijcai_2023
```

## Native Unix Installation

### Pre-requisites
* Python >= 3.9 with Pip
* [Clingo with lua support](https://github.com/potassco/clingo/blob/master/INSTALL.md)
* (Optional) virtualenv or conda.

### Installation
1. Create python environment and install dependencies
```bash
virtualenv nsil_p3
source nsil_p3/bin/activate
pip install -r requirements.txt
```

2. Download image data
```bash
chmod +x ./download_data.sh
./download_data.sh
```

3. Create a setup script with the following contents:
```bash
BASE_PATH=/path/to/NSIL
export PYTHONPATH=$BASE_PATH
cd $BASE_PATH/examples/$1
```
and replace the value of `BASE_PATH` accordingly. Save this file to `paper_experiments/ijcai_2023/scripts/setup.sh` and ensure it has execute permissions. If you also want to run the NeurASP and FF-NSL baseline experiments, save a copy to `paper_experiments/ijcai_2023/scripts/naive_baselines/setup.sh`, again ensuring executable permissions.

4. Move ILASP and FastLAS binaries to the root directory. We provide apple silicon and intel ubuntu linux binaries, copy accordingly.
```bash
mv LAS_binaries/ILASP_apple_silicon ILASP
mv LAS_binaries/FastLAS_apple_silicon FastLAS
```

5. Set the python path to the root directory
```bash
export PYTHONPATH=/path/to/NSIL
```

6. To view documentation:
```bash
cd paper_experiments/ijcai_2023/
chmod +x ./start_web_servers.sh
./start_web_servers.sh
```

7. To run experiments:
```bash
cd paper_experiments/ijcai_2023/scripts
find . -type f -exec chmod +x {} \;
./run_arithmetic_repeats.sh -p 100 -s 0 -m 5 # For example - see documentation for more commands.
```

## Citation
```
TBC
```
