# Relocation of `conda` environments

Supercompting clusters may not have access to the internet. 
This makes it difficult to configure `python` environments locally.

`conda-pack` offers a solution for relocating `conda` environments to a new location. 
The full documentation is [here](https://conda.github.io/conda-pack/).

## Installation of `conda-pack`

Install `conda-pack` in a dedicated environment :

    conda create --name conda-pack
    conda activate conda-pack
    conda install conda-pack

# Usage

Let's consider the following environment `pytorch-1.7.1` for relocation

    conda create --name pytorch-1.7.1
    conda activate pytorch-1.7.1
    conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch

Note that `cuda 10.1` should be a version compatible with GPU drivers for target cluster (here `mycluster`). Therefore,
the envrionment is said _relocatable_.

The environment `pytorch-1.7.1` is _packed_ using the command  

    conda pack -n pytorch-1.7.1

An archive `pytorch-1.7.1.tar.gz` is ceated and have to be copy on `mycluster` 

    scp pytorch-1.7.1.tar.gz <login>@mycluster:~

To use this relocated environment on `mycluster`, untar the archive and just load the environment

    mkdir pytorch-1.7.1
    tar -zxf pytorch-1.7.1.tar.gz -C pytorch-1.7.1
    source pytorch-1.7.1/bin/activate

> Use `bash` shell environment.

That's it.
