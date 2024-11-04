# Seedmol 2024

## Instalação

### [Miniforge](https://github.com/conda-forge/miniforge)

```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
Abra um novo shell

### [DeepMD](https://docs.deepmodeling.com/projects/deepmd/en/r2/install/easy-install.html#install-off-line-packages)

```
mamba create -n deepmd deepmd-kit=*=*cpu libdeepmd=*=*cpu lammps -c https://conda.deepmodeling.com
mamba activate deepmd
```

## Sampling
