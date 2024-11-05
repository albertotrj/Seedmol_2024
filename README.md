# Seedmol 2024

## Instalação

### [Miniforge](https://github.com/conda-forge/miniforge)

```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
Abra um novo shell

### [DeepMD](https://docs.deepmodeling.com/projects/deepmd/en/r2/install/easy-install.html#install-with-conda)

```
mamba create -n deepmd deepmd-kit=*=*cpu libdeepmd=*=*cpu lammps -c https://conda.deepmodeling.com
mamba activate deepmd
```

## Sampling

Vamos fazer uma dinâmica de um sistema pequeno de água líquida (33 moléculas) usando o código [LAMMPS](https://docs.lammps.org/Manual.html) (Large-scale Atomic/Molecular Massively Parallel Simulator).

Na pasta ```sampling``` você vai encontrar os inputs necessários.

Para rodar, execute:
```
export OMP_NUM_THREADS=4
lmp -in equilibration.in
lmp -in dynamics.in
```
