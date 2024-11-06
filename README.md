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
mamba create -n deepmd deepmd-kit lammps horovod
mamba activate deepmd
mamba install ase
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

Para extrair os dados no formato apropriado para o DeepMD, execute:
```
chmod +x ./lammps_to_deepmd.py
./lammps_to_deepmd.py
curl -H 'Accept: application/vnd.github.v3.raw' -O -L https://api.github.com/repos/deepmodeling/deepmd-kit/contents/data/raw/raw_to_set.sh
chmod +x ./raw_to_set.sh
raw_to_set.sh 2700
cd data/
mkdir train test
mv set.000 train/
mv set.001 test/
cp type*.raw train/
cp type*.raw test/
cd ..
```

## Training

Vamos treinar uma rede neural com os dados que temos.

O input para o DeepMD está no arquivo ```water.json```.

Ajuste o script ```run.bsh```.

```
cd train
chmod +x ./run.bsh
./run.bsh
```

Abra outro shell e monitore o arquivo ```lcurve.out```.

Você pode plotar a convergência do treino executando
```
./plot_training.py --epoch 2700
```
