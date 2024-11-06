#!/usr/bin/env python

import os
import numpy as np
from ase.io import read


filename = 'dynamics.lammpstraj'
frames = read(filename, index=':', format='lammps-dump-text')

# filename = 'positions.lammpstraj'
# frames_coords = read(filename, index=':', format='lammps-dump-text')
# 
# filename = 'forces.lammpstraj'
# frames_forces = read(filename, index=':', format='lammps-dump-text')

frames_energies = np.loadtxt('energies.dat')

if not os.path.isdir('data'):
    os.mkdir('data')
os.chdir('data')

atoms = frames[0]
type_map_dict = {}

print(atoms.symbols.species())

with open('type_map.raw', 'w') as f:
    for i, symbol in enumerate(atoms.symbols.species()):
        f.write('%s\n' % symbol)
        type_map_dict[symbol] = i

print('Types map created:')
print(type_map_dict)


# save deepmd data
with open('type.raw',   'w') as f_type:
    types  = [type_map_dict[symbol] for symbol in frames[0].symbols]
    np.savetxt(f_type, [types], fmt='%d\n')

with open('energy.raw', 'w') as f_energ, \
     open('force.raw',  'w') as f_force, \
     open('coord.raw',  'w') as f_coord, \
     open('box.raw',    'w') as f_box:
    for e, frame in zip(frames_energies, frames):
        # read
        # forces   = frames_forces.get_positions()
        # coords   = frames_coords.get_positions()
        # box      = frames_coords.get_cell()
        forces = frame.get_forces()
        coords = frame.get_positions()
        box    = frame.get_cell()
        
        # write
        f_energ.write('%e\n' % e)
        np.savetxt(f_force, [forces.ravel()])
        np.savetxt(f_coord, [coords.ravel()])
        np.savetxt(f_box, [box.ravel()])
        

