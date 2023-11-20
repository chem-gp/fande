# Helper to convert i-PI style extended xyz to ASE style extended xyz/traj files


import numpy as np                                                                                                                                                                                  
from ase.io import read,write  


def ipi2ase(infile, outfile=None, format='extxyz', index=':', **kwargs):

        if outfile is None:
                outfile = infile[:-4] + '_ASE.xyz'
        traj = read(infile, index=index)          
        print("File has been read.")                                                                                                                                                    
        cells = [] 
        with open(infile) as f: 
                for il,l in enumerate(f): 
                        if '#' in l: 
                                c = l.strip().split(' ') 
                                c = np.array([x for x in c if x.strip()][2:8]) 
                                c = np.array([float(x) for x in c]) 
                                cells.append(c) 

        print("Cells info has been read.")
        for ifrm,frm in enumerate(traj): 
                frm.set_pbc(True) 
                cell = cells[ifrm] 
                frm.set_cell(cell) 
                pos = frm.get_scaled_positions() 
                write(outfile, traj, format=format)

        print("Output file has been written.")

        return traj