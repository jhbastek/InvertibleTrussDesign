import numpy as np
import pandas as pd


class Topology:
    def __init__(self):
        self.connectity = []
        self.coordinates = []
        # self.lattice_type1 = lattice['lattice_type1']
        # self.lattice_rep1 = lattice['lattice_rep1']
        # self.lattice_type2 = lattice['lattice_type2']
        # self.lattice_rep2 = lattice['lattice_rep2']
        # self.lattice_type3 = lattice['lattice_type3']
        # self.lattice_rep3 = lattice['lattice_rep3']

    def create_lattice(self,lattice):
        
        conn1, coord1 = self.create_individual_lattice(lattice['lattice_type1'],lattice['lattice_rep1'])
        conn2, coord2 = self.create_individual_lattice(lattice['lattice_type2'],lattice['lattice_rep2'])
        conn3, coord3 = self.create_individual_lattice(lattice['lattice_type3'],lattice['lattice_rep3'])

        compound_lattice_conn, compound_lattice_coord = self.create_compound_lattice(conn1,conn2,conn3,coord1,coord2,coord3)

        final_lattice_coord = self.affinely_deform_lattice(compound_lattice_coord,lattice)

        return compound_lattice_conn, final_lattice_coord


    def create_individual_lattice(self,lattice_index,lattice_rep):
        conn, coord = self.get_topology(lattice_index)
        temp_conn = conn
        temp_coord = coord
        UC_nodes = len(coord)
        if lattice_rep == 2:
            for i in range(3):
                temp_coord[:,i] = temp_coord[:,i]+1
                coord = np.concatenate(coord,np.concatenate((temp_coord[:,0],temp_coord[:,1],temp_coord[:,2]),axis=1))
                conn = np.concatenate(conn,temp_conn+UC_nodes*2**i)
                temp_conn = conn
                temp_coord = coord
        return conn, coord

    def create_compound_lattice(self,conn1,conn2,conn3,coord1,coord2,coord3):
        tot_conn = np.concatenate(conn1,conn2+len(coord1),conn3+len(coord1)+len(coord2))
        tot_coord = np.concatenate(coord1,coord2,coord3)

        tot_nodes = len(tot_coord)
        red_conn = tot_conn
        duplicate_nodes = []

        tol = 1.e-5

        for i in range(tot_nodes):
            for j in range((i+1), tot_nodes):
                if np.norm(tot_coord[i,:] - tot_coord[j,:]) < tol:
                    red_conn[red_conn==j] = i
                    duplicate_nodes = duplicate_nodes.append(j)
        
        if duplicate_nodes:
            duplicate_nodes = np.unique(duplicate_nodes)
            duplicate_nodes = np.sort(duplicate_nodes)[::-1]
            for i in duplicate_nodes:
                red_conn[red_conn>i] = red_conn[red_conn>i] - 1
        
        # delete duplicate nodes
        red_coord = np.unique(tot_coord,axis=0)
        
        # delete duplicate connectivities
        for i in range(len(red_conn)):
            if red_conn(i,0)>red_conn(i,1):
                temp = red_conn[i,1]
                red_conn[i,1] = red_conn[i,0]
                red_conn[i,0] = temp

        red_conn = np.unique(red_conn,axis=0)

        return red_conn, red_coord






    def get_topology(lattice_type):
        if lattice_type == 0:
            connectivity = np.array([0, 1],[1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6],[3, 7])
            coordinates = np.array([0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[1.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[1.0, 1.0, 1.0],[0.0, 1.0, 1.0])
        elif lattice_type == 1:
            connectivity = np.array([0, 12],[1, 12],[2, 12],[3, 12],[4, 13],[5, 13],[6, 13],[7, 13],[0, 10],[3, 10],[4, 10],[7, 10],[1, 11],[2, 11],[5, 11],[6, 11],[0, 8],[1, 8],
                [4, 8],[5, 8],[2, 9],[3, 9],[6, 9],[7, 9],[8, 10],[8, 11],[8, 12],[8, 13],[9, 10],[9, 11],[9, 12],[9, 13],[10, 12],[10, 13],[11, 12],[11, 13])
            coordinates = np.array([0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[1.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],[0.5, 0.0, 0.5],[0.5, 1.0, 0.5],[0.0, 0.5, 0.5],[1.0, 0.5, 0.5],[0.5, 0.5, 0.0],[0.5, 0.5, 1.0])
        elif lattice_type == 2:
            connectivity = np.array([0, 8],[1, 8],[2, 8],[3, 8],[4, 8],[5, 8],[6, 8],[7, 8])
            coordinates = np.array([0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[1.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[1.0, 1.0, 1.0],[0.0, 1.0, 1.0],[0.5, 0.5, 0.5])
        elif lattice_type == 3:
            connectivity = np.array([0,4],[0,5],[1,4],[1,6],[2,4],[2,7],[3,4],[3,8],[4,5],[4,6],[4,7],[4,8])
            coordinates = np.array([0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[1.0,1.0,0.0],[0.5,0.5,0.5],[0.0,0.0,1.0],[1.0,0.0,1.0],[0.0,1.0,1.0],[1.0,1.0,1.0])
        elif lattice_type == 4:
            connectivity = np.array([0,1],[1,2],[2,3],[3,0],[0,4],[1,4],[2,4],[3,4],[4,5],[4,6],[4,7],[4,8],[5,6],[6,7],[7,8],[8,5])
            coordinates = np.array([0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0],[0.5,0.5,0.5],[0.0,1.0,1.0],[0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0])
        elif lattice_type == 5:
            connectivity = np.array([0,3],[3,1],[1,4],[4,2],[3,5],[3,6],[4,7],[4,8],[5,9],[6,9],[7,10],[8,10],[9,11],[9,12],[10,12],[10,13])
            coordinates = np.array([0.0,0.0,0.0],[0.5,0.5,0.0],[1.0,1.0,0.0],[0.25,0.25,0.25],[0.75,0.75,0.25],[0.0,0.5,0.5],[0.5,0.0,0.5],[0.5,1.0,0.5],
                [1.0,0.5,0.5],[0.25,0.25,0.75],[0.75,0.75,0.75],[0.0,0.0,1.0],[0.5,0.5,1.0],[1.0,1.0,1.0])
        elif lattice_type ==6:
            connectivity = np.array([0,2],[1,3],[2,4],[3,4],[4,5],[5,6],[5,7],[6,8],[7,9],[10,12],[11,13],[12,14],[13,14],[14,15],[15,16],[15,17],[16,18],[17,19],
                [2,12],[3,13],[4,14],[5,15],[6,16],[7,17])
            coordinates = np.array([0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,0.166666,0.0],[1.0,0.166666,0.0],[0.5,0.333333,0.0],[0.5,0.666666,0.0],[0.0,0.833333,0.0],
                [1.0,0.833333,0.0],[0.0,1.0,0.0],[1.0,1.0,0.0],[0.0,0.0,1.0],[1.0,0.0,1.0],[0.0,0.166666,1.0],[1.0,0.166666,1.0],[0.5,0.333333,1.0],[0.5,0.666666,1.0],
                [0.0,0.833333,1.0],[1.0,0.833333,1.0],[0.0,1.0,1.0],[1.0,1.0,1.0])
        return connectivity, coordinates