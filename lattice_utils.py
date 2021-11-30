from abc import abstractclassmethod
import numpy as np
from voigt_rotation import get_rotation_matrix_np
import matplotlib.pyplot as plt

class Topology:
    def __init__(self,lattice):
        self.connectity, self.coordinates, self.diameter = self.create_lattice(lattice)

    def create_lattice(self,lattice):
        conn1, coord1 = self.create_individual_lattice(lattice['lattice_type1'],lattice['lattice_rep1'])
        conn2, coord2 = self.create_individual_lattice(lattice['lattice_type2'],lattice['lattice_rep2'])
        conn3, coord3 = self.create_individual_lattice(lattice['lattice_type3'],lattice['lattice_rep3'])
        compound_lattice_conn, compound_lattice_coord = self.create_compound_lattice(conn1,conn2,conn3,coord1,coord2,coord3)
        diameter = self.compute_diameter(compound_lattice_conn, compound_lattice_coord, lattice)
        final_lattice_coord = self.affinely_deform_lattice(compound_lattice_coord,lattice)
        return compound_lattice_conn, final_lattice_coord, diameter

    def affinely_deform_lattice(self,compound_lattice_coord,lattice):
        affine_def = self.get_affine_deformation(lattice)
        final_lattice_coord = np.linalg.multi_dot([compound_lattice_coord,affine_def.transpose()])
        return final_lattice_coord

    def get_affine_deformation(self,lattice):
        U = np.diag([lattice['U1'],lattice['U2'],lattice['U3']])
        V = np.diag([lattice['V1'],lattice['V2'],lattice['V3']])
        R1 = get_rotation_matrix_np(lattice['R1_theta'],lattice['R1_rot_ax1'],lattice['R1_rot_ax2'])
        R2 = get_rotation_matrix_np(lattice['R2_theta'],lattice['R2_rot_ax1'],lattice['R2_rot_ax2'])
        return np.linalg.multi_dot([R2, V, R1, U])

    def create_individual_lattice(self,lattice_index,lattice_rep):
        conn, coord = self.get_topology(lattice_index)
        temp_conn = conn.copy()
        temp_coord = coord.copy()
        UC_nodes = len(coord)
        if lattice_rep == 2:
            for i in range(3):
                temp_coord[:,i] = temp_coord[:,i]+1
                coord = np.concatenate((coord,np.stack((temp_coord[:,0],temp_coord[:,1],temp_coord[:,2]),axis=1)))
                conn = np.concatenate((conn,temp_conn+UC_nodes*2**i))
                temp_conn = conn.copy()
                temp_coord = coord.copy()
            # shift center to origin
            coord -= np.array([0.5, 0.5, 0.5])
            # normalize lattice
            coord /= 2.
        return conn, coord

    def create_compound_lattice(self,conn1,conn2,conn3,coord1,coord2,coord3):
        tot_conn = np.concatenate((conn1,conn2+len(coord1),conn3+len(coord1)+len(coord2)))
        tot_coord = np.concatenate((coord1,coord2,coord3))

        tot_nodes = len(tot_coord)
        red_conn = tot_conn
        duplicate_nodes = []

        tol = 1.e-6
        for i in range(tot_nodes):
            for j in range((i+1), tot_nodes):
                if np.linalg.norm(tot_coord[i,:] - tot_coord[j,:]) < tol:
                    red_conn[red_conn==j] = i
                    duplicate_nodes.append(j)

        if duplicate_nodes:
            duplicate_nodes = np.unique(duplicate_nodes)
            duplicate_nodes = np.sort(duplicate_nodes)[::-1]
            for i in duplicate_nodes:
                red_conn[red_conn>i] = red_conn[red_conn>i] - 1
        
        # delete duplicate nodes and preserve order
        _, idx = np.unique(tot_coord, axis=0, return_index=True)
        red_coord = tot_coord[np.sort(idx)]

        # delete duplicate connectivities
        for i in range(len(red_conn)):
            if red_conn[i,0]>red_conn[i,1]:
                temp = red_conn[i,1]
                red_conn[i,1] = red_conn[i,0]
                red_conn[i,0] = temp

        red_conn = np.unique(red_conn,axis=0)

        return red_conn, red_coord

    def plotTruss(self):
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.coordinates[:,0],self.coordinates[:,1],self.coordinates[:,2],'ro')
        self.connect_points(ax)
        plt.show()

    def connect_points(self, ax):
        for i in range(len(self.connectity)):
            x1, x2 = self.coordinates[self.connectity[i,0],0], self.coordinates[self.connectity[i,1],0]
            y1, y2 = self.coordinates[self.connectity[i,0],1], self.coordinates[self.connectity[i,1],1]
            z1, z2 = self.coordinates[self.connectity[i,0],2], self.coordinates[self.connectity[i,1],2]
            ax.plot3D([x1,x2],[y1,y2],[z1,z2],'gray')

    def compute_diameter(self,conn,coord,lattice):
        beam_weights = self.generate_beam_weights(conn,coord)
        weighted_length = 0.
        for i in range(len(conn)):
            weighted_length += beam_weights[i]*np.linalg.norm((coord[conn[i,0]],coord[conn[i,1]]))
        affine_def = self.get_affine_deformation(lattice)
        tot_volume = np.linalg.det(affine_def)
        # print(lattice['relative_density'])
        area = lattice['relative_density']*tot_volume / weighted_length 
        diameter = np.sqrt(4.*area/np.pi)
        return diameter

    def generate_beam_weights(self,conn,coord):
        eps = 1.e-6
        beam_weigts = np.ones((len(conn)))
        for i in range(len(conn)):
            for j in range(3):
                if abs(coord[conn[i,0],j]-0.5) < eps and abs(coord[conn[i,1],j]-0.5) < eps or\
                abs(coord[conn[i,0],j]+0.5) < eps and abs(coord[conn[i,1],j]+0.5) < eps:
                    beam_weigts[i] /= 2.
        return beam_weigts

    def get_topology(self,lattice_type):
        if lattice_type == 0:
            connectivity = np.array([[0, 1],[1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6],[3, 7]])
            coordinates = np.array([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[1.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[1.0, 1.0, 1.0],[0.0, 1.0, 1.0]])
            # shift center to origin
            coordinates -= np.array([0.5, 0.5, 0.5])
        elif lattice_type == 1:
            connectivity = np.array([[0, 12],[1, 12],[2, 12],[3, 12],[4, 13],[5, 13],[6, 13],[7, 13],[0, 10],[3, 10],[4, 10],[7, 10],[1, 11],[2, 11],[5, 11],[6, 11],[0, 8],[1, 8],
                [4, 8],[5, 8],[2, 9],[3, 9],[6, 9],[7, 9],[8, 10],[8, 11],[8, 12],[8, 13],[9, 10],[9, 11],[9, 12],[9, 13],[10, 12],[10, 13],[11, 12],[11, 13]])
            coordinates = np.array([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[1.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],[0.5, 0.0, 0.5],[0.5, 1.0, 0.5],[0.0, 0.5, 0.5],[1.0, 0.5, 0.5],[0.5, 0.5, 0.0],[0.5, 0.5, 1.0]])
            coordinates -= np.array([0.5, 0.5, 0.5])
        elif lattice_type == 2:
            connectivity = np.array([[0, 8],[1, 8],[2, 8],[3, 8],[4, 8],[5, 8],[6, 8],[7, 8]])
            coordinates = np.array([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[1.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[1.0, 1.0, 1.0],[0.0, 1.0, 1.0],[0.5, 0.5, 0.5]])
            coordinates -= np.array([0.5, 0.5, 0.5])
        elif lattice_type == 3:
            connectivity = np.array([[0,4],[0,5],[1,4],[1,6],[2,4],[2,7],[3,4],[3,8],[4,5],[4,6],[4,7],[4,8]])
            coordinates = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[1.0,1.0,0.0],[0.5,0.5,0.5],[0.0,0.0,1.0],[1.0,0.0,1.0],[0.0,1.0,1.0],[1.0,1.0,1.0]])
            coordinates -= np.array([0.5, 0.5, 0.5])
        elif lattice_type == 4:
            connectivity = np.array([[0,1],[1,2],[2,3],[3,0],[0,4],[1,4],[2,4],[3,4],[4,5],[4,6],[4,7],[4,8],[5,6],[6,7],[7,8],[8,5]])
            coordinates = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0],[0.5,0.5,0.5],[0.0,1.0,1.0],[0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0]])
            coordinates -= np.array([0.5, 0.5, 0.5])
        elif lattice_type == 5:
            connectivity = np.array([[0,3],[3,1],[1,4],[4,2],[3,5],[3,6],[4,7],[4,8],[5,9],[6,9],[7,10],[8,10],[9,11],[9,12],[10,12],[10,13]])
            coordinates = np.array([[0.0,0.0,0.0],[0.5,0.5,0.0],[1.0,1.0,0.0],[0.25,0.25,0.25],[0.75,0.75,0.25],[0.0,0.5,0.5],[0.5,0.0,0.5],[0.5,1.0,0.5],
                [1.0,0.5,0.5],[0.25,0.25,0.75],[0.75,0.75,0.75],[0.0,0.0,1.0],[0.5,0.5,1.0],[1.0,1.0,1.0]])
            coordinates -= np.array([0.5, 0.5, 0.5])
        elif lattice_type ==6:
            connectivity = np.array([[0,2],[1,3],[2,4],[3,4],[4,5],[5,6],[5,7],[6,8],[7,9],[10,12],[11,13],[12,14],[13,14],[14,15],[15,16],[15,17],[16,18],[17,19],
                [2,12],[3,13],[4,14],[5,15],[6,16],[7,17]])
            coordinates = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,0.166666,0.0],[1.0,0.166666,0.0],[0.5,0.333333,0.0],[0.5,0.666666,0.0],[0.0,0.833333,0.0],
                [1.0,0.833333,0.0],[0.0,1.0,0.0],[1.0,1.0,0.0],[0.0,0.0,1.0],[1.0,0.0,1.0],[0.0,0.166666,1.0],[1.0,0.166666,1.0],[0.5,0.333333,1.0],[0.5,0.666666,1.0],
                [0.0,0.833333,1.0],[1.0,0.833333,1.0],[0.0,1.0,1.0],[1.0,1.0,1.0]])
            coordinates -= np.array([0.5, 0.5, 0.5])
        return connectivity, coordinates

def line_line_intersect()