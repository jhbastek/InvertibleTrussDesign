import numpy as np
import plotly.express as plt
import plotly.graph_objects as go
import operator
import functools
from src.voigt_rotation import get_rotation_matrix_np

class Topology:
    def __init__(self,lattice):
        self.connectity, self.coordinates, self.diameter = self.create_lattice(lattice)

    def create_lattice(self,lattice):
        conn1, coord1 = self.create_individual_lattice(lattice['lattice_type1'],lattice['lattice_rep1'])
        conn2, coord2 = self.create_individual_lattice(lattice['lattice_type2'],lattice['lattice_rep2'])
        conn3, coord3 = self.create_individual_lattice(lattice['lattice_type3'],lattice['lattice_rep3'])
        compound_lattice_conn, compound_lattice_coord = self.create_compound_lattice(conn1,conn2,conn3,coord1,coord2,coord3)
        compound_lattice_conn, compound_lattice_coord = self.correctBeamIntersections(compound_lattice_conn,compound_lattice_coord)
        compound_lattice_coord_def = self.affinely_deform_lattice(compound_lattice_coord,lattice)
        diameter = self.compute_diameter(compound_lattice_conn, compound_lattice_coord, compound_lattice_coord_def, lattice)
        return compound_lattice_conn, compound_lattice_coord_def, diameter

    # apply affine deformation
    def affinely_deform_lattice(self,compound_lattice_coord,lattice):
        affine_def = self.get_affine_deformation(lattice)
        final_lattice_coord = np.linalg.multi_dot([compound_lattice_coord,affine_def.transpose()])
        return final_lattice_coord

    # compute affine deformation matrix
    def get_affine_deformation(self,lattice):
        U = np.diag([lattice['U1'],lattice['U2'],lattice['U3']])
        V = np.diag([lattice['V1'],lattice['V2'],lattice['V3']])
        R1 = get_rotation_matrix_np(lattice['R1_theta'],lattice['R1_rot_ax1'],lattice['R1_rot_ax2'])
        R2 = get_rotation_matrix_np(lattice['R2_theta'],lattice['R2_rot_ax1'],lattice['R2_rot_ax2'])
        return np.linalg.multi_dot([R2, V, R1, U])

    # create lattice based on given elementary topology and tesselation
    def create_individual_lattice(self,lattice_index,lattice_tess):
        conn, coord = self.get_topology(lattice_index)
        temp_conn = conn.copy()
        temp_coord = coord.copy()
        UC_nodes = len(coord)
        if lattice_tess == 2:
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
        tot_conn, tot_coord = self.remove_overlapping_nodes(tot_conn,tot_coord)
        return tot_conn, tot_coord

    def remove_overlapping_nodes(self,conn,coord):
        tot_nodes = len(coord)
        red_conn = conn
        duplicate_nodes = []
        tol = 1.e-6
        for i in range(tot_nodes):
            for j in range((i+1), tot_nodes):
                if np.linalg.norm(coord[i,:] - coord[j,:]) < tol:
                    red_conn[red_conn==j] = i
                    duplicate_nodes.append(j)

        if duplicate_nodes:
            duplicate_nodes = np.unique(duplicate_nodes)
            duplicate_nodes = np.sort(duplicate_nodes)[::-1]
            for i in duplicate_nodes:
                red_conn[red_conn>i] = red_conn[red_conn>i] - 1
        
        # delete duplicate nodes and preserve order
        _, idx = np.unique(coord, axis=0, return_index=True)
        red_coord = coord[np.sort(idx)]

        # delete duplicate connectivities
        for i in range(len(red_conn)):
            if red_conn[i,0]>red_conn[i,1]:
                temp = red_conn[i,1]
                red_conn[i,1] = red_conn[i,0]
                red_conn[i,0] = temp

        red_conn = np.unique(red_conn,axis=0)

        return red_conn, red_coord

    def compute_diameter(self,conn,coord,coord_def,lattice):
        beam_weights = self.generate_beam_weights(conn,coord)
        weighted_length = 0.
        for i in range(len(conn)):
            weighted_length += beam_weights[i]*np.linalg.norm(coord_def[conn[i,0]]-coord_def[conn[i,1]])
        affine_def = self.get_affine_deformation(lattice)
        tot_volume = np.linalg.det(affine_def)
        area = lattice['relative_density']*tot_volume / weighted_length
        diameter = np.sqrt(4.*area/np.pi)
        return diameter

    # generate factor for beams that are only partly in considered UC
    def generate_beam_weights(self,conn,coord):
        eps = 1.e-6
        beam_weigts = np.ones((len(conn)))
        for i in range(len(conn)):
            for j in range(3):
                if abs(coord[conn[i,0],j]-0.5) < eps and abs(coord[conn[i,1],j]-0.5) < eps or\
                abs(coord[conn[i,0],j]+0.5) < eps and abs(coord[conn[i,1],j]+0.5) < eps:
                    beam_weigts[i] /= 2.
        return beam_weigts

    # compute the shortest distance between two lines, see http://paulbourke.net/geometry/pointlineplane/ for details
    def line_line_intersect(self,p1,p2,p3,p4):
        spatialTolerance = 1.e-6
        corner_intersection = 0
        p13 = p1 - p3
        p43 = p4 - p3
        if (abs(p43[0]) < 1.e-10 and abs(p43[1]) < 1.e-10 and abs(p43[2]) < 1.e-10):
            intersect_flag = False
            return intersect_flag, None, None
        p21 = p2 - p1
        if (abs(p21[0]) < 1.e-10 and abs(p21[1]) < 1.e-10 and abs(p21[2]) < 1.e-10):
            intersect_flag = False
            return intersect_flag, None, None

        d1343 = p13[0] * p43[0] + p13[1] * p43[1] + p13[2] * p43[2]
        d4321 = p43[0] * p21[0] + p43[1] * p21[1] + p43[2] * p21[2]
        d1321 = p13[0] * p21[0] + p13[1] * p21[1] + p13[2] * p21[2]
        d4343 = p43[0] * p43[0] + p43[1] * p43[1] + p43[2] * p43[2]
        d2121 = p21[0] * p21[0] + p21[1] * p21[1] + p21[2] * p21[2]

        # parallel beams have denom=0
        denom = d2121 * d4343 - d4321 * d4321
        if (abs(denom) < 1.e-10):
            intersect_flag = False
            return intersect_flag, None, None
        else:
            numer = d1343 * d4321 - d1321 * d4343
            mua = numer / denom
            mub = (d1343 + d4321 * mua) / d4343

        pa = p1 + mua * p21
        pb = p3 + mub * p43

        # only consider exact intersections
        if (np.linalg.norm(pb-pa) > spatialTolerance):
            intersect_flag = False
            return intersect_flag, None, None
        # do not consider intersections outside of beam length
        elif (mua + spatialTolerance < 0. or 1. < mua - spatialTolerance or mub + spatialTolerance < 0. or 1. < mub - spatialTolerance):
            intersect_flag = False
            return intersect_flag, None, None
        # do not remove adjacent beams (sharing a node)
        elif ((abs(mua) < spatialTolerance or abs(mua - 1.) < spatialTolerance) and (abs(mub) < spatialTolerance or abs(mub - 1.) < spatialTolerance)):
            intersect_flag = False
            return intersect_flag, None, None
        else:
            intersect_flag = True
            pc = None
            if (abs(mua) < spatialTolerance and abs(mub) > spatialTolerance and abs(mub - 1.) > spatialTolerance):
                corner_intersection = 1
                # p1
            elif (abs(mua - 1.) < spatialTolerance and abs(mub) > spatialTolerance and abs(mub - 1.) > spatialTolerance):
                corner_intersection = 2
                # p2    
            elif (abs(mub) < spatialTolerance and abs(mua) > spatialTolerance and abs(mua - 1.) > spatialTolerance):
                corner_intersection = 3
                # p3
            elif (abs(mub - 1.) < spatialTolerance and abs(mua) > spatialTolerance and abs(mua - 1.) > spatialTolerance):
                corner_intersection = 4
                # p4
            else:
                pc = pb + (pa - pb) / 2.
            return intersect_flag, corner_intersection, pc

    # correct all intersecting beams and introduces new nodes at points of intersection
    def correctBeamIntersections(self,conn,coord):

        prev_elements = 0

        conn_list = conn.tolist()
        coord_list = coord.tolist()

        # loop over algorithm until all intersections have been updated 
        # (necessary for beams with multiple intersections)
        while (len(conn_list) != prev_elements):
            prev_elements = len(conn_list)
            
            node1IdA = 0
            node2IdA = 0
            node1IdB = 0
            node2IdB = 0

            numberOfNodes = len(coord_list)
            numberOfElements = len(conn_list)
            availableNodeNumber = numberOfNodes

            elementsMarkedForDeletion = []
            for elementIndex in range(numberOfElements): 
                elementsMarkedForDeletion.append(0)

            # iterate over all element combinations
            for elementIndexA in range(numberOfElements-1):
                for elementIndexB in range(elementIndexA+1,numberOfElements):
                    # grab the two nodes of first element
                    node1IdA = conn_list[elementIndexA][0]
                    node2IdA = conn_list[elementIndexA][1]

                    # Find the coordinate of the two nodes of first element
                    node1PositionA = coord[node1IdA]
                    node2PositionA = coord[node2IdA]

                    # grab the two nodes of second element
                    node1IdB = conn_list[elementIndexB][0]
                    node2IdB = conn_list[elementIndexB][1]

                    # Find the coordinate of the two nodes of second element
                    node1PositionB = coord[node1IdB]
                    node2PositionB = coord[node2IdB]

                    intersect_flag, corner_intersection, pc = self.line_line_intersect(node1PositionA,node2PositionA,node1PositionB,node2PositionB)
                    # intersect_flag = False
                    if(intersect_flag):
                        if(corner_intersection == 0):
                            # create new node in midpoint of intersection
                            coord_list.append((pc))
                            # add the connections between the new node and 4 original nodes
                            conn_list.append([node1IdA,availableNodeNumber])
                            conn_list.append([node1IdB,availableNodeNumber])
                            conn_list.append([node2IdA,availableNodeNumber])
                            conn_list.append([node2IdB,availableNodeNumber])
                            availableNodeNumber += 1
                            elementsMarkedForDeletion[elementIndexA] = 1
                            elementsMarkedForDeletion[elementIndexB] = 1
                        elif(corner_intersection == 1):
                            # split connections of the cornered element
                            conn_list.append([node1IdB,node1IdA])
                            conn_list.append([node2IdB,node1IdA])
                            elementsMarkedForDeletion[elementIndexB] = 1
                        elif(corner_intersection == 2):
                            # split connections of the cornered element
                            conn_list.append([node1IdB,node2IdA])
                            conn_list.append([node2IdB,node2IdA])
                            elementsMarkedForDeletion[elementIndexB] = 1
                        elif(corner_intersection == 3):
                            # split connections of the cornered element
                            conn_list.append([node1IdA,node1IdB])
                            conn_list.append([node2IdA,node1IdB])
                            elementsMarkedForDeletion[elementIndexA] = 1
                        elif(corner_intersection == 4):
                            # split connections of the cornered element
                            conn_list.append([node1IdA,node2IdB])
                            conn_list.append([node2IdA,node2IdB])
                            elementsMarkedForDeletion[elementIndexA] = 1
            
            # delete old elements
            elementIndex = 0
            while elementIndex < numberOfElements:
                if (elementsMarkedForDeletion[elementIndex] == 1):
                    del conn_list[elementIndex]
                    del elementsMarkedForDeletion[elementIndex]
                    numberOfElements -= 1
                    if (elementIndex > 0):
                        elementIndex -= 1
                elementIndex += 1

            # remove duplicates
            conn, coord = self.remove_overlapping_nodes(np.asarray(conn_list),np.asarray(coord_list))
            conn_list = conn.tolist()
            coord_list = coord.tolist()

        return np.asarray(conn_list), np.asarray(coord_list)

    def plot(self):
        fig = plt.scatter_3d(self.coordinates, x=0, y=1, z=2, width=100)
        fig = self.connect_points(fig)
        fig.update_layout(title_text='Predicted lattice (diameter = {:.3f})'.format(self.diameter), title_x=0.5)
        fig.update_traces(marker=dict(size=5,
                              line=dict(width=20)))
        fig.show()

    def connect_points(self, fig):
        fig_temp = [fig]
        for i in range(len(self.connectity)):
            x1, x2 = self.coordinates[self.connectity[i,0],0], self.coordinates[self.connectity[i,1],0]
            y1, y2 = self.coordinates[self.connectity[i,0],1], self.coordinates[self.connectity[i,1],1]
            z1, z2 = self.coordinates[self.connectity[i,0],2], self.coordinates[self.connectity[i,1],2]
            fig_temp.append(plt.line_3d(x=[x1,x2],y=[y1,y2],z=[z1,z2]))
        fig_temp = go.Figure(data=functools.reduce(operator.add, [_.data for _ in fig_temp]))
        return fig_temp

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
        elif lattice_type == 6:
            connectivity = np.array([[0,2],[1,3],[2,4],[3,4],[4,5],[5,6],[5,7],[6,8],[7,9],[10,12],[11,13],[12,14],[13,14],[14,15],[15,16],[15,17],[16,18],[17,19],
                [2,12],[3,13],[4,14],[5,15],[6,16],[7,17]])
            coordinates = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,0.166666,0.0],[1.0,0.166666,0.0],[0.5,0.333333,0.0],[0.5,0.666666,0.0],[0.0,0.833333,0.0],
                [1.0,0.833333,0.0],[0.0,1.0,0.0],[1.0,1.0,0.0],[0.0,0.0,1.0],[1.0,0.0,1.0],[0.0,0.166666,1.0],[1.0,0.166666,1.0],[0.5,0.333333,1.0],[0.5,0.666666,1.0],
                [0.0,0.833333,1.0],[1.0,0.833333,1.0],[0.0,1.0,1.0],[1.0,1.0,1.0]])
            coordinates -= np.array([0.5, 0.5, 0.5])
        elif lattice_type == 7:
            connectivity = np.array([[0,1],[2,3]])
            coordinates = np.array([[0.0,0.0,-0.5],[0.0,0.0,0.5],[-1.0,0.0,0.0],[0.0,0.0,0.0]])
        return connectivity, coordinates