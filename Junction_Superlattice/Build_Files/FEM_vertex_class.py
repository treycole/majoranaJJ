"""
    Contains the class FEM_vertex, which is a finite element mesh vertex
    of a 2D mesh
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
import FEM_element_class as FEC

class FEM_vertex:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.coor = np.array([x,y])

        self.dof_bool = False # Boolean of whether the vertex cooresponds to a dof.
                              #     * default to False. Turned to True when dof_tag is assigned
        self.elements = []    # list of FEM elements that the vertex is apart of

    def assign_vertex_tag(self,vertex_tag):
        ### Assigns vertex_tag to vertex.
        ###     * Note that this tag corresponds to the mesh and not dofs
        ###         * If the vertex is on a boundary, it won't have a dof_tag, but it will still have a vertex_tag
        if (type(vertex_tag) == int) or (type(vertex_tag) == long):
            self.vertex_tag = vertex_tag
        else:
            print ('\n' + '*' * 50 + '\n'); print ('vertex_tag of a vertex object must be an int or long')
            print ('type(vertex_tag): ' + str(type(vertex_tag)))
            print ('\n' + '*' * 50); sys.exit()

    def assign_dof_tag(self,dof_tag):
        ### Assigns dof_tag to vertex.
        ###     * Note that this tag corresponds to the dofs and not the mesh
        ###         * If the vertex is on a boundary, it won't have a dof_tag, but it will still have a vertex_tag
        if (type(dof_tag) == int) or (type(dof_tag) == long):
            self.dof_tag = dof_tag
        else:
            print ('\n' + '*' * 50 + '\n'); print ('dof_tag of a vertex object must be an int or long')
            print ('type(dof_tag): '+ str(type(dof_tag)))
            print ('\n' + '*' * 50); sys.exit()
        self.dof_bool = True

    def add_element(self,element):
        ### If the element isn't already in the list, adds the element to the element list
        ### Checks that the vertex is apart of the element. If not, throws up a warning.

        if isinstance(element,FEC.FEM_element):
            for i in range(len(self.elements)):
                if element.are_equal(self.elements[i]):
                    print ('element already belongs to vertex')
                    return None
            self.elements.append(element)
        else:
            print ('\n' + '*' * 50 + '\n'); print ('element is not an instance of FEM_element class')
            print ('type(element): '+ str(type(element)))
            print ('\n' + '*' * 50); sys.exit()

    def are_equal(self,vertex2):
        if isinstance(vertex2,FEM_vertex):
            if np.array_equal(self.coor,vertex2.coor):
                return True
            else:
                return False
        else:
            print ('\n\n!WARNING!: vertex2 is not an instance of FEM_vertex...returning False\n\n')
            return False

    def plot_vertex(self):
        plt.scatter([self.x],[self.y],c = 'r')
        if len(self.elements) != 0:
            for i in range(len(self.elements)):
                X = self.elements[i].vertices[:,0]
                Y = self.elements[i].vertices[:,1]
                plt.plot([X[0],X[1]],[Y[0],Y[1]],c = 'b')
                plt.plot([X[0],X[2]],[Y[0],Y[2]],c = 'b')
                plt.plot([X[2],X[1]],[Y[2],Y[1]],c = 'b')
        plt.grid()
        plt.show()
