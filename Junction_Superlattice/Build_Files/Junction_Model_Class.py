

import sys
import numpy as np
import matplotlib.pyplot as plt
import parameters as par
import FEM_vertex_class as FVC
import FEM_element_class as FEC
import FEM_mesh_class as FMC
import Hamiltonian_subClass as HSC
import scipy.sparse as Spar
import sparse_mtx_manipulation as SMM


class Junction_Model:

    def __init__(self,Lx,W_sc,W_j,W_c1,W_c2,L_c,a_sc_targ,a_j_targ,m_eff,alpha,W_sc_buffer = -1,ay_extended_targ = -1):

        ### Geometric parameters
        self.Lx = Lx     # length of supercell
        self.W_sc = W_sc # width of superconducting regions
        self.W_j = W_j   # width of wide part of junction
        self.W_c1 = W_c1 # width of constriction on the bottom side of the junction
        self.W_c2 = W_c2 # width of constriction on the top side of the junction
        self.L_c = L_c   # lenght of constriction region
        self.W_sc_buffer = W_sc_buffer

        self.a_SC_targ = a_sc_targ   # Target mesh constant in the region covered by SC
        self.a_j_targ = a_j_targ     # Target mesh constant in the junction region (only in y-direction)
        self.ay_extended_targ = ay_extended_targ # Target ay mesh constant in the SC region outside the buffer region

        self.m_eff = m_eff
        self.alpha = alpha


        print ('Generating mesh...')
        self.MESH = FMC.FEM_mesh()
        if W_sc_buffer == -1:
            self.mesh_gen_v1()
        else:
            if ay_extended_targ == -1:
                print ("Please specify ay_extended_targ")
                sys.exit()
            self.mesh_gen_v2_buffer()
        self.MESH.phys_regs = [101,102,103]
        self.MESH.PLOT.phys_regs = [101,102,103]

        self.MESH.gen_diff_ops()
        print ("...Done")

        ### Sub object for handling the Hamiltonian compilation
        self.HAM = HSC.HAM_subClass(self)

    def mesh_gen_v1(self):
        ### Creates the elements, vertices, and addes them to the MESH.
        ### While we are adding these elements, we also assign vertices to elements

        Nx1 = int(.5*(self.Lx - self.L_c)/self.a_SC_targ); ax1 = .5*(self.Lx - self.L_c)/float(Nx1)
        Nx2 = int(self.L_c/self.a_SC_targ) + 1; ax2 = self.L_c/float(Nx2-1.)
        Nx3 = int(.5*(self.Lx - self.L_c)/self.a_SC_targ) - 1; ax3 = .5*(self.Lx - self.L_c)/float(Nx3+1.)
        Nx = Nx1 + Nx2 + Nx3 # total number of vertices in each row

        Ny1 = int(self.W_sc /self.a_SC_targ); ay1 = (self.W_sc)/float(Ny1)
        if self.W_c1 != 0.:
            Ny2 = int(self.W_c1 /self.a_j_targ); ay2 = (self.W_c1)/float(Ny2)
        else:
            Ny2 = 0; ay2 = -1
        Ny3 = int((self.W_j - self.W_c1 - self.W_c2)/self.a_j_targ); ay3 = (self.W_j - self.W_c1 - self.W_c2)/float(Ny3)
        if self.W_c2 != 0.:
            Ny4 = int(self.W_c2 /self.a_j_targ); ay4 = (self.W_c2)/float(Ny4)
        else:
            Ny4 = 0; ay4 = -1
        Ny5 = Ny1 + 1; ay5 = ay1*1.
        Ny = Ny1 + Ny2 + Ny3 + Ny4 + Ny5

        X = []
        X1 = []
        X2 = []
        X3 = []
        for i in range(Nx1):
            X1.append(i*ax1)
            X.append(i*ax1)
        for i in range(Nx2):
            X2.append((Nx1)*ax1 + (i)*ax2)
            X.append((Nx1)*ax1 + (i)*ax2)
        for i in range(Nx3):
            X3.append((Nx1)*ax1 + (Nx2-1)*ax2 + (i+1)*ax3)
            X.append((Nx1)*ax1 + (Nx2-1)*ax2 + (i+1)*ax3)
        X1 = np.array(X1)
        X2 = np.array(X2)
        X3 = np.array(X3)
        X = np.array(X)
        #print (Nx1,Nx2,Nx3)
        #plt.scatter(X1,X1*0. + 0.)
        #plt.scatter(X2,X2*0. + 1.)
        #plt.scatter(X3,X3*0. + 2.)
        #plt.scatter(X,X*0. + 3.)
        #plt.show()


        Y = []
        Y1 = []
        Y2 = []
        Y3 = []
        Y4 = []
        Y5 = []
        for i in range(Ny1):
            Y1.append(i*ay1)
            Y.append(i*ay1)
        for i in range(Ny2):
            Y2.append(Ny1*ay1 + i*ay2)
            Y.append(Ny1*ay1 + i*ay2)
        for i in range(Ny3):
            Y3.append(Ny1*ay1 + Ny2*ay2 + i*ay3)
            Y.append(Ny1*ay1 + Ny2*ay2 + i*ay3)
        for i in range(Ny4):
            Y4.append(Ny1*ay1 + Ny2*ay2 + Ny3*ay3 + i*ay4)
            Y.append(Ny1*ay1 + Ny2*ay2 + Ny3*ay3 + i*ay4)
        for i in range(Ny5):
            Y5.append(Ny1*ay1 + Ny2*ay2 + Ny3*ay3 + Ny4*ay4 + i*ay5)
            Y.append(Ny1*ay1 + Ny2*ay2 + Ny3*ay3 + Ny4*ay4 + i*ay5)

        Y1 = np.array(Y1)
        Y2 = np.array(Y2)
        Y3 = np.array(Y3)
        Y4 = np.array(Y4)
        Y5 = np.array(Y5)
        Y = np.array(Y)
        #print (Ny1,Ny2,Ny3,Ny4,Ny5)
        #plt.scatter(Y1,Y1*0. + 0.)
        #plt.scatter(Y2,Y2*0. + 1.)
        #plt.scatter(Y3,Y3*0. + 2.)
        #plt.scatter(Y4,Y4*0. + 3.)
        #plt.scatter(Y5,Y5*0. + 4.)
        #plt.scatter(Y,Y*0. + 5.)
        #plt.show()

        counter = 0; counter_dof = 0
        for j in range(Ny):
            y = Y[j]
            for i in range(Nx):
                x = X[i]

                ### Create vertex
                vertex = FVC.FEM_vertex(x,y)
                vertex.assign_vertex_tag(counter) # Every vertex gets a tag, even if its on the boundary

                ### Check if vertex is on boundary or hard-walled region
                if (j == 0) or (j == Ny - 1):
                    pass
                else:
                    vertex.assign_dof_tag(counter_dof)
                    counter_dof += 1

                ### Add vertex to MESH
                self.MESH.add_vertex(vertex)



                ### Create first element (bottom triangle)
                if j != Ny - 1:
                    if i != Nx - 1:
                        element1 = FEC.FEM_element() # type 1 element (i.e. bottom triangle)
                        vertices1 = np.array([
                                             [X[i],Y[j]],
                                             [X[i+1],Y[j]],
                                             [X[i+1],Y[j+1]]
                        ])
                        element1.assign_vertices(vertices1)
                        element1.assign_vertex_tags(np.array([counter,counter+1,counter+1+Nx])) # adding vertex tags to all vertices that are apart of the element
                        self.MESH.add_element(element1)
                        centroid = element1.centroid
                        xx = centroid[0]; yy = centroid[1]
                        if yy < (self.W_sc):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j - self.W_c2):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 103
                        elif yy < (self.W_sc + self.W_j):
                            phys_tag = 102
                        else:
                            phys_tag = 103
                        element1.assign_phys_tag(phys_tag)


                        ### Create 2nd element (top triangle)
                        element2 = FEC.FEM_element() # type 2 element (i.e. top triangle)
                        vertices2 = np.array([
                                             [X[i],Y[j]],
                                             [X[i],Y[j+1]],
                                             [X[i+1],Y[j+1]]
                        ])
                        element2.assign_vertices(vertices2)
                        element2.assign_phys_tag(phys_tag)
                        element2.assign_vertex_tags(np.array([counter,counter+Nx,counter+1+Nx])) # adding vertex tags to all vertices that are apart of the element
                        self.MESH.add_element(element2)

                        centroid = element2.centroid
                        xx = centroid[0]; yy = centroid[1]
                        if yy < (self.W_sc):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j - self.W_c2):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 103
                        elif yy < (self.W_sc + self.W_j):
                            phys_tag = 102
                        else:
                            phys_tag = 103
                        element2.assign_phys_tag(phys_tag)


                    else:
                        element1 = FEC.FEM_element() # type 1 element (i.e. bottom triangle)
                        vertices1 = np.array([
                                             [X[i],Y[j]],
                                             [X[i] + ax3,Y[j]],
                                             [X[i] + ax3,Y[j+1]]
                        ])
                        element1.assign_vertices(vertices1)
                        element1.assign_vertex_tags(np.array([counter,counter+1 - Nx,counter+1+Nx - Nx])) # adding vertex tags to all vertices that are apart of the element
                        self.MESH.add_element(element1)

                        centroid = element1.centroid
                        xx = centroid[0]; yy = centroid[1]
                        if yy < (self.W_sc):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j - self.W_c2):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 103
                        elif yy < (self.W_sc + self.W_j):
                            phys_tag = 102
                        else:
                            phys_tag = 103
                        element1.assign_phys_tag(phys_tag)

                        #print (counter,counter+1 - Nx,counter+1+Nx - Nx)

                        ### Create 2nd element (top triangle)
                        element2 = FEC.FEM_element() # type 2 element (i.e. top triangle)
                        vertices2 = np.array([
                                             [X[i],Y[j]],
                                             [X[i],Y[j+1]],
                                             [X[i] + ax3,Y[j+1]]
                        ])
                        element2.assign_vertices(vertices2)
                        element2.assign_vertex_tags(np.array([counter,counter+Nx,counter+1+Nx - Nx])) # adding vertex tags to all vertices that are apart of the element
                        self.MESH.add_element(element2)

                        centroid = element2.centroid
                        xx = centroid[0]; yy = centroid[1]
                        if yy < (self.W_sc):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j - self.W_c2):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 103
                        elif yy < (self.W_sc + self.W_j):
                            phys_tag = 102
                        else:
                            phys_tag = 103
                        element2.assign_phys_tag(phys_tag)

                counter += 1

        ### Create vertex to dof map and dof to vertex map
        self.MESH.vertex_tag_to_dof_tag()

        ### Assign dof tags to element vertices
        for i in range(len(self.MESH.elements)):
            self.MESH.elements[i].find_dof_tags(self.MESH.vtd)

    def mesh_gen_v2_buffer(self):
        ### Creates the elements, vertices, and addes them to the MESH.
        ### While we are adding these elements, we also assign vertices to elements

        Nx1 = int(.5*(self.Lx - self.L_c)/self.a_SC_targ); ax1 = .5*(self.Lx - self.L_c)/float(Nx1)
        Nx2 = int(self.L_c/self.a_SC_targ) + 1; ax2 = self.L_c/float(Nx2-1.)
        Nx3 = int(.5*(self.Lx - self.L_c)/self.a_SC_targ) - 1; ax3 = .5*(self.Lx - self.L_c)/float(Nx3+1.)
        Nx = Nx1 + Nx2 + Nx3 # total number of vertices in each row

        Ny1 = int((self.W_sc - self.W_sc_buffer)/self.ay_extended_targ); ay1 = (self.W_sc - self.W_sc_buffer)/float(Ny1)
        Ny2 = int(self.W_sc_buffer /self.a_SC_targ); ay2 = (self.W_sc_buffer)/float(Ny2)
        if self.W_c1 != 0.:
            Ny3 = int(self.W_c1 /self.a_j_targ); ay3 = (self.W_c1)/float(Ny3)
        else:
            Ny3 = 0; ay3 = -1
        Ny4 = int((self.W_j - self.W_c1 - self.W_c2)/self.a_j_targ); ay4 = (self.W_j - self.W_c1 - self.W_c2)/float(Ny4)
        if self.W_c2 != 0.:
            Ny5 = int(self.W_c2 /self.a_j_targ); ay5 = (self.W_c2)/float(Ny5)
        else:
            Ny5 = 0; ay5 = -1
        Ny6 = int(self.W_sc_buffer /self.a_SC_targ); ay6 = (self.W_sc_buffer)/float(Ny6)
        Ny7 = Ny1 + 1; ay7 = ay1*1.
        Ny = Ny1 + Ny2 + Ny3 + Ny4 + Ny5 + Ny6 + Ny7

        X = []
        X1 = []
        X2 = []
        X3 = []
        for i in range(Nx1):
            X1.append(i*ax1)
            X.append(i*ax1)
        for i in range(Nx2):
            X2.append((Nx1)*ax1 + (i)*ax2)
            X.append((Nx1)*ax1 + (i)*ax2)
        for i in range(Nx3):
            X3.append((Nx1)*ax1 + (Nx2-1)*ax2 + (i+1)*ax3)
            X.append((Nx1)*ax1 + (Nx2-1)*ax2 + (i+1)*ax3)
        X1 = np.array(X1)
        X2 = np.array(X2)
        X3 = np.array(X3)
        X = np.array(X)
        #print (Nx1,Nx2,Nx3)
        #plt.scatter(X1,X1*0. + 0.)
        #plt.scatter(X2,X2*0. + 1.)
        #plt.scatter(X3,X3*0. + 2.)
        #plt.scatter(X,X*0. + 3.)
        #plt.show()


        Y = []
        Y1 = []
        Y2 = []
        Y3 = []
        Y4 = []
        Y5 = []
        Y6 = []
        Y7 = []
        for i in range(Ny1):
            Y1.append(i*ay1)
            Y.append(i*ay1)
        for i in range(Ny2):
            Y2.append(Ny1*ay1 + i*ay2)
            Y.append(Ny1*ay1 + i*ay2)
        for i in range(Ny3):
            Y3.append(Ny1*ay1 + Ny2*ay2 + i*ay3)
            Y.append(Ny1*ay1 + Ny2*ay2 + i*ay3)
        for i in range(Ny4):
            Y4.append(Ny1*ay1 + Ny2*ay2 + Ny3*ay3 + i*ay4)
            Y.append(Ny1*ay1 + Ny2*ay2 + Ny3*ay3 + i*ay4)
        for i in range(Ny5):
            Y5.append(Ny1*ay1 + Ny2*ay2 + Ny3*ay3 + Ny4*ay4 + i*ay5)
            Y.append(Ny1*ay1 + Ny2*ay2 + Ny3*ay3 + Ny4*ay4 + i*ay5)
        for i in range(Ny6):
            Y6.append(Ny1*ay1 + Ny2*ay2 + Ny3*ay3 + Ny4*ay4 + Ny5*ay5 + i*ay6)
            Y.append(Ny1*ay1 + Ny2*ay2 + Ny3*ay3 + Ny4*ay4 + Ny5*ay5 + i*ay6)
        for i in range(Ny7):
            Y7.append(Ny1*ay1 + Ny2*ay2 + Ny3*ay3 + Ny4*ay4 + Ny5*ay5 + Ny6*ay6 + i*ay7)
            Y.append(Ny1*ay1 + Ny2*ay2 + Ny3*ay3 + Ny4*ay4 + Ny5*ay5 + Ny6*ay6 + i*ay7)

        Y1 = np.array(Y1)
        Y2 = np.array(Y2)
        Y3 = np.array(Y3)
        Y4 = np.array(Y4)
        Y5 = np.array(Y5)
        Y6 = np.array(Y6)
        Y7 = np.array(Y7)
        Y = np.array(Y)
        #print (Ny1,Ny2,Ny3,Ny4,Ny5)
        #plt.scatter(Y1,Y1*0. + 0.)
        #plt.scatter(Y2,Y2*0. + 1.)
        #plt.scatter(Y3,Y3*0. + 2.)
        #plt.scatter(Y4,Y4*0. + 3.)
        #plt.scatter(Y5,Y5*0. + 4.)
        #plt.scatter(Y,Y*0. + 5.)
        #plt.show()

        counter = 0; counter_dof = 0
        for j in range(Ny):
            y = Y[j]
            for i in range(Nx):
                x = X[i]

                ### Create vertex
                vertex = FVC.FEM_vertex(x,y)
                vertex.assign_vertex_tag(counter) # Every vertex gets a tag, even if its on the boundary

                ### Check if vertex is on boundary or hard-walled region
                if (j == 0) or (j == Ny - 1):
                    pass
                else:
                    vertex.assign_dof_tag(counter_dof)
                    counter_dof += 1

                ### Add vertex to MESH
                self.MESH.add_vertex(vertex)



                ### Create first element (bottom triangle)
                if j != Ny - 1:
                    if i != Nx - 1:
                        element1 = FEC.FEM_element() # type 1 element (i.e. bottom triangle)
                        vertices1 = np.array([
                                             [X[i],Y[j]],
                                             [X[i+1],Y[j]],
                                             [X[i+1],Y[j+1]]
                        ])
                        element1.assign_vertices(vertices1)
                        element1.assign_vertex_tags(np.array([counter,counter+1,counter+1+Nx])) # adding vertex tags to all vertices that are apart of the element
                        self.MESH.add_element(element1)
                        centroid = element1.centroid
                        xx = centroid[0]; yy = centroid[1]
                        if yy < (self.W_sc):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j - self.W_c2):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 103
                        elif yy < (self.W_sc + self.W_j):
                            phys_tag = 102
                        else:
                            phys_tag = 103
                        element1.assign_phys_tag(phys_tag)


                        ### Create 2nd element (top triangle)
                        element2 = FEC.FEM_element() # type 2 element (i.e. top triangle)
                        vertices2 = np.array([
                                             [X[i],Y[j]],
                                             [X[i],Y[j+1]],
                                             [X[i+1],Y[j+1]]
                        ])
                        element2.assign_vertices(vertices2)
                        element2.assign_phys_tag(phys_tag)
                        element2.assign_vertex_tags(np.array([counter,counter+Nx,counter+1+Nx])) # adding vertex tags to all vertices that are apart of the element
                        self.MESH.add_element(element2)

                        centroid = element2.centroid
                        xx = centroid[0]; yy = centroid[1]
                        if yy < (self.W_sc):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j - self.W_c2):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 103
                        elif yy < (self.W_sc + self.W_j):
                            phys_tag = 102
                        else:
                            phys_tag = 103
                        element2.assign_phys_tag(phys_tag)


                    else:
                        element1 = FEC.FEM_element() # type 1 element (i.e. bottom triangle)
                        vertices1 = np.array([
                                             [X[i],Y[j]],
                                             [X[i] + ax3,Y[j]],
                                             [X[i] + ax3,Y[j+1]]
                        ])
                        element1.assign_vertices(vertices1)
                        element1.assign_vertex_tags(np.array([counter,counter+1 - Nx,counter+1+Nx - Nx])) # adding vertex tags to all vertices that are apart of the element
                        self.MESH.add_element(element1)

                        centroid = element1.centroid
                        xx = centroid[0]; yy = centroid[1]
                        if yy < (self.W_sc):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j - self.W_c2):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 103
                        elif yy < (self.W_sc + self.W_j):
                            phys_tag = 102
                        else:
                            phys_tag = 103
                        element1.assign_phys_tag(phys_tag)

                        #print (counter,counter+1 - Nx,counter+1+Nx - Nx)

                        ### Create 2nd element (top triangle)
                        element2 = FEC.FEM_element() # type 2 element (i.e. top triangle)
                        vertices2 = np.array([
                                             [X[i],Y[j]],
                                             [X[i],Y[j+1]],
                                             [X[i] + ax3,Y[j+1]]
                        ])
                        element2.assign_vertices(vertices2)
                        element2.assign_vertex_tags(np.array([counter,counter+Nx,counter+1+Nx - Nx])) # adding vertex tags to all vertices that are apart of the element
                        self.MESH.add_element(element2)

                        centroid = element2.centroid
                        xx = centroid[0]; yy = centroid[1]
                        if yy < (self.W_sc):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 101
                        elif yy < (self.W_sc + self.W_c1):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j - self.W_c2):
                            phys_tag = 102
                        elif yy < (self.W_sc + self.W_j) and ( .5*(self.Lx - self.L_c) < xx < .5*(self.Lx + self.L_c) ):
                            phys_tag = 103
                        elif yy < (self.W_sc + self.W_j):
                            phys_tag = 102
                        else:
                            phys_tag = 103
                        element2.assign_phys_tag(phys_tag)

                counter += 1

        ### Create vertex to dof map and dof to vertex map
        self.MESH.vertex_tag_to_dof_tag()

        ### Assign dof tags to element vertices
        for i in range(len(self.MESH.elements)):
            self.MESH.elements[i].find_dof_tags(self.MESH.vtd)

    def state_weight_junction(self,U):
        ### calculates the weight of the states in the juction
        U_hc = np.conjugate(np.transpose(U))
        M = self.MESH.DIFF_OPS.Diag_2
        N = SMM.zero_csr_mtx(self.MESH.DIFF_OPS.Diag.shape[0])
        DIAG_J = Spar.bmat([
                           [M,N],
                           [N,M]
        ],format = 'csc')
        weight_junction = np.diag( np.dot(U_hc, DIAG_J.dot(U)) )
        return weight_junction.real
