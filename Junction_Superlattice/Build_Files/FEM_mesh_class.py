"""
    Contains the class FEM_mesh, which is a finite element 2D mesh with
    triangular type elements
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from matplotlib.collections import LineCollection
import scipy.sparse as Spar
import FEM_vertex_class as FVC
import FEM_element_class as FEC
from scipy import interpolate
from bisect import bisect_left

def BinarySearch(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1


class FEM_mesh:
    def __init__(self):
        self.vertices = []  # list of FEM_vertex instances
        self.elements = []  # list of FEM_element instances
        self.X = []; self.Y = []
        self.phys_regs = []

        ### Sub object for handling matrix assembly
        self.MTX_assembly = FEM_mtx_assembly_subobject(self)

        ### Sub object for handling plotting functions
        self.PLOT = Plot_subObject(self)

    def add_vertex(self,vertex):
        ### adds FEM_vertex instance to the vertex list
        ###     * Note that we don't check if the vertex is already in the list
        if isinstance(vertex,FVC.FEM_vertex):
            self.vertices.append(vertex)
            self.X.append(vertex.x)
            self.Y.append(vertex.y)
        else:
            print ('\n' + '*' * 50 + '\n'); print ('vertex is not an instance of FEM_vertex class')
            print ('type(vertex): ' +  str(type(vertex)))
            print ('\n' + '*' * 50); sys.exit()

    def add_element(self,element):
        ### adds FEM_element instance to the element list
        ###     * Note that we don't check if the element is already in the list
        if isinstance(element,FEC.FEM_element):
            self.elements.append(element)
        else:
            print ('\n' + '*' * 50 + '\n'); print ('element is not an instance of FEM_element class')
            print ('type(element): ' +  str(type(element)))
            print ('\n' + '*' * 50); sys.exit()

    def vertex_tag_to_dof_tag(self):
        ### Creates an array that maps each vertex tag to a dof tag
        ###     * If a vertex is found not to have a vertex tage, an error is thrown up
        ###     * vtd[i] == -1 means the ith vertex does not map to a dof site

        self.vtd = -1*np.ones(len(self.vertices),dtype = 'int')   # vtd (vertex tag to dof tag)
        self.idx_arrNan = []    # index array of vertices that are not dof sites
        self.idx_arrDof = []    # index array of vertices that are dof sites
        counter_dof = 0
        for i in range(self.vtd.size):
            vertex = self.vertices[i]
            try:
                v_tag = vertex.vertex_tag
            except:
                print ('Could not find a vertex tag of the %d vertex' % (i))
                sys.exit()
            if vertex.dof_bool: # seeing if the vertex has a dof_tag
                self.vtd[v_tag] = vertex.dof_tag
                counter_dof += 1
                self.idx_arrDof.append(i)
            else:
                self.vtd[v_tag] = -1 # no dof_tag
                self.idx_arrNan.append(i)

        self.num_dof = counter_dof
        self.dtv = -1*np.ones(counter_dof,dtype = 'int') # dtv (dof tag to vertex tag)
        for i in range(self.vtd.size):
            if self.vtd[i] != -1:
                self.dtv[self.vtd[i]] = i

        self.idx_arrDof = np.array(self.idx_arrDof)
        self.idx_arrNan = np.array(self.idx_arrNan)
        #self.X_dof = np.array(self.X)[self.idx_arrDof]
        #self.Y_dof = np.array(self.Y)[self.idx_arrDof]

    def match_vertices_to_elements(self,monitor = False):
        ### For each element, we search through the vertices to find which vertices
        ### belong to the element. Also assign elements to vertices.
        ###         * Slow algorithm, but I don't think its a big deal at this point
        ###         * Note that you must have created a vertex to dof map before running this function

        ### Loop through all elements
        for i in range(len(self.elements)):
            elem = self.elements[i]
            elem.find_vertices(self.vertices)
            elem.find_dof_tags(self.vtd)
            if monitor and (i % 10 == 0):
                print (len(self.elements) - i)

    def read_gmesh(self,file_path):
        ### This functions takes the file_path of a gmsh .msh file
        ### and reads the data from that mesh such that it becomes
        ### usable within our program

        file = open(file_path,"r")
        for i in range(4): file.readline()  # skipping some initial format lines

        num_vert = int(file.readline())     # number of vertices in mesh
        for i in range(num_vert):           # Loop through all the vertices
            line = file.readline().split()
            x = float(line[1]); y = float(line[2])
            vertex = FVC.FEM_vertex(x,y)    # instance of FEM_vertex class
            vertex.assign_vertex_tag(i)     # Tagging the vertex (every vertex gets a tag, not just dof sites)
            self.add_vertex(vertex)         # add the vertex to the mesh

        for i in range(2): file.readline()  # skipping some format lines

        num_elems_tol = int(file.readline())     # number of elements total in mesh. This include the boundary elements as well
        bound_verts = []                    # list of vertex tags that belong to the boundary, i.e. not dof sites
        for i in range(num_elems_tol):
            elem = file.readline().split()
            if elem[1] == "1":              # This is a boundary element
                s= int(elem[2])             # number of tags to ignore
                bv1 = int(elem[3+s])-1      # vertex tag of the 1st boundary vertex of this element
                bv2 = int(elem[3+s+1])-1    # vertex tag of the 2nd boundary vertex of this element
                if bv1 not in bound_verts:
                    bound_verts.append(bv1)
                if bv2 not in bound_verts:
                    bound_verts.append(bv2)
            elif elem[1] == "2":            # this is not a boundary element, but a "regular" element (i.e. triangular element)
                element = FEC.FEM_element() # instance of FEM_element class
                s = int(elem[2])            # number of tags before the vertex tags
                phys_tag = int(elem[3])          # physical region tag of that element
                element.assign_phys_tag(phys_tag)
                if phys_tag not in self.phys_regs:
                    self.phys_regs.append(phys_tag)
                vert_tags = np.zeros(3,dtype = 'int')
                vert_coor = np.zeros((3,2))
                for j in range(3):
                    vert_tags[j] = int(elem[3+s+j]) - 1 # vertex tags of the vertices that compose the element
                    vertex = self.vertices[vert_tags[j]]
                    vert_coor[j,0] = vertex.x
                    vert_coor[j,1] = vertex.y
                element.assign_vertices(vert_coor)
                element.assign_vertex_tags(vert_tags)
                self.add_element(element)

        ### Assigning dof tags to vertices that are not on the boundary
        bound_verts.sort() # sorting boundary vertices list
        dof_counter = 0    # used for counting dof index
        for j in range(num_vert):
            if BinarySearch(bound_verts, j) == -1: # Testing if jth vertex is a boundary vertex
                self.vertices[j].assign_dof_tag(dof_counter)
                dof_counter += 1

        ### Generating vertex_to_dof map and dof_to_vertex map
        self.vertex_tag_to_dof_tag()

        ### Finding dof tags for the elements
        for j in range(len(self.elements)):
            self.elements[j].find_dof_tags(self.vtd)

    def gen_diff_ops(self):
        self.DIFF_OPS = Diff_ops_subobject(self)

class Plot_subObject:
    ### Sub object of FEM_mesh that handles plotting
    def __init__(self,mesh_obj):
        self.mesh_obj = mesh_obj
        self.vertices = self.mesh_obj.vertices
        self.elements = self.mesh_obj.elements
        #self.phys_regs = self.mesh_obj.phys_regs

    def PLOT_funcion(self,func):
        ### Plots the function, func, which is defined on the dof sites
        ### Doesn't fucking work!

        X = self.mesh_obj.X_dof
        Y = self.mesh_obj.Y_dof
        f = interpolate.interp2d(X, Y, func, kind='cubic')
        X_new = np.linspace(np.min(X), np.max(X),100)
        Y_new = np.linspace(np.min(Y), np.max(Y),50)
        xx, yy = np.meshgrid(X_new, Y_new)
        XX = xx.flatten(); YY = yy.flatten(); D = np.zeros(XX.size)
        for i in range(D.size):
            D[i] = f(XX[i],YY[i])
        plt.scatter(XX,YY,c = D,cmap = 'hot')
        plt.show()

    def PLOT_STATE(self,vec):
        self.Xg = self.mesh_obj.Xg
        self.Yg = self.mesh_obj.Yg
        idx_arrNan = self.mesh_obj.idx_arrNan
        idx_arrDof = self.mesh_obj.idx_arrDof

        VEC_full = np.zeros(self.Xg.shape[0]*self.Xg.shape[1])
        VEC_full[idx_arrNan] = 0. #np.nan
        VEC_full[idx_arrDof] = np.square(np.absolute(vec))
        V = vec_grid_create(self.Xg.shape[0],self.Xg.shape[1],VEC_full)
        #Vm = ma.masked_where(np.isnan(V),V)
        Vm = V
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.pcolormesh(self.Xg,self.Yg,Vm, cmap='hot',shading = 'gouraud')
        #ax.pcolormesh(self.Xg,self.Yg,Vm, cmap='hot')
        ax.set_aspect(1.0)
        plt.show()

    def PLOT_STATE2(self,vec):
        X = np.array(self.mesh_obj.X)[self.mesh_obj.idx_arrDof]
        Y = np.array(self.mesh_obj.Y)[self.mesh_obj.idx_arrDof]
        Z = np.square(np.absolute(vec))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        try:
            ax.tricontourf(X/10.,Y/10.,Z,1000,cmap = 'hot')
        except:
            ax.tripcolor(X/10.,Y/10.,Z,cmap = 'hot')
        ax.set_aspect(1.)
        plt.show()

    def PLOT_STATE_Lut(self,vec,bands = False,BdG = False):
        if not bands:
            vec_Sq = np.square(np.absolute(vec))
            if BdG:
                s = int(vec_Sq.shape[0]/2)
                vec_Sq = vec_Sq[:s] + vec_Sq[s:]
            Z = np.zeros(int(vec_Sq.size/4)); s = vec_Sq.size/4
            for i in range(4):
                Z = Z[:] + vec_Sq[i*s:(i+1)*s]
            X = np.array(self.mesh_obj.X)[self.mesh_obj.idx_arrDof]
            Y = np.array(self.mesh_obj.Y)[self.mesh_obj.idx_arrDof]
            #Z = np.square(np.absolute(vec))
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            try:
                ax.tricontourf(X/10.,Y/10.,Z,1000,cmap = 'hot')
            except:
                ax.tripcolor(X/10.,Y/10.,Z,cmap = 'hot')
            ax.set_aspect(1.)
            plt.show()
        else:
            vec_Sq = np.square(np.absolute(vec))
            if BdG:
                s = int(vec_Sq.shape[0]/2)
                vec_Sq = vec_Sq[:s] + vec_Sq[s:]
            fig = plt.figure()
            s = vec_Sq.size/4
            MAX = np.max(vec_Sq)
            for i in range(4):
                Z = vec_Sq[i*s:(i+1)*s]
                X = np.array(self.mesh_obj.X)[self.mesh_obj.idx_arrDof]
                Y = np.array(self.mesh_obj.Y)[self.mesh_obj.idx_arrDof]
                ax = fig.add_subplot(4,1,i+1)
                try:
                    ax.tricontourf(X/10.,Y/10.,Z,1000,cmap = 'hot',vmin = 0.,vmax= MAX)
                except:
                    ax.tripcolor(X/10.,Y/10.,Z,cmap = 'hot',vmin = 0.,vmax= MAX)
                ax.set_aspect(1.)
            plt.show()

    def PLOT_STATE3(self,vec,BdG = False):
        vec_Sq = np.square(np.absolute(vec))
        if BdG:
            s = int(vec_Sq.shape[0]/2)
            vec_Sq = vec_Sq[:s] + vec_Sq[s:]
        fig = plt.figure()
        s = int(vec_Sq.size/2)
        MAX = np.max(vec_Sq)
        Z = np.zeros(s)
        for i in range(2):
            Z = Z + vec_Sq[i*s:(i+1)*s]
        X = np.array(self.mesh_obj.X)[self.mesh_obj.idx_arrDof]
        Y = np.array(self.mesh_obj.Y)[self.mesh_obj.idx_arrDof]
        ax = fig.add_subplot(1,1,1)
        try:
            ax.tricontourf(X/10.,Y/10.,Z,1000,cmap = 'hot',vmin = 0.,vmax= MAX)
        except:
            ax.tripcolor(X/10.,Y/10.,Z,cmap = 'hot',vmin = 0.,vmax= MAX)
        #ax.set_aspect(1.)
        plt.show()

    def PLOT_Laplace(self,sol_arr):
        X = np.array(self.mesh_obj.X)
        Y = np.array(self.mesh_obj.Y)
        Z = sol_arr
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        try:
            pc = ax.tricontourf(X/10.,Y/10.,Z,1000,cmap = 'copper')
            #pc = ax.tricontourf(X/10.,Y/10.,Z,1000,cmap = 'gist_rainbow')
        except:
            pc = ax.tripcolor(X/10.,Y/10.,Z,cmap = 'copper')
            #pc = ax.tripcolor(X/10.,Y/10.,Z,cmap = 'gist_rainbow')
        fig.colorbar(pc, ax=ax)
        plt.show()

    def plot_vertices(self):
        X_bound = []; Y_bound = []
        X = []; Y = []
        for i in range(len(self.vertices)):
            vertex = self.vertices[i]
            if vertex.dof_bool: # vertex cooresponds to a degree of freedom site
                X.append(vertex.x); Y.append(vertex.y)
            else:
                X_bound.append(vertex.x); Y_bound.append(vertex.y)

        plt.scatter(X,Y, c = 'b')
        plt.scatter(X_bound,Y_bound, c = 'r')
        plt.show()

    def plot_elements(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        c = ['b','r','g']
        for i in range(len(self.elements)):
            print (len(self.elements) - i)
            elem = self.elements[i] # ith element of the mesh
            X = elem.vertices[:,0]
            Y = elem.vertices[:,1]
            #ax.plot([X[0],X[1]],[Y[0],Y[1]],c = c[elem.phys_tag-1])
            #ax.plot([X[0],X[2]],[Y[0],Y[2]],c = c[elem.phys_tag-1])
            #ax.plot([X[2],X[1]],[Y[2],Y[1]],c = c[elem.phys_tag-1])
            ax.plot([X[0],X[1]],[Y[0],Y[1]],c = c[0])
            ax.plot([X[0],X[2]],[Y[0],Y[2]],c = c[0])
            ax.plot([X[2],X[1]],[Y[2],Y[1]],c = c[0])
            #ax.scatter(elem.centroid[0],elem.centroid[1],c = c[elem.phys_tag-1])
        ax.grid()
        plt.show()

    def plot_elements2(self):
        self.phys_regs = self.mesh_obj.phys_regs
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        num_elems = len(self.elements)
        segments = np.zeros((3*num_elems,2,2))
        norm = plt.Normalize(0., 1.)
        colors = np.zeros(3*num_elems); c = np.linspace(0.,1.,len(self.phys_regs))
        for i in range(len(self.elements)):
            #print len(self.elements) - i
            elem = self.elements[i] # ith element of the mesh
            X = elem.vertices[:,0]
            Y = elem.vertices[:,1]
            segments[3*i+0,:,0] = np.array([X[0],X[1]])
            segments[3*i+0,:,1] = np.array([Y[0],Y[1]])
            segments[3*i+1,:,0] = np.array([X[0],X[2]])
            segments[3*i+1,:,1] = np.array([Y[0],Y[2]])
            segments[3*i+2,:,0] = np.array([X[1],X[2]])
            segments[3*i+2,:,1] = np.array([Y[1],Y[2]])
            for j in range(len(self.phys_regs)):
                if elem.phys_tag == self.phys_regs[j]:
                    idx = j;
                    break
                #else:
                #    print elem.phys_tag, self.phys_regs[j]
            colors[3*i:3*(i+1)] = c[idx]
        lc = LineCollection(segments, cmap='gist_rainbow', norm=norm)
        lc.set_array(colors)
        lc.set_linewidth(.5)
        line = ax.add_collection(lc)
        #ax.grid()
        ax.patch.set_facecolor('black')
        ax.set_xlim(np.min(segments[:,:,0]),np.max(segments[:,:,0]))
        ax.set_ylim(np.min(segments[:,:,1]),np.max(segments[:,:,1]))
        #ax.set_aspect(1.)
        plt.show()

    def plot_mesh(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        ### Plotting elements
        c = ['r','g','purple']
        print ('A')
        for i in range(len(self.elements)):
            elem = self.elements[i] # ith element of the mesh
            print (len(self.elements) - i, elem.phys_tag)
            X = elem.vertices[:,0]
            Y = elem.vertices[:,1]
            #ax.plot([X[0],X[1]],[Y[0],Y[1]],c = c[elem.phys_tag-1],linewidth = .5)
            #ax.plot([X[0],X[2]],[Y[0],Y[2]],c = c[elem.phys_tag-1],linewidth = .5)
            #ax.plot([X[2],X[1]],[Y[2],Y[1]],c = c[elem.phys_tag-1],linewidth = .5)
            ax.plot([X[0],X[1]],[Y[0],Y[1]],c = 'k',linewidth = .5)
            ax.plot([X[0],X[2]],[Y[0],Y[2]],c = 'k',linewidth = .5)
            ax.plot([X[2],X[1]],[Y[2],Y[1]],c = 'k',linewidth = .5)
            #ax.scatter(elem.centroid[0],elem.centroid[1],c = c[elem.phys_tag-1],marker = 's')
            ax.scatter(elem.centroid[0],elem.centroid[1],c = 'r',marker = 's')

        ### Plotting vertices
        X_bound = []; Y_bound = []
        X = []; Y = []
        for i in range(len(self.vertices)):
            vertex = self.vertices[i]
            if vertex.dof_bool: # vertex cooresponds to a degree of freedom site
                X.append(vertex.x); Y.append(vertex.y)
            else:
                X_bound.append(vertex.x); Y_bound.append(vertex.y)

        ax.scatter(X,Y, c = 'b',zorder =10)
        ax.scatter(X_bound,Y_bound, c = 'k',zorder = 10)
        ax.grid()
        plt.show()

    def plot_mesh2(self):
        self.phys_regs = self.mesh_obj.phys_regs
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        num_elems = len(self.elements)
        segments = np.zeros((3*num_elems,2,2))
        norm = plt.Normalize(0., 1.)
        colors = np.zeros(3*num_elems); c = np.linspace(0.,1.,len(self.phys_regs))
        for i in range(len(self.elements)):
            #print len(self.elements) - i
            elem = self.elements[i] # ith element of the mesh
            X = elem.vertices[:,0]
            Y = elem.vertices[:,1]
            segments[3*i+0,:,0] = np.array([X[0],X[1]])
            segments[3*i+0,:,1] = np.array([Y[0],Y[1]])
            segments[3*i+1,:,0] = np.array([X[0],X[2]])
            segments[3*i+1,:,1] = np.array([Y[0],Y[2]])
            segments[3*i+2,:,0] = np.array([X[1],X[2]])
            segments[3*i+2,:,1] = np.array([Y[1],Y[2]])
            for j in range(len(self.phys_regs)):
                if elem.phys_tag == self.phys_regs[j]:
                    idx = j;
                    break
                #else:
                #    print elem.phys_tag, self.phys_regs[j]
            colors[3*i:3*(i+1)] = c[idx]
        lc = LineCollection(segments, cmap='gist_rainbow', norm=norm)
        lc.set_array(colors)
        lc.set_linewidth(.5)
        line = ax.add_collection(lc)
        ax.grid()
        ax.patch.set_facecolor('black')
        xx = 10.; yy = 10.
        ax.set_xlim(np.min(segments[:,:,0])-xx,np.max(segments[:,:,0])+xx)
        ax.set_ylim(np.min(segments[:,:,1])-yy,np.max(segments[:,:,1])+yy)
        #plt.show()

        ### Plotting vertices
        X_bound = []; Y_bound = []
        X = []; Y = []
        for i in range(len(self.vertices)):
            vertex = self.vertices[i]
            if vertex.dof_bool: # vertex cooresponds to a degree of freedom site
                X.append(vertex.x); Y.append(vertex.y)
            else:
                X_bound.append(vertex.x); Y_bound.append(vertex.y)

        ax.scatter(X,Y, c = 'g',zorder =10, s= 10.)
        ax.scatter(X_bound,Y_bound, c = 'w',zorder = 10, s= 12.)
        ax.grid()
        ax.set_aspect(1.)
        plt.show()

class FEM_mtx_assembly_subobject:
    ### Sub object of FEM_mesh that handles assembling of matrices
    def __init__(self,mesh_obj):
        self.mesh_obj = mesh_obj
        self.vertices = self.mesh_obj.vertices
        self.elements = self.mesh_obj.elements

    def assemble_overlap_mtx(self):
        ### Creates the overlap matrix for functions corresponding to dof sites
        ###    * Note that you must match vertices to elements before running this function

        row = []; col = []; data = []

        ### Loop through all elements
        for n in range(len(self.elements)):
            elem = self.elements[n]
            ### Loop through the vertices of the nth element
            for i in range(3):
                dof_idx_i = elem.dof_tags[i] # dof index corresponding to the ith vertex of the nth element
                if dof_idx_i != -1: # Checking that the ith vertex corresponds to a dof site
                    for j in range(3):
                        dof_idx_j = elem.dof_tags[j] # dof index corresponding to the jth vertex of the nth element
                        if dof_idx_j != -1: # Checking that the jth vertex corresponds to a dof site
                            row.append(dof_idx_i); col.append(dof_idx_j); data.append(elem.overlap_mtx[i,j])
        N = self.mesh_obj.num_dof   # number of dof sites in mesh
        S = Spar.csc_matrix((data,(row,col)), shape=(N, N),dtype = 'complex')
        return S

    def assemble_overlap_mtx_mod(self,constants):
        ### Creates the overlap matrix times a constant for each region
        ### for functions corresponding to dof sites
        ###    * Note that you must match vertices to elements before running this function

        row = []; col = []; data = []

        ### Loop through all elements
        for n in range(len(self.elements)):
            elem = self.elements[n]
            phys_tag = elem.phys_tag - 101
            ### Loop through the vertices of the nth element
            for i in range(3):
                dof_idx_i = elem.dof_tags[i] # dof index corresponding to the ith vertex of the nth element
                if dof_idx_i != -1: # Checking that the ith vertex corresponds to a dof site
                    for j in range(3):
                        dof_idx_j = elem.dof_tags[j] # dof index corresponding to the jth vertex of the nth element
                        if dof_idx_j != -1: # Checking that the jth vertex corresponds to a dof site
                            row.append(dof_idx_i); col.append(dof_idx_j); data.append(elem.overlap_mtx[i,j]*constants[phys_tag])
        N = self.mesh_obj.num_dof   # number of dof sites in mesh
        S = Spar.csc_matrix((data,(row,col)), shape=(N, N),dtype = 'complex')
        return S

    def assemble_y_mtx_mod(self,constants):
        ### Creates the y matrix times a constant for each region
        ### for functions corresponding to dof sites
        ###    * Note that you must match vertices to elements before running this function

        row = []; col = []; data = []

        ### Loop through all elements
        for n in range(len(self.elements)):
            elem = self.elements[n]
            phys_tag = elem.phys_tag - 101
            ### Loop through the vertices of the nth element
            for i in range(3):
                dof_idx_i = elem.dof_tags[i] # dof index corresponding to the ith vertex of the nth element
                if dof_idx_i != -1: # Checking that the ith vertex corresponds to a dof site
                    for j in range(3):
                        dof_idx_j = elem.dof_tags[j] # dof index corresponding to the jth vertex of the nth element
                        if dof_idx_j != -1: # Checking that the jth vertex corresponds to a dof site
                            row.append(dof_idx_i); col.append(dof_idx_j); data.append(elem.y_mtx[i,j]*constants[phys_tag])
        N = self.mesh_obj.num_dof   # number of dof sites in mesh
        S = Spar.csc_matrix((data,(row,col)), shape=(N, N),dtype = 'complex')
        return S

    def assemble_x_mtx_mod(self,constants):
        ### Creates the x matrix times a constant for each region
        ### for functions corresponding to dof sites
        ###    * Note that you must match vertices to elements before running this function

        row = []; col = []; data = []

        ### Loop through all elements
        for n in range(len(self.elements)):
            elem = self.elements[n]
            phys_tag = elem.phys_tag - 101
            ### Loop through the vertices of the nth element
            for i in range(3):
                dof_idx_i = elem.dof_tags[i] # dof index corresponding to the ith vertex of the nth element
                if dof_idx_i != -1: # Checking that the ith vertex corresponds to a dof site
                    for j in range(3):
                        dof_idx_j = elem.dof_tags[j] # dof index corresponding to the jth vertex of the nth element
                        if dof_idx_j != -1: # Checking that the jth vertex corresponds to a dof site
                            row.append(dof_idx_i); col.append(dof_idx_j); data.append(elem.x_mtx[i,j]*constants[phys_tag])
        N = self.mesh_obj.num_dof   # number of dof sites in mesh
        S = Spar.csc_matrix((data,(row,col)), shape=(N, N),dtype = 'complex')
        return S

    def assemble_neg_Lap_mtx(self):
        ### Creates the negative Laplacian matrix for functions corresponding to dof sites
        ###    * Note that you must match vertices to elements before running this function

        row = []; col = []; data = []

        ### Loop through all elements
        for n in range(len(self.elements)):
            elem = self.elements[n]
            ### Loop through the vertices of the nth element
            for i in range(3):
                dof_idx_i = elem.dof_tags[i] # dof index corresponding to the ith vertex of the nth element
                if dof_idx_i != -1: # Checking that the ith vertex corresponds to a dof site
                    for j in range(3):
                        dof_idx_j = elem.dof_tags[j] # dof index corresponding to the jth vertex of the nth element
                        if dof_idx_j != -1: # Checking that the jth vertex corresponds to a dof site
                            row.append(dof_idx_i); col.append(dof_idx_j); data.append(elem.nLap_mtx[i,j])
        N = self.mesh_obj.num_dof   # number of dof sites in mesh
        A = Spar.csc_matrix((data,(row,col)), shape=(N, N),dtype = 'complex')
        return A

    def assemble_neg_Lap_mtx_alt(self,m_eff):
        ### Creates the negative Laplacian matrix times the effective mass of each region
        ### for functions corresponding to dof sites
        ###    * Note that you must match vertices to elements before running this function
        row = []; col = []; data = []
        m_rep = []
        for i in range(len(m_eff)):
            if abs(m_eff[i]) < 10. **(-10):
                m_rep.append(0.)
            else:
                m_rep.append(1./m_eff[i])

        ### Loop through all elements
        for n in range(len(self.elements)):
            elem = self.elements[n]
            phys_tag = elem.phys_tag - 101
            ### Loop through the vertices of the nth element
            for i in range(3):
                dof_idx_i = elem.dof_tags[i] # dof index corresponding to the ith vertex of the nth element
                if dof_idx_i != -1: # Checking that the ith vertex corresponds to a dof site
                    for j in range(3):
                        dof_idx_j = elem.dof_tags[j] # dof index corresponding to the jth vertex of the nth element
                        if dof_idx_j != -1: # Checking that the jth vertex corresponds to a dof site
                            row.append(dof_idx_i); col.append(dof_idx_j); data.append(elem.nLap_mtx[i,j]*m_rep[phys_tag])
        N = self.mesh_obj.num_dof   # number of dof sites in mesh
        A = Spar.csc_matrix((data,(row,col)), shape=(N, N),dtype = 'complex')
        return A

    def assemble_kxSq_mtx_alt(self,factors):
        ### Creates the kx^2 matrix times of a factor for each physical region
        ### for functions corresponding to dof sites
        ###    * Note that you must match vertices to elements before running this function
        row = []; col = []; data = []

        ### Loop through all elements
        for n in range(len(self.elements)):
            elem = self.elements[n]
            phys_tag = elem.phys_tag - 101
            ### Loop through the vertices of the nth element
            for i in range(3):
                dof_idx_i = elem.dof_tags[i] # dof index corresponding to the ith vertex of the nth element
                if dof_idx_i != -1: # Checking that the ith vertex corresponds to a dof site
                    for j in range(3):
                        dof_idx_j = elem.dof_tags[j] # dof index corresponding to the jth vertex of the nth element
                        if dof_idx_j != -1: # Checking that the jth vertex corresponds to a dof site
                            row.append(dof_idx_i); col.append(dof_idx_j); data.append(elem.kxSq_mtx[i,j]*factors[phys_tag])
        N = self.mesh_obj.num_dof   # number of dof sites in mesh
        A = Spar.csc_matrix((data,(row,col)), shape=(N, N),dtype = 'complex')
        return A

    def assemble_kySq_mtx_alt(self,factors):
        ### Creates the ky^2 matrix times of a factor for each physical region
        ### for functions corresponding to dof sites
        ###    * Note that you must match vertices to elements before running this function
        row = []; col = []; data = []

        ### Loop through all elements
        for n in range(len(self.elements)):
            elem = self.elements[n]
            phys_tag = elem.phys_tag - 101
            ### Loop through the vertices of the nth element
            for i in range(3):
                dof_idx_i = elem.dof_tags[i] # dof index corresponding to the ith vertex of the nth element
                if dof_idx_i != -1: # Checking that the ith vertex corresponds to a dof site
                    for j in range(3):
                        dof_idx_j = elem.dof_tags[j] # dof index corresponding to the jth vertex of the nth element
                        if dof_idx_j != -1: # Checking that the jth vertex corresponds to a dof site
                            row.append(dof_idx_i); col.append(dof_idx_j); data.append(elem.kySq_mtx[i,j]*factors[phys_tag])
        N = self.mesh_obj.num_dof   # number of dof sites in mesh
        A = Spar.csc_matrix((data,(row,col)), shape=(N, N),dtype = 'complex')
        return A

    def assemble_kxky_mtx_alt(self,factors):
        ### Creates the kxky matrix times of a factor for each physical region
        ### for functions corresponding to dof sites
        ###    * Note that you must match vertices to elements before running this function
        row = []; col = []; data = []

        ### Loop through all elements
        for n in range(len(self.elements)):
            elem = self.elements[n]
            phys_tag = elem.phys_tag - 101
            ### Loop through the vertices of the nth element
            for i in range(3):
                dof_idx_i = elem.dof_tags[i] # dof index corresponding to the ith vertex of the nth element
                if dof_idx_i != -1: # Checking that the ith vertex corresponds to a dof site
                    for j in range(3):
                        dof_idx_j = elem.dof_tags[j] # dof index corresponding to the jth vertex of the nth element
                        if dof_idx_j != -1: # Checking that the jth vertex corresponds to a dof site
                            row.append(dof_idx_i); col.append(dof_idx_j); data.append(elem.kxky_mtx[i,j]*factors[phys_tag])
        N = self.mesh_obj.num_dof   # number of dof sites in mesh
        A = Spar.csc_matrix((data,(row,col)), shape=(N, N),dtype = 'complex')
        return A

    def assemble_kx_mtx_alt(self,factors):
        ### Creates the kx matrix times of a factor for each physical region
        ### for functions corresponding to dof sites
        ###    * Note that you must match vertices to elements before running this function
        row = []; col = []; data = []

        ### Loop through all elements
        for n in range(len(self.elements)):
            elem = self.elements[n]
            phys_tag = elem.phys_tag - 101
            ### Loop through the vertices of the nth element
            for i in range(3):
                dof_idx_i = elem.dof_tags[i] # dof index corresponding to the ith vertex of the nth element
                if dof_idx_i != -1: # Checking that the ith vertex corresponds to a dof site
                    for j in range(3):
                        dof_idx_j = elem.dof_tags[j] # dof index corresponding to the jth vertex of the nth element
                        if dof_idx_j != -1: # Checking that the jth vertex corresponds to a dof site
                            row.append(dof_idx_i); col.append(dof_idx_j); data.append(elem.kx_mtx[i,j]*factors[phys_tag])
        N = self.mesh_obj.num_dof   # number of dof sites in mesh
        A = Spar.csc_matrix((data,(row,col)), shape=(N, N),dtype = 'complex')
        return A

    def assemble_ky_mtx_alt(self,factors):
        ### Creates the ky matrix times of a factor for each physical region
        ### for functions corresponding to dof sites
        ###    * Note that you must match vertices to elements before running this function
        row = []; col = []; data = []

        ### Loop through all elements
        for n in range(len(self.elements)):
            elem = self.elements[n]
            phys_tag = elem.phys_tag - 101
            ### Loop through the vertices of the nth element
            for i in range(3):
                dof_idx_i = elem.dof_tags[i] # dof index corresponding to the ith vertex of the nth element
                if dof_idx_i != -1: # Checking that the ith vertex corresponds to a dof site
                    for j in range(3):
                        dof_idx_j = elem.dof_tags[j] # dof index corresponding to the jth vertex of the nth element
                        if dof_idx_j != -1: # Checking that the jth vertex corresponds to a dof site
                            row.append(dof_idx_i); col.append(dof_idx_j); data.append(elem.ky_mtx[i,j]*factors[phys_tag])
        N = self.mesh_obj.num_dof   # number of dof sites in mesh
        A = Spar.csc_matrix((data,(row,col)), shape=(N, N),dtype = 'complex')
        return A

    def assemble_interface_pot_mtx(self,ax,dof_int_idx):
        ### Creates the matrix representation of a dirac delta potential of unit strength along
        ### an interface assumed to be parallel to the x-direction
        ###     * dof_int_idx are the dof indices of the vertices on the SM-SC interface

        row = []; col = []; data = []
        for i in range(dof_int_idx.size):
            idx = dof_int_idx[i]
            row.append(idx); col.append(idx); data.append(2. * ax/3.)
            if i != 0:
                row.append(dof_int_idx[i-1]); col.append(idx); data.append(ax/6.)
            if i != dof_int_idx.size - 1:
                row.append(dof_int_idx[i+1]); col.append(idx); data.append(ax/6.)
        N = self.mesh_obj.num_dof   # number of dof sites in mesh
        A = Spar.csc_matrix((data,(row,col)), shape=(N, N),dtype = 'complex')
        return A

class Diff_ops_subobject:
    ### Sub_object of FEM_mesh that will store the differential operators matrix representatives
    def __init__(self,mesh_obj):
        self.mesh_obj = mesh_obj
        self.gen_ops()

    def gen_ops(self):
        self.Diag_1 = self.mesh_obj.MTX_assembly.assemble_overlap_mtx_mod([1.,0.,0.])
        self.Diag_2 = self.mesh_obj.MTX_assembly.assemble_overlap_mtx_mod([0.,1.,0.])
        self.Diag_3 = self.mesh_obj.MTX_assembly.assemble_overlap_mtx_mod([0.,0.,1.])

        self.kx = self.mesh_obj.MTX_assembly.assemble_kx_mtx_alt([1.,1.,1.])
        self.ky = self.mesh_obj.MTX_assembly.assemble_ky_mtx_alt([1.,1.,1.])
        self.kxky = self.mesh_obj.MTX_assembly.assemble_kxky_mtx_alt([1.,1.,1.])
        self.kxSq = self.mesh_obj.MTX_assembly.assemble_kxSq_mtx_alt([1.,1.,1.])
        self.kySq = self.mesh_obj.MTX_assembly.assemble_kySq_mtx_alt([1.,1.,1.])
        self.Diag = self.mesh_obj.MTX_assembly.assemble_overlap_mtx_mod([1.,1.,1.])








def vec_grid_create(Nx,Ny,vec):
    C = np.zeros((Nx,Ny))
    for i in range(Nx):
        for j in range(Ny):
            C[i,j] = vec[i+j*Nx]
    return C
