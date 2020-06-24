"""
    Contains the class FEM_element, which is a finite element instance of the 2D
    triangular type
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import FEM_integrals as FINT
import FEM_vertex_class as FVC

class FEM_element:
    def __init__(self):
        self.phys_tag = None    # physical tag of an element
        self.vertices = np.empty((3,2)); self.vertices[:] = np.nan    # numpy array (3,2) of coordinates of the vertices of the element
        self.vertex_tags = np.empty(3); self.vertex_tags[:] = np.nan # numpy int array (3) for the tags of the vertices of the element

    def assign_phys_tag(self,phys_tag):
        ### Assigns the physical tag (i.e. SC, SM, MI, etc.) to the element

        if (type(phys_tag) == int) or (type(phys_tag) == long):
            self.phys_tag = phys_tag
        else:
            print ('\n' + '*' * 50 + '\n'); print ('phys_tag of an element must be an int or long')
            print ('type(phys_tag): ', type(phys_tag))
            print ('\n' + '*' * 50); sys.exit()

    def assign_vertices(self,vertices):
        ### Assigns the 3 vertices of the element
        ### Also calculates the coefficients (a,b,c) for the linear functions of the element
        ### Also calculates several polynomial integrals on the element
        ### Also calculates the centroid of the triangle

        ### Assign vertices
        if type(vertices) == np.ndarray: # Check data type
            if vertices.shape == (3,2):   # Check shape of vertices
                self.vertices = vertices
            else:
                print ('\n' + '*' * 50 + '\n'); print ('vertices of an element must be a numpy.ndarray of shape (3,2)')
                print ('type(vertices): '+ str(type(vertices)))
                print ('vertices.shape: '+ str(vertices.shape))
                print ('\n' + '*' * 50); sys.exit()
        else:
            print ('\n' + '*' * 50 + '\n'); print ('vertices of an element must be a numpy.ndarray of shape (3,2)')
            print ('type(vertices): '+ str(type(vertices)))
            print ('\n' + '*' * 50); sys.exit()

        ### Calculate linear function coefficients
        self.a, self.b, self.c = FINT.element_funcs(self.vertices[:,0],self.vertices[:,1])

        ### Calculate polynomial integrals on element
        self.integral_arr = FINT.element_integrals(self.vertices[:,0],self.vertices[:,1])

        ### Calculate the centroid of the triangle
        self.centroid = np.array([np.sum(self.vertices[:,0])/3.,np.sum(self.vertices[:,1])/3.])

        ### Creating subobject for calculating matrix elements
        self.MTX_elems = FEM_element_mtx_elements_subobject(self)

    def assign_vertex_tags(self,vertex_tags):
        ### Assigns the tags of the vertices, i.e. what is the index in the vertex array for each of the vertices of the element

        if type(vertex_tags) == np.ndarray: # Check data type
            if vertex_tags.shape == (3,):     # Check shape of vertex_tags
                self.vertex_tags = vertex_tags
            else:
                print ('\n' + '*' * 50 + '\n'); print ('vertex_tags of an element must be a numpy.ndarray of shape (3,)')
                print ('type(vertex_tags): ', type(vertex_tags))
                print ('vertex_tags.shape: ', vertex_tags.shape)
                print ('\n' + '*' * 50); sys.exit()
        else:
            print ('\n' + '*' * 50 + '\n'); print ('vertex_tags of an element must be a numpy.ndarray of shape (3,)')
            print ('type(vertex_tags): ', type(vertex_tags))
            print ('\n' + '*' * 50); sys.exit()

    def find_dof_tags(self,vtd_map):
        ### Using the vertex_tags of the vertices and the vertex to dof map, this assigns dof tags of the element's vertices
        ###     * If a vertex doesn't map to a dof, then the dof_tag should be -1
        #print vtd_map.shape, self.vertex_tags
        self.dof_tags = vtd_map[self.vertex_tags]

    def plot_element(self):
        X = self.vertices[:,0]
        Y = self.vertices[:,1]
        plt.scatter(self.vertices[:,0],self.vertices[:,1],c = 'b')
        plt.scatter(self.centroid[0],self.centroid[1],c = 'r')

        plt.plot([X[0],X[1]],[Y[0],Y[1]],c = 'b')
        plt.plot([X[0],X[2]],[Y[0],Y[2]],c = 'b')
        plt.plot([X[2],X[1]],[Y[2],Y[1]],c = 'b')
        plt.grid()
        plt.show()

    def are_equal(self,element2):
        ### Tests if element2 is the same as the current element
        if isinstance(element2,FEM_element):
            if not (np.all(np.isnan(self.vertices))) :
                if (np.all(np.isnan(element2.vertices))):
                    return False

                perms = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
                for i in range(len(perms)):
                    #print '\n', perms[i]
                    #for j in range(3):
                    #    print self.vertices[j,:], element2.vertices[perms[i][j],:]
                    if np.array_equal(self.vertices,element2.vertices[perms[i],:]):
                        return True
                return False
            else:
                if (np.all(np.isnan(element2.vertices))):
                    return True
                else:
                    return False
        else:
            print ('\n\n!WARNING!: element2 is not an instance of FEM_element...returning False\n\n')
            return False

    def vertex_within(self,vertex,return_idx = False):
        ### Tests if FEM_vertex instance is one of the element's vertices
        if isinstance(vertex,FVC.FEM_vertex):
            if (np.all(np.isnan(self.vertices))): # element has no vertices yet assigned
                if return_idx:
                    return False, -1
                else:
                    return False
            else:
                for i in range(3):
                    if np.array_equal(vertex.coor,self.vertices[i,:]):
                        if return_idx:
                            return True, i
                        else:
                            return True
                if return_idx:
                    return False, -1
                else:
                    return False
        else:
            print ('\n\n!WARNING!: vertex is not an instance of FEM_vertex...returning False\n\n')
            if return_idx:
                return False, -1
            else:
                return False

    def find_vertices(self,vertices):
        ### search the list of vertices for vertices that belong to the element
        vertex_tags = -1*np.ones(3,dtype = 'int')
        counter = 0
        for i in range(len(vertices)):
            vertex = vertices[i]
            bool, idx = self.vertex_within(vertex,return_idx=True)
            if bool:
                vertex.add_element(self)
                vertex_tags[idx] = vertex.vertex_tag
                counter += 1
            if counter == 3:
                break
        self.assign_vertex_tags(vertex_tags)

class FEM_element_mtx_elements_subobject:
    ### Sub object of an FEM_element instance that deals with computing matrix elements on that element
    def __init__(self,elem_obj):
        self.elem_obj = elem_obj # instance of FEM_element (parent object)

        ### Linear functions defined on the element
        self.a = self.elem_obj.a
        self.b = self.elem_obj.b
        self.c = self.elem_obj.c

        self.integral_arr = self.elem_obj.integral_arr # array of polynomial integrals defined on the element

        self.overlap_matrix_calc() # Calculate the linear function overlap matrix
        self.neg_Laplacian_matrix_calc() # Calculate the negative Laplacian matrix
        self.y_matrix_calc()
        self.x_matrix_calc()
        self.kx_matrix_calc()
        self.ky_matrix_calc()
        self.kxSq_matrix_calc()
        self.kySq_matrix_calc()
        self.kxky_matrix_calc()

    def overlap_matrix_calc(self):
        ### creates array of the overlap matrix elements between the functions of the element
        ### overlap_mtx[0,2] = overlap of first and third linear functions for examples
        self.overlap_mtx = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                self.overlap_mtx[i,j] = self.integrate_overlap(i,j)
        self.elem_obj.overlap_mtx = self.overlap_mtx

    def integrate_overlap(self,vertex1_idx,vertex2_idx):
        ### Integrates the two linear functions associated with the two vertices, vertex1 and vertex2
        ### on this element
        i = vertex1_idx; j = vertex2_idx
        a = self.a; b = self.b; c = self.c # coefficients of the linear functions of the element
        int_arr = self.integral_arr # integrals of polynomials over the element
        return (a[i]*a[j]*int_arr[0] + (a[i]*b[j]+a[j]*b[i])*int_arr[1] + (a[i]*c[j]+a[j]*c[i])*int_arr[2]
             + (b[i]*c[j]+b[j]*c[i])*int_arr[3] + b[i]*b[j]*int_arr[4] + c[i]*c[j]*int_arr[5])

    def neg_Laplacian_matrix_calc(self):
        ### Creates array of the matrix elements of the negative 2D Laplacian
        ### nLap_mtx[0,2] = matrix element <0|-Laplacian|2>, where |2> is the third linear function of the element

        self.nLap_mtx = np.zeros((3,3))
        a = self.a; b = self.b; c = self.c # coefficients of the linear functions of the element
        Area = self.integral_arr[0] # area of the element
        for i in range(3):
            for j in range(3):
                self.nLap_mtx[i,j] = (b[i]*b[j] + c[i]*c[j]) * Area
        self.elem_obj.nLap_mtx = self.nLap_mtx

    def y_matrix_calc(self):
        ### Calculates y matrix elements (<i|y|j>, where |j> is the jth linear function of the element)

        self.y_mtx = np.zeros((3,3))
        a = self.a; b = self.b; c = self.c # coefficients of the linear functions of the element
        INT = self.integral_arr
        for i in range(3):
            for j in range(3):
                v = a[i]*a[j]*INT[2] + (a[i]*b[j]+a[j]*b[i])*INT[3] + (a[i]*c[j]+a[j]*c[i])*INT[5] \
                   +b[i]*b[j]*INT[7] + (b[i]*c[j]+b[j]*c[i])*INT[8] + c[i]*c[j]*INT[9]
                self.y_mtx[i,j] = v
        self.elem_obj.y_mtx = self.y_mtx

    def x_matrix_calc(self):
        ### Calculates x matrix elements (<i|x|j>, where |j> is the jth linear function of the element)

        self.x_mtx = np.zeros((3,3))
        a = self.a; b = self.b; c = self.c # coefficients of the linear functions of the element
        INT = self.integral_arr
        for i in range(3):
            for j in range(3):
                v = a[i]*a[j]*INT[1] + (a[i]*b[j]+a[j]*b[i])*INT[4] + (a[i]*c[j]+a[j]*c[i])*INT[3] \
                   +b[i]*b[j]*INT[6] + (b[i]*c[j]+b[j]*c[i])*INT[7] + c[i]*c[j]*INT[8]
                self.x_mtx[i,j] = v
        self.elem_obj.x_mtx = self.x_mtx

    def kxSq_matrix_calc(self):
        ### Creates array of the matrix elements of the kx^2 operator
        ### kxSq_mtx[0,2] = matrix element <0|kx^2|2>, where |2> is the third linear function of the element

        self.kxSq_mtx = np.zeros((3,3))
        a = self.a; b = self.b; c = self.c # coefficients of the linear functions of the element
        Area = self.integral_arr[0] # area of the element
        for i in range(3):
            for j in range(3):
                self.kxSq_mtx[i,j] = (b[i]*b[j]) * Area
        #print '\n', self.kxSq_mtx
        self.elem_obj.kxSq_mtx = self.kxSq_mtx

    def kySq_matrix_calc(self):
        ### Creates array of the matrix elements of the kx^2 operator
        ### kySq_mtx[0,2] = matrix element <0|ky^2|2>, where |2> is the third linear function of the element

        self.kySq_mtx = np.zeros((3,3))
        a = self.a; b = self.b; c = self.c # coefficients of the linear functions of the element
        Area = self.integral_arr[0] # area of the element
        for i in range(3):
            for j in range(3):
                self.kySq_mtx[i,j] = (c[i]*c[j]) * Area
        self.elem_obj.kySq_mtx = self.kySq_mtx

    def kxky_matrix_calc(self):
        ### Creates array of the matrix elements of the kx^2 operator
        ### kxky_mtx[0,2] = matrix element <0|kxky|2>, where |2> is the third linear function of the element

        self.kxky_mtx = np.zeros((3,3))
        a = self.a; b = self.b; c = self.c # coefficients of the linear functions of the element
        Area = self.integral_arr[0] # area of the element
        for i in range(3):
            for j in range(3):
                self.kxky_mtx[i,j] = .5*(b[i]*c[j]+b[j]*c[i]) * Area
        self.elem_obj.kxky_mtx = self.kxky_mtx

    def kx_matrix_calc(self):
        ### Creates array of the matrix elements of the kx^2 operator
        ### kx_mtx[0,2] = matrix element <0|kx|2>, where |2> is the third linear function of the element

        self.kx_mtx = np.zeros((3,3),dtype = 'complex')
        a = self.a; b = self.b; c = self.c # coefficients of the linear functions of the element
        Area = self.integral_arr[0] # area of the element
        for i in range(3):
            for j in range(3):
                self.kx_mtx[i,j] = (-1j/2.)*(a[i]*b[j] - b[i]*a[j]) * Area - (1j/2.)*(c[i]*b[j] - b[i]*c[j])*self.integral_arr[2]
        self.elem_obj.kx_mtx = self.kx_mtx

    def ky_matrix_calc(self):
        ### Creates array of the matrix elements of the kx^2 operator
        ### ky_mtx[0,2] = matrix element <0|ky|2>, where |2> is the third linear function of the element

        self.ky_mtx = np.zeros((3,3),dtype = 'complex')
        a = self.a; b = self.b; c = self.c # coefficients of the linear functions of the element
        Area = self.integral_arr[0] # area of the element
        for i in range(3):
            for j in range(3):
                self.ky_mtx[i,j] = (-1j/2.)*(a[i]*c[j] - c[i]*a[j]) * Area - (1j/2.)*(b[i]*c[j] - c[i]*b[j])*self.integral_arr[1]
        self.elem_obj.ky_mtx = self.ky_mtx

### Testing
if False:

    elem = FEM_element()
    ver = np.array([ [0.,0.], [1.2,.1], [.5,.8] ])
    elem.assign_vertices(ver)
    elem.assign_phys_tag(2)
    elem.assign_vertex_tags(np.array([1,2,6],dtype = 'int'))
    #elem.plot_element()
    #print isinstance(elem,FEM_element)
    #print isinstance(ver,FEM_element)

    elem2 = FEM_element()
    ver2 = np.array([ [0.,0.], [1.2,.1], [.5,.8] ])
    elem2.assign_vertices(ver2)

    #print elem.are_equal(elem2)


    vert = FVC.FEM_vertex(1.2,.1)
    print (elem.vertex_within(vert))
    #print elem.vertex_within(0.)

    vert.add_element(elem)
