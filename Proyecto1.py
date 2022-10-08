import struct
from progress.bar import Bar
from numpy import sin, cos

r = ObjetoRender()
r.glInit()
r.glCreateWindow(1000, 1000)
r.glViewPort(0, 0, 1000, 1000)

# r.lookAt(VectorTridimensional(0, 0, 50), VectorTridimensional(0, 0, 0), VectorTridimensional(0, 1, 1))
r.lookAt(VectorTridimensional(0, 0, 100), VectorTridimensional(0, 0, 0), VectorTridimensional(0, 2, 2))
print("Background:")
background = Texture('./materiales/fondo.bmp')
r.pixels = background.pixels

print("Mascara 1:")
texturaMascara1 = Texture('./materiales/mascara1.bmp')
r.load('./materiales/mascara1.obj', translate=(0, 0, 0),
       scale=(4, 4, 4), rotate=(0, 0, 0), texture=texturaMascara1)

print("Mascara 2:")
texturaMascara2 = Texture('./materiales/mascara2.bmp')
r.load('./materiales/mascara2.obj', translate=(-0.9, 0, 0),
       scale=(4, 4, 4), rotate=(0, 0, 0), texture=texturaMascara2)

print("Mascara 3:")
texturaMascara3 = Texture('./materiales/mascara3.bmp')
r.load('./materiales/mascara3.obj', translate=(4.3, -2.3, 0),
       scale=(2, 2, 2), rotate=(0, 0, 0), texture=texturaMascara3)

print("Escalera:")
texturaEscalera = Texture('./materiales/escalera.bmp')
r.load('./materiales/escalera.obj', translate=(-1, -3.92, 0),
       scale=(0.9, 0.9, 0.9), rotate=(0, 0, 0), texture=texturaEscalera)

print("Table:")
texturaTable = Texture('./materiales/table.bmp')
r.load('./materiales/table.obj', translate=(-0.4, -2.4, 0),
       scale=(0.3, 0.3, 0.3), rotate=(2, 0, 0), texture=texturaTable)

r.glFinish()


V2 = variableTipoTupla('Point2D', ['x', 'y'])
VectorTridimensional = variableTipoTupla('Point3D', ['x', 'y', 'z'])
V4 = variableTipoTupla('Point4D', ['x', 'y', 'z', 'w'])


def sum(v0, v1):
    return VectorTridimensional(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)


def sub(v0, v1):
    return VectorTridimensional(
        v0.x - v1.x,
        v0.y - v1.y,
        v0.z - v1.z
    )


def subVectors(vec1, vec2):
    subList = []
    subList.extend((vec1[0] - vec2[0], vec1[1] - vec2[1], vec1[2] - vec2[2]))
    return subList


def mul(v0, k):
    return VectorTridimensional(v0.x * k, v0.y * k, v0.z * k)


def multiply(dotNumber, normal):
    arrMul = []
    arrMul.extend(
        (dotNumber * normal[0], dotNumber * normal[1], dotNumber * normal[2]))
    return arrMul


def dot(v0, v1):
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z


def dot2(norm, lX, lY, lZ):
    return ((norm[0] * lX) + (norm[1] * lY) + (norm[2] * lZ))


def cross(v0, v1):
    cx = v0.y * v1.z - v0.z * v1.y
    cy = v0.z * v1.x - v0.x * v1.z
    cz = v0.x * v1.y - v0.y * v1.x
    return VectorTridimensional(cx, cy, cz)


def cross0(v0, v1):
    arr_cross = []
    arr_cross.extend((v0[1] * v1[2] - v1[1] * v0[2], -(v0[0]
                     * v1[2] - v1[0] * v0[2]), v0[0] * v1[1] - v1[0] * v0[1]))
    return arr_cross


def length(v0):
    return (v0.x**2 + v0.y**2 + v0.z**2)**0.5


def norm(v0):
    l = length(v0)

    if l == 0:
        return VectorTridimensional(0, 0, 0)

    return VectorTridimensional(
        v0.x/l,
        v0.y/l,
        v0.z/l
    )


def baricentriBox(A, B, C):
    xs = [A.x, B.x, C.x]
    xs.sort()
    ys = [A.y, B.y, C.y]
    ys.sort()
    return xs[0], xs[-1], ys[0], ys[-1]


def barycentric(A, B, C, P):
    bary = cross(
        VectorTridimensional(C.x - A.x, B.x - A.x, A.x - P.x),
        VectorTridimensional(C.y - A.y, B.y - A.y, A.y - P.y)
    )

    if abs(bary[2]) < 1:
        return -1, -1, -1

    return (
        1 - (bary[0] + bary[1]) / bary[2],
        bary[1] / bary[2],
        bary[0] / bary[2]
    )


def div(v0, norm):
    if (norm == 0):
        arr0_norm = []
        arr0_norm.extend((0, 0, 0))
        return arr0_norm
    else:
        arr_div = []
        arr_div.extend((v0[0] / norm, v0[1] / norm, v0[2] / norm))
        return arr_div


def frobeniusNorm(v0):
    return ((v0[0]**2 + v0[1]**2 + v0[2]**2)**(1/2))


def sub2(x0, x1, y0, y1):
    arr_sub = []
    arr_sub.extend((x0 - x1, y0 - y1))
    return arr_sub


def sub3(x0, x1, y0, y1, z0, z1):
    arr_sub = []
    arr_sub.extend((x0 - x1, y0 - y1, z0 - z1))
    return arr_sub


def multiplyVM(v, m):
    result = []
    for i in range(len(m)):
        total = 0
        for j in range(len(v)):
            total += m[i][j] * v[j]
        result.append(total)
    return result


def char(c):
    return struct.pack('=c', c.encode('ascii'))


def word(w):
    return struct.pack('=h', w)


def dword(d):
    return struct.pack('=l', d)


def color(r, g, b):
    return bytes([int(b), int(g), int(r)])

class Texture(object):
    def __init__(self, path):
        self.path = path
        self.read()

    def read(self):
        image = open(self.path, "rb")
        image.seek(2 + 4 + 4) 
        header_size = struct.unpack("=l", image.read(4))[0]  
        image.seek(2 + 4 + 4 + 4 + 4)
        
        self.width = struct.unpack("=l", image.read(4))[0]  
        self.height = struct.unpack("=l", image.read(4))[0]  
        self.pixels = []
        image.seek(header_size)

        bar = Bar('Llenando pixeles: ', max=self.height)
        for y in range(self.height):
            self.pixels.append([])
            for x in range(self.width):
                b = ord(image.read(1))
                g = ord(image.read(1))
                r = ord(image.read(1))
                
                self.pixels[y].append(color(r,g,b))
            bar.next()
        bar.finish()
        image.close()

    def get_color(self, tx, ty):
        x = int(tx * self.width)
        y = int(ty * self.height)
        
        return self.pixels[y][x]

    def get_color_with_intensity(self, tx, ty, intensity=1):
        x = round(tx * self.width)
        y = round(ty * self.height)

        
        int_values = [temp for temp in (self.pixels[y][x])]

        b = round(int_values[0] * intensity)
        g = round(int_values[1] * intensity)
        r = round(int_values[2] * intensity)

        if(r<0):
            r = 0

        if(g<0):
            g = 0

        if(b<0):
            b = 0

        return color(r,g,b)
    
class Obj(object):
    def __init__(self, filename):
        with open(filename) as f:
            self.lines = f.read().splitlines()

        self.vertices = []
        self.tvertices = []
        self.nvertices = []
        self.faces = []
        self.read()

    def read(self):
        for line in self.lines:
            if line:
                prefix, value = line.split(' ', 1)

                if prefix == 'v':
                    self.vertices.append(list(map(float, value.split(' '))))
                if prefix == 'f':
                    self.faces.append([list(map(int , face.split('/'))) for face in value.split(' ')])
                if prefix == 'vt':
                    vertice = list(map(float, value.split(' ')))

                    if(len(vertice) == 2):
                        vertice.append(0)

                    self.tvertices.append(vertice)  
                if prefix == 'vn':
                    self.nvertices.append(list(map(float, value.split(' '))))

def product_matrix(A,B):
    
    result = [[sum(a * b for a, b in zip(A_row, B_col))
                            for B_col in zip(*B)]
                                    for A_row in A]

    return result

def mydot(v1, v2):
     return sum([x*y for x,y in zip(v1, v2)])

def product_matrix_vector(G, v):
    return [mydot(r,v) for r in G]



class ObjetoRender(object):

    def glInit(self):
        self.color = color(50, 50, 50)
        self.clean_color = color(0, 0, 0)
        self.filename = 'Escena Final.bmp'
        self.pixels = [[]]
        self.zbuffer = [[]]
        self.light = VectorTridimensional(0, 0, 1)

        self.width = 0
        self.height = 0

        self.OffsetX = 0
        self.OffsetY = 0
        self.ImageHeight = 0
        self.ImageWidth = 0

        self.View = None

        self.active_normalMap = None

    def glClear(self):
        self.pixels = [
            [self.clean_color for x in range(self.width)]
            for y in range(self.height)
        ]

        self.zbuffer = [
            [-99999 for x in range(self.width)]
            for y in range(self.height)
        ]

    def glClearColor(self, r, g, b):
        self.clean_color = color(int(r * 255), int(g * 255), int(b * 255))
    def glColor(self, r, g, b):
        self.current_color = color(int(r * 255), int(g * 255), int(b * 255))
    def glCreateWindow(self, width, height):
        self.width = width
        self.height = height
        self.glClear()

    def glViewPort(self, x, y, width, height):

        self.OffsetX = int(x)
        self.OffsetY = int(y)

        self.ImageWidth = int(width)
        self.ImageHeight = int(height)

    def glVertex(self, x, y):

        x = int((x+1)*(self.ImageWidth/2)+self.OffsetX)
        y = int((y+1)*(self.ImageHeight/2)+self.OffsetY)

        self.pixels[y-1][x-1] = self.current_color

    def glFinish(self):
        f = open(self.filename, 'bw')

        f.write(char('B'))
        f.write(char('M'))
        f.write(dword(14 + 40 + self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(14 + 40))

        f.write(dword(40))
        f.write(dword(self.width))
        f.write(dword(self.height))
        f.write(word(1))
        f.write(word(24))
        f.write(dword(0))
        f.write(dword(self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))

        for x in range(self.height):
            for y in range(self.width):
                f.write(self.pixels[x][y])

    def glPoint(self, x, y):
        if not (-1 <= x <= 1) or not (-1 <= y <= 1):

            raise Exception('unexpected value')

        self.glVertex(x, y)

    def glLine(self, x0, y0, x1, y1):
        x0 = int((x0+1)*(self.ImageWidth/2)+self.OffsetX)
        y0 = int((y0+1)*(self.ImageHeight/2)+self.OffsetY)
        x1 = int((x1+1)*(self.ImageWidth/2)+self.OffsetX)
        y1 = int((y1+1)*(self.ImageHeight/2)+self.OffsetY)
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)

        steep = dy > dx

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

            dy = abs(y1 - y0)
            dx = abs(x1 - x0)

        offset = 0 * 2 * dx
        threshold = 0.5 * 2 * dx
        y = y0

        points = []
        bar = Bar('Cargando puntos: ', max=x1-x0)
        for x in range(x0, x1):
            if steep:
                points.append((y, x))
            else:
                points.append((x, y))

            offset += (dy/dx) * 2 * dx
            if offset >= threshold:
                y += 1 if y0 < y1 else -1
                threshold += 1 * 2 * dx
            bar.next()
        bar.finish()

        bar = Bar('Llenando puntos: ', max=len(points))
        for point in points:
            self.glPoint(((point[0]-self.OffsetX)*(2/self.ImageWidth)-1),
                         ((point[1]-self.OffsetY)*(2/self.ImageHeight)-1))
            bar.next()
        bar.finish()

    def transform(self, vertex):
        augmentedVertex = V4(vertex[0], vertex[1], vertex[2], 1)

        matrix1 = product_matrix_vector(self.model, augmentedVertex)
        matrix2 = product_matrix_vector(self.View,matrix1)
        matrix3 = product_matrix_vector(self.Projection,matrix2)
        transformedVertex = product_matrix_vector(self.Viewport,matrix3)

        transformedVertex = V4(*transformedVertex)

        return VectorTridimensional(
            transformedVertex.x / transformedVertex.w,
            transformedVertex.y / transformedVertex.w,
            transformedVertex.z / transformedVertex.w
        )

    def load(self, filename, translate=(0, 0, 0), scale=(1, 1, 1),rotate = (0, 0, 0), texture=None, normal = None):
        model = Obj(filename)
        self.loadMatrix(translate, scale, rotate)
        self.active_texture = texture
        self.active_normalMap = normal

        light = VectorTridimensional(0, 0, 1)

        bar = Bar('Cargando caras: ', max=len(model.faces))
        for face in model.faces:
            vcount = len(face)

            if vcount == 3:
                f1 = face[0][0] - 1
                f2 = face[1][0] - 1
                f3 = face[2][0] - 1

                a = self.transform(model.vertices[f1])
                b = self.transform(model.vertices[f2])
                c = self.transform(model.vertices[f3])

                normal = norm(cross(sub(b, a), sub(c, a)))
                intensity = dot(normal, light)

                if not texture:
                    grey = round(255 * intensity)
                    if grey < 0:
                        continue
                    self.triangle(a, b, c, color=color(grey, grey, grey))
                else:
                    t1 = face[0][1] - 1
                    t2 = face[1][1] - 1
                    t3 = face[2][1] - 1

                    tA = VectorTridimensional(*model.tvertices[t1])
                    tB = VectorTridimensional(*model.tvertices[t2])
                    tC = VectorTridimensional(*model.tvertices[t3])

                    tn1 = face[0][2] - 1
                    tn2 = face[1][2] - 1
                    tn3 = face[2][2] - 1
                    
                    tnA = VectorTridimensional(*model.nvertices[tn1])
                    tnB = VectorTridimensional(*model.nvertices[tn2])
                    tnC = VectorTridimensional(*model.nvertices[tn3])

                    self.triangle(a, b, c, texture=texture, texture_coords=(tA, tB, tC), texture_n= (tnA,tnB,tnC), intensity=intensity)
            else:
                f1 = face[0][0] - 1
                f2 = face[1][0] - 1
                f3 = face[2][0] - 1
                f4 = face[3][0] - 1

                vertices = [
                    self.transform(model.vertices[f1]),
                    self.transform(model.vertices[f2]),
                    self.transform(model.vertices[f3]),
                    self.transform(model.vertices[f4])
                ]

                normal = norm(cross(sub(vertices[0], vertices[1]), sub(
                    vertices[1], vertices[2])))  
                intensity = dot(normal, light)
                grey = round(255 * intensity)

                A, B, C, D = vertices
                
                if not texture:
                    grey = round(255 * intensity)
                    if grey < 0:
                        continue
                    self.triangle(A, B, C, color(grey, grey, grey))
                    self.triangle(A, C, D, color(grey, grey, grey))            
                else:
                    t1 = face[0][1] - 1
                    t2 = face[1][1] - 1
                    t3 = face[2][1] - 1
                    t4 = face[3][1] - 1
                    tA = VectorTridimensional(*model.tvertices[t1])
                    tB = VectorTridimensional(*model.tvertices[t2])
                    tC = VectorTridimensional(*model.tvertices[t3])
                    tD = VectorTridimensional(*model.tvertices[t4])

                    tn1 = face[0][2] - 1
                    tn2 = face[1][2] - 1
                    tn3 = face[2][2] - 1
                    tn4 = face[3][2] - 1
                    
                    tnA = VectorTridimensional(*model.nvertices[tn1])
                    tnB = VectorTridimensional(*model.nvertices[tn2])
                    tnC = VectorTridimensional(*model.nvertices[tn3])
                    tnD = VectorTridimensional(*model.nvertices[tn4])
                    
                    self.triangle(A, B, C, texture=texture, texture_coords=(tA, tB, tC), texture_n= (tnA,tnB,tnC), intensity=intensity)
                    self.triangle(A, C, D, texture=texture, texture_coords=(tA, tC, tD), texture_n= (tnA,tnC,tnD), intensity=intensity)
            bar.next()
        bar.finish()
    def triangle(self, A, B, C, color=None, texture=None, texture_coords=(),texture_n= (), intensity=1):
        xmin, xmax, ymin, ymax = baricentriBox(A, B, C)
        xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                P = V2(x, y)
                w, u, v = barycentric(A, B, C, P)

                if w < 0 or v < 0 or u < 0:
                    continue

                tA, tB, tC = texture_coords
                tnA, tnB, tnC = texture_n

                z = A.z * w + B.z * v + C.z * u

                tempx = int(((x/self.width)+1) *
                            (self.ImageWidth/2)+self.OffsetX)
                tempy = int(((y/self.height)+1) *
                            (self.ImageHeight/2)+self.OffsetY)


                if z > self.zbuffer[tempx][tempy]:

                    self.current_color = self.shader(
                        self, 
                        bar=(w,u,v),
                        light = self.light,
                        vertices = (A, B, C),
                        texture_coords = (tA,tB,tC), 
                        normals= (tnA,tnB,tnC),
                    )
                    
                    self.glVertex(x/self.width, y/self.height)
                    self.zbuffer[tempx][tempy] = z


    def showZbuffer(self):
        # Prints the pixels to the screen
        f = open('bmpresultado.bmp', 'bw')

        # File header (14 bytes)
        f.write(char('B'))
        f.write(char('M'))
        f.write(dword(14 + 40 + self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(14 + 40))

        # Image header (40 bytes)
        f.write(dword(40))
        f.write(dword(self.width))
        f.write(dword(self.height))
        f.write(word(1))
        f.write(word(24))
        f.write(dword(0))
        f.write(dword(self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))

        for y in range(0, self.height-1):
            for x in range(0, self.width):

                try:

                    toWrite = color(
                        self.zbuffer[x][y], self.zbuffer[x][y], self.zbuffer[x][y])

                except:
                    toWrite = color(0, 0, 0)

                f.write(toWrite)

        f.close()

    def loadMatrix(self, translate = (0, 0, 0), scale = (1, 1, 1), rotate = (0, 0, 0)):
        translate = VectorTridimensional(*translate)
        scale = VectorTridimensional(*scale)
        rotate = VectorTridimensional(*rotate)
        
        translateMatrix = [
            [1, 0, 0, translate.x],
            [0, 1, 0, translate.y],
            [0, 0, 1, translate.z],
            [0, 0, 0, 1]
        ]

        scaleMatrix = [
            [scale.x, 0, 0, 0],
            [0, scale.y, 0, 0],
            [0, 0, scale.z, 0],
            [0, 0, 0, 1]
        ]

        rotateXMatrix = [
            [1, 0, 0, 0],
            [0, cos(rotate.x), -sin(rotate.x), 0],
            [0, sin(rotate.x), cos(rotate.x), 0],
            [0, 0, 0, 1]
        ]

        rotateYMatrix = [
            [cos(rotate.y), 0, sin(rotate.y), 0],
            [0, 1, 0, 0],
            [-sin(rotate.y), 0, cos(rotate.y), 0],
            [0, 0, 0, 1]
        ]

        rotateZMatrix = [
            [cos(rotate.z), -sin(rotate.z), 0, 0],
            [sin(rotate.z), cos(rotate.z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]

        rotateMatrix = product_matrix(product_matrix(rotateYMatrix,rotateZMatrix),rotateZMatrix)
        self.model = product_matrix(product_matrix(translateMatrix,scaleMatrix),rotateMatrix)

    def lookAt(self, eye, center, up):
        z = norm(sub(eye, center))
        x = norm(cross(up, z))
        y = norm(cross(z, x))
        self.loadViewMatrix(x, y, z, center)
        self.loadProjectionMatrix(-1 / length(sub(eye, center)))
        self.loadViewportMatrix()


    def loadViewMatrix(self, x, y, z, center):
        Mi = [
            [x.x, x.y, x.z,  0],
            [y.x, y.y, y.z, 0],
            [z.x, z.y, z.z, 0],
            [0,     0,   0, 1]
        ]

        Op = [
            [1, 0, 0, -center.x],
            [0, 1, 0, -center.y],
            [0, 0, 1, -center.z],
            [0, 0, 0, 1]
        ]

        self.View = product_matrix(Mi,Op)


    def loadProjectionMatrix(self, coeff):
        self.Projection =  [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, coeff, 1]
        ]

    def loadViewportMatrix(self, x = 0, y = 0):
        self.Viewport =  [
            [self.width/2, 0, 0, x + self.width/2],
            [0, self.height/2, 0, y + self.height/2],
            [0, 0, 128, 128],
            [0, 0, 0, 1]
        ]

    def shader(self, render, **kwargs):
        w, u, v = kwargs['bar']
        L = kwargs['light']
        A, B, C = kwargs['vertices']
        tA, tB, tC = kwargs['texture_coords']
        nA, nB, nC = kwargs['normals']
        nx = nA.x * w + nB.x * u + nC.x * v
        ny = nA.y * w + nB.y * u + nC.y * v
        nz = nA.z * w + nB.z * u + nC.z * v
        normal = (nx, ny, nz)

        ta = (tA.x, tA.y)
        tb = (tB.x, tB.y)
        tc = (tC.x, tC.y)

        i = dot(norm(VectorTridimensional(nx,ny,nz)),L)

        tx = tA.x * w + tB.x * u + tC.x * v
        ty = tA.y * w + tB.y * u + tC.y * v

        r,g,b = 1,1,1

        if render.active_texture:
            texColor= render.active_texture.get_color_with_intensity(tx, ty, i)
            b *= texColor[0] / 255
            g *= texColor[1] / 255
            r *= texColor[2] / 255

        if render.active_normalMap:

            texNormal = render.active_normalMap.get_color(tx, ty)
            texNormal = [ (texNormal[2] / 255) * 2 - 1,
                        (texNormal[1] / 255) * 2 - 1,
                        (texNormal[0] / 255) * 2 - 1]

            texNormal = div(texNormal, frobeniusNorm(texNormal))

            # B - A
            edge1 = sub3(B[0], A[0], B[1], A[1], B[2], A[2])
            # C - A
            edge2 = sub3(C[0], A[0], C[1], A[1], C[2], A[2])
            # tb - ta 
            deltaUV1 = sub2(tb[0], ta[0], tb[1], ta[1])
            # tc - ta
            deltaUV2 = sub2(tc[0], ta[0], tc[1], ta[1])

            tangent = [0,0,0]
            f = 1 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1])
            tangent[0] = f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0])
            tangent[1] = f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1])
            tangent[2] = f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2])
            tangent = div(tangent, frobeniusNorm(tangent))
            tangent = div(tangent, frobeniusNorm(tangent))
            tangent = subVectors(tangent, multiply(dot2(tangent, normal[0], normal[1], normal[2]), normal))
            tangent = tangent / frobeniusNorm(tangent)

            bitangent = cross0(normal, tangent)
            bitangent = bitangent / frobeniusNorm(bitangent)


            tangentMatrix = [
                [tangent[0],bitangent[0],normal[0]],
                [tangent[1],bitangent[1],normal[1]],
                [tangent[2],bitangent[2],normal[2]]
            ]

            light = L 
            light = multiplyVM(light, tangentMatrix)
            light = div(light, frobeniusNorm(light))

            intensity = dot2(texNormal, light[0], light[1], light[2])

        else:
            intensity = dot2(normal, render.light[0], render.light[1], render.light[2])

        b *= intensity
        g *= intensity
        r *= intensity

        b *= 255
        g *= 255
        r *= 255

        if intensity > 0:
            return color(r, g, b)

        else:
            return color(0,0,0)

