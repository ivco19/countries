

class node:
    def __init__(self, nnode, value):
        self.id = nnode
        self.value = value
        self.outgoing = {}
        self.incoming = {}

    def __str__(self):
        # para que funcione el print
        string = str(self.id) + ' node, outgoing: ' + \
                 str([x.id for x in self.outgoing]) + \
                 ' incoming: ' + str([x.id for x in self.incoming])
        return string

    def add_neighbor(self, neighbor, names, values):
        self.outgoing[neighbor] = {}
        for n, v in zip(names, values):
            self.outgoing[neighbor][n] = v
        #print(neighbor.id, '---->', self)
        #print(self.id)
 
    def be_neighbor(self, neighbor, names, values):
        self.incoming[neighbor] = {}
        for n, v in zip(names, values):
            self.incoming[neighbor][n] = v
        #print(neighbor.id, '- - >', self)

    def get_connections(self):
        return self.outgoing.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.outgoing[neighbor]



class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    # node functions --------------------------------------
    def add_node(self, nnode, value):
        self.num_vertices = self.num_vertices + 1
        new_node = node(nnode, value)
        self.vert_dict[nnode] = new_node
        return new_node

    def get_node(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def get_node_value(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n].value
        else:
            return None

    def set_node(self, n, value):
        if n in self.vert_dict:
            v = self.get_node(n)
            v.value = value
        else:
            return None

    def get_vertices(self):
        return self.vert_dict.keys()

    def get_nodes_to(self, nnode):
        v = self.get_node(nnode)
        c = []
        for i in v.incoming:
            c.append(i.id)
        return(c)

    def get_nodes_from(self, nnode):
        v = self.get_node(nnode)
        c = []
        for i in v.outgoing:
            c.append(i.id)
        return(c)


    # edge functions --------------------------------------
    def add_edge(self, frm, to, names = [], values = 0):
        """
        warning: does not verify if edge already exists
        """
        mis1 = frm not in self.vert_dict
        mis2 = to not in self.vert_dict
        if mis1 or mis2:
            print('Not a node called ')
            print(frm)
            print(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], names, values)

        self.vert_dict[to].be_neighbor(self.vert_dict[frm], names, values)

    def get_edge(self, frm, to, field):
        if frm not in self.vert_dict:
            self.add_node(frm)
        if to not in self.vert_dict:
            self.add_node(to)

        v_frm = self.get_node(frm)
        v_to  = self.get_node(to)

        ws = v_frm.get_weight(v_to)
        value = ws[field]
        return(value)

 
    # graph functions --------------------------------------
    def show(self):
        for v in self:
            print('NODE %s:  %f' % (v.id, v.value))
            for w in v.get_connections():
                vid = v.get_id()
                wid = w.get_id()
                print ('             %s -> %s %s' % (vid, wid, v.get_weight(w)))

    # computation functions --------------------------------
    def node_activation(self, nnode, key):
        l = self.get_nodes_to(nnode)
        x = []
        for v in l:
            a = self.get_node(v)
            x.append( self.get_edge(a.id, nnode, key) )
        return(x)







#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

if __name__ == '__main__':

    g = Graph()

    for i, inode in enumerate(['A','B','C','D']):
        print(i)
        g.add_node(inode, 0)

    nms = ['x', 'y']
    g.add_edge('A', 'B', nms, [1, 100])
    g.add_edge('A', 'C', nms, [2, 200])
    g.add_edge('B', 'D', nms, [3, 300])
    g.add_edge('D', 'B', nms, [4, 400])
    g.add_edge('D', 'C', nms, [5, 500])
    g.add_edge('C', 'C', nms, [6, 600])

    g.add_edge('B', 'D', nms, [333, 333])

    g.show()

#    # activation of B:
#    x, y = g.node_activation('B')
#
#    print(x)
#    print(y)


#    # activation of all nodes
#    for v in g:
#        x, y = g.node_activation(v.id)
#
#
#        XXX = g.get_value(v.id) + 
#
#        g.set_value(v.id, XXX)
#
#



# test
#    g = Graph()
#    g.add_node('I', 1)
#    g.add_node('C', 0)
#    g.add_node('R', 0)
#
#    nms = ['prob', 'delay']
#
#    vals = [0.23, 45]
#    g.add_edge('I', 'C', nms, vals)  
#    vals = [0.04, 11]
#    g.add_edge('I', 'R', nms, vals)  
#    vals = [0.97, 6]
#    g.add_edge('C', 'I', nms, vals)  
#
#    g.show()
#
#    ws = g.get_edge('I', 'C', 'prob')
#    print(ws)
#    ws = g.get_edge('C', 'I', 'prob')
#    print(ws)
 
