import numpy as np
import pydot
from confusion_matrix import ConfMatrix

if __name__ == '__main__':
    
    from optparse import OptionParser
    op = OptionParser(__doc__)
    
    op.add_option("-r", default='overlap_raw.npz.npy', dest="raw", 
                  help="overlap raw numpy file]")
    op.add_option("-n", default='overlap_norm.npz.npy', dest="overlap", 
                  help="overlap norm numpy file]")
    op.add_option("-o", default='overlap_matrix.m', dest="matrix", 
                  help="ConfMatrix]")
    op.add_option("-i", default='',dest="ignore_idx", help='ignore idx list') 
    op.add_option("-f", default='overlap2onto.png', dest="fname", help='saving fname') 
    op.add_option("-m", default=100.0,type=int, dest="min_overlap", help='min overlap') 
    op.add_option("--filter", default=False, action ='store_true', dest="filter", 
                  help='filter same node with different names') 
    
    opts, args = op.parse_args(sys.argv)
        
    fname = opts.fname
    raw = np.load(opts.raw)
    overlap = np.load(opts.overlap)
    matrix = ConfMatrix.load_matrix(opts.matrix)

    ignore_idx = [[int(el) for el in opts.ignore_idx.split(',')]
    xsize = raw.diagonal()
    nx,ny = raw.shape
        
    graph = pydot.Dot(graph_type='digraph')
    nodes = []
    # create the nodes
    for label in labels:
        node = pydot.Node(label)
        nodes.append(node)
        graph.add_node(node)
        
    # filter diff name same thing
    if True:
        fname  = fname.replace('.png', '_cleaned.png')
        print "remove same node with different names"
        for x in range(nx):
            for y in range(ny):
                if x!=y and (overlap[x,y]==opts.min_overlap) and xsize[x]==xsize[y]:
                    print "keeping %i->%s, removing %i->%s" %(x, labels[x], y, labels[y])
                    ignore_idx.append(y)
        print "remaining nodes: %i->%i" %(nx, nx-len(ignore_idx))

    # nodes to check
    idx = [i for i in range(nx) if i not in ignore_idx]

    # connect the nodes
    for x in idx:
        print "gen. connections to %s (%i/%i)" %(labels[x], x+1, nx)
        for y in idx:
            if x!=y:
                if overlap[x,y]>=opts.min_overlap:
                    xy_overlap = '%2.0f%%' %overlap[x,y]
                    graph.add_edge(pydot.Edge(nodes[x], nodes[y], 
                                              label=str(int(xsize[x]))))

    # and we are done
    graph.write_png(fname)
