import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
import numpy as np

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)


def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)

def show_graph_with_labels(networks, node_position, node_id):
    rows, cols = np.where(networks == 1)
    edges = list(zip(rows.tolist(), cols.tolist()))
    segments = [(node_position[s].tolist(), node_position[t].tolist()) for s, t in edges]
    xyzn = np.concatenate((node_position[node_id - 1], (node_id - 1)[:, np.newaxis]), axis=1).astype(np.int)

    fig = plt.figure(dpi=150)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.gca(projection='3d')
    ax.set_axis_off()
    ax.set_facecolor('xkcd:black')
    # plot vertices
    ax.scatter(xyzn[:, 0], xyzn[:, 1], xyzn[:, 2], c=node_id, cmap="jet_r", marker='o', s=20)
    # plot edges
    edge_col = Line3DCollection(segments, lw=0.2) # edgecolors='white'
    ax.add_collection3d(edge_col)
    # add vertices annotation.
    for i in range(xyzn.shape[0]):
        xyz_pos = [xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]]
        annotation = xyzn[i, 3]
        annotate3D(ax, s=str(''), xyz=xyz_pos, fontsize=0, xytext=(-3, 3),
                   textcoords='offset points', ha='right', va='bottom')
        # annotate3D(ax, xyz=xyz_pos,ha='right', va='bottom')
    plt.show()