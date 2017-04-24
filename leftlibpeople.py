import matplotlib.pyplot as plt
import itertools
import rotanimate
from annotation3D import annotate3D
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict

# setting up figure
plt.ion()
fig = plt.figure(dpi=100)
ax = fig.add_subplot(111, projection='3d')
# fig_2 = plt.figure(dpi=100)
# ax_2 = fig_2.add_subplot(111)

# setting up statistical tools from the sklearn package.
# This page explains the base of the technique and variants:
# http://scikit-learn.org/stable/modules/decomposition.html
pca_3 = PCA(n_components=3)
# pca_2 = PCA(n_components=2)

# left-lib hitleys:
hitleys = [[.487, 1-.737, .754, 1-.766, 'alysson'],
           [.462, .359, .714, .34, 'pupo'],
           [.494, 1-.833, .824, 1-.821, 'leite'],
           [.596, 1-.731, .795, 1-.83, 'valdez'],
           [.635, .314, .783, 1-.83, 'castanheira'],
           [.417, .442, .638, .446, 'souza'],
           [.34, .333, .791, .28, 'ramos'],
           [.891, 1-.859, .84, 1-.876, 'goncalo'],
           [.583, .314, .658, .222, 'stivelman'],
           [.314, .385, .843, .331, 'horsth'],
           [.788, .212, .815, 1-.867, 'santullo'],
           [1-.731, .314, .86, 1-.706, 'torres'],
           [.615, 1-.724, .797, 1-.766, 'ferreira'],
           [.462, 1-.705, .658, .448, 'almeida'],
           [.744, .218, .745, 1-.835, 'vasconcelos'],
           [.385, .276, .825, .267, 'veras'],
           [.462, 1-.718, .869, 1-.747, 'borgo'],
           [.705, 1-.808, .878, 1-.903, 'silva'],
           [.429, 1-.763, .853, 1-.747, 'menezes'],
           [.346, .314, .761, .326, 'porto'],
           [1-.756, 1-.769, .852, .345, 'castilho'],
           [.346, .365, .746, .34, 'giroto'],
           [.256, .244, .843, .368, 'azevedo'],
           [.846, 1-.795, .889, 1-.963, 'cesar'],
           [1-.905, 1-.902, .907, 1-.926, 'lucas'],
           [1-.833, 1-.756, .771, .377, 'azeredo'],
           [.5, .25, .836, 1-.733, 'marques'],
           [.449, .295, .814, .257, 'bezerra'],
           [1-.795, 1-.846, .932, 1-.706, 'gerent'],
           [.808, .327, .693, 1-.722, 'rodrigues'],
           [.872, 1-.756, .861, 1-.908, 'dias'],
           [.385, .231, .841, .299, 'albano'],
           [.647, 1-.712, .801, 1-.761, 'baltar'],
           [.571, 1-.724, .87, 1-.816, 'magliano'],
           [.353, .378, .804, .303, 'bobadilha'],
           [.494, .34, .728, .234, 'mascarenhas'],
           [.615, .333, .635, .225, 'claudino'],
           [.615, .269, .663, .236, 'maiandi'],
           [.481, 1-.801, .685, .308, 'pimentel'],
           [1-.763, .487, .568, .517, 'herrmann'],
           [.353, 1-.801, .818, 1-.775, 'corsino'],
           [.256, .231, .908, .257, 'sousa']]

hitleys_d = OrderedDict((x[4], x[:4]) for x in hitleys)

references = [[.5, .5, .5, .5, 'perfect_center']]

references_d = OrderedDict((x[4], x[:4]) for x in references)


# actual work: first find the fit using the pca_x component and than use it to 
# transform our 4-dimension data.
hitleys_3 = pca_3.fit(list(hitleys_d.values())).transform(list(hitleys_d.values()))
references_3 = pca_3.transform(list(references_d.values()))
# hitleys_2 = pca_2.fit(list(hitleys_d.values())).transform(list(hitleys_d.values()))

# Just unwrapping in a data display friendly way.
X_3, Y_3, Z_3 = zip(*hitleys_3)
RX_3, RY_3, RZ_3 = zip(*references_3)
# X_2, Y_2 = zip(*hitleys_2)

# plotting the scatter plots.
ax.scatter3D(X_3, Y_3, Z_3, c='b')
ax.scatter3D(RX_3, RY_3, RZ_3, c='r')
# ax_2.scatter(X_2, Y_2)

for label, xyz_ in zip([x[4] for x in itertools.chain(hitleys, references)], 
                       itertools.chain(hitleys_3, references_3)):
    annotate3D(ax, s=label, xyz=xyz_, fontsize=10, xytext=(-3,3),
               textcoords='offset points', ha='right', va='bottom')

rotanimate.rotanimate(ax, 100, 'hitleys.gif', delay=20)