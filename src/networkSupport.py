import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


def createGraphNetwork(adjMatrix):
    # Create undirected nodes from Numpy Array
    G = nx.from_numpy_array(adjMatrix, create_using=nx.Graph())

    # We give a position and plot
    if True:
        pos = nx.spring_layout(G)  # pos = nx.nx_agraph.graphviz_layout(G)
        nx.draw_networkx(G, pos)
        plt.show()

    # Create Connected Components
    # The length of connected tells us how many trees are
    numConnectCompoPerTree = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]

    numSubTrees = len(numConnectCompoPerTree)
    print("The number of Image Sets is: ", str(numSubTrees))

    # Este es falso si hay algún punto que no está conectado y por lo tanto
    # hay 2 o más
    resultado = nx.is_connected(G)
    #print("La red no está conectada totalmente?: ", resultado, "hay varios trees")

    # Get list of nodos
    nodes = list(G.nodes())

    # sabemos cual es la cantidad de subtrees
    # vamos a encontrarlos iterando en todos los nodos
    subTreesIdx = np.empty((numSubTrees,1),dtype=object)
    tmpContador = 0
    for idx in range(len(nodes)):
        node = nodes[idx]
        if tmpContador < numSubTrees:
            # Regresa lista de nodos conectados
            listaNodesConectados = nx.node_connected_component(G,node)
            # Check if we already have this stored
            boolCheck = np.zeros((numSubTrees,1))
            for i in range(subTreesIdx.shape[0]):
                resBool = subTreesIdx[i,0] == listaNodesConectados
                if resBool:
                    boolCheck[i,0] = 1
            # IF there's at least a single 1 means the subtree
            # already accounted for
            # If all zeros, we make it the current value
            if np.all((boolCheck == 0)):
                subTreesIdx[tmpContador, 0] = listaNodesConectados
                tmpContador = tmpContador + 1


    # Vamos por los nodos
    nodos = []
    coneccionNodos = []
    concomps = np.array(nodes)

    subTrees = np.empty((len(subTreesIdx),1),dtype=object)
    # asignaremos numeros diferentes por nodos para generar con comps
    # y obtendremos los subTrees
    connector = 1
    for i in range(subTreesIdx.shape[0]):
        concomps[list(subTreesIdx[i,0])] = connector
        connector = connector + 1
        subTrees[i,0] = adjMatrix[np.ix_(list(subTreesIdx[i,0]),list(subTreesIdx[i,0]))]



    return subTreesIdx, subTrees, concomps