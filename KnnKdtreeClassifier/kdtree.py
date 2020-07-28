import numpy as np


class Node(object):
    def __init__(self, data):
        self.data = data
        self.hyperplane_feature = None
        self.median = None
        self.left_subtree_root = None
        self.right_subtree_root = None
        self.parent = None
        self.size = len(data)

    def is_not_leaf(self):
        return self.left_subtree_root is not None and self.right_subtree_root is not None


class Kdtree(object):
    def __init__(self, x, y, leaf_size=1, weights='uniform'):
        '''Храню в X множество образцов x, их метки y, индексы i для функции kneighbors, расстояния d (пока нули) для функции kneighbors'''
        self.n_queries = x.shape[0]
        self.n_features = x.shape[1]

        zeros = np.zeros((self.n_queries, 3))
        self.X = np.append(x, zeros, axis=1)
        self.X[:, -3] = y

        for i in range(self.n_queries):
            self.X[i][-2] = i
        
        '''В каждой вершине храню объединение data его поддеревьев для удобства реализации обратного хода'''
        self.root = Node(self.X)
        self.leaf_size = leaf_size
        self.weights = weights
        self.build(self.root)

    def sort_by_feature(self, arr, feature_index):
        '''Вспомогательная функция сортировки np.array по feature_index компоненте'''
        return arr[np.argsort(arr[:, feature_index], kind="mergesort")]

    def build(self, root, depth=0):
        '''
        Рeкурсивное построение дерева. На вход: новый корень и глубина для определния hyperplane_feature
        Храню родителей каждой вершины для удобства реализации обратного хода (у корня parent =  None)
        '''
        n = root.size
        if n > self.leaf_size:
            root.hyperplane_feature = depth % self.n_features

            root.data = self.sort_by_feature(root.data, root.hyperplane_feature)
            median = n // 2

            root.median = root.data[median]
            left_subtree_root = Node(root.data[:median])
            right_subtree_root = Node(root.data[median:])
            left_subtree_root.parent = root
            right_subtree_root.parent = root
            root.left_subtree_root = left_subtree_root
            root.right_subtree_root = right_subtree_root

            self.build(left_subtree_root, depth + 1)
            self.build(right_subtree_root, depth + 1)
        else:
            return

    def distance(self, x, y):
        '''L1-метрика'''
        distance = 0
        for i in range(self.n_features):
            distance += np.fabs(x[i] - y[i])
        return distance

    def closest_leaf(self, pivot):
        '''Здесь находим ближайший лист'''
        current_node = self.root
        while current_node.is_not_leaf():
            index = current_node.hyperplane_feature
            if pivot[index] <= current_node.median[index]:
                current_node = current_node.left_subtree_root
            else:
                current_node = current_node.right_subtree_root
        return current_node

    def closest_k_neighbors(self, pivot, node, k_neighbors):
        '''Основная функция поиска k соседей'''
        if node.size < k_neighbors:
            return self.closest_k_neighbors(pivot, node.parent, k_neighbors)

        data = node.data
        for i in data:
            distance = self.distance(pivot, i)
            i[-1] = distance

        data = self.sort_by_feature(data, -1)
        if node.parent is None:
            return data[:k_neighbors]
        
        '''Проверка условия обратного хода в дереве'''
        distance_to_furthest_neighbor = data[k_neighbors - 1][-1]

        hyperplane = list(pivot)
        hyperplane[node.parent.hyperplane_feature] = node.parent.median[node.parent.hyperplane_feature]
        distance_to_hyperplane = self.distance(pivot, hyperplane)

        if distance_to_hyperplane < distance_to_furthest_neighbor:
            return self.closest_k_neighbors(pivot, node.parent, k_neighbors)
        else:
            return data[:k_neighbors]
    
    '''Далее идут реализация голосования и функция классификации точки с помощью него'''
    def get_weights(self, distances):
        if self.weights == 'uniform':
            return np.ones(distances.shape)
        elif self.weights == 'distance':
            result = np.zeros(distances.shape[0])
            for i in range(distances.shape[0]):
                result[i] = 1 / distances[i]
            return result
        elif callable(self.weights):
            return self.weights(distances)

    def voting(self, kn):
        weights_of_objects = self.get_weights(kn[..., -1])
        weights_of_classes = dict()
        for i in range(kn.shape[0]):
            if weights_of_classes.get(kn[i][-3]) is not None:
                weights_of_classes[kn[i][-3]] += weights_of_objects[i]
            else:
                weights_of_classes[kn[i][-3]] = weights_of_objects[i]

        max_key = max(weights_of_classes, key=weights_of_classes.get)
        return max_key, weights_of_classes

    def classify_point(self, pivot, n_neighbours):
        leaf = self.closest_leaf(pivot)
        kn = self.closest_k_neighbors(pivot, leaf, n_neighbours)
        max_key, weights_of_classes = self.voting(kn)
        return max_key
