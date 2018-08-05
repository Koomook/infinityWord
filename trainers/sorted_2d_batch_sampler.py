from collections import defaultdict
import random
from os.path import dirname, abspath, join, exists
from os import makedirs
import json
from pymongo import MongoClient
import pickle

BASE_DIR = dirname(dirname(abspath(__file__)))
MONGODB_CONFIG = json.load(open(join(BASE_DIR, 'datasets', 'mongodb_config.json')))
CLIENT = MongoClient(host=MONGODB_CONFIG['host'],
                     username=MONGODB_CONFIG['username'],
                     password=MONGODB_CONFIG['password'],
                     authSource=MONGODB_CONFIG['authSource'],
                     authMechanism=MONGODB_CONFIG['authMechanism'])
DB = CLIENT.get_database(MONGODB_CONFIG['database'])


class FixedSizeClustering:

    def __init__(self, datapoints, cluster_size, drop_last):
        self.datapoints = datapoints  # list of (int, int)
        self.cluster_size = cluster_size  # int
        self.drop_last = drop_last

        self.start_num_grids = len(datapoints)

    def find_clusters(self):

        max_x, min_x, max_y, min_y = self.find_max_min(self.datapoints)

        num_grids = self.start_num_grids
        datapoints_to_cluster = [(index, x, y) for index, (x,y) in enumerate(self.datapoints)]

        grid_width = (max_x - min_x) / num_grids
        grid_height = (max_y - min_y) / num_grids

        clusters = []

        while True:
            grid = defaultdict(lambda: defaultdict(list))
            for index, x, y in datapoints_to_cluster:

                grid_i = int((x - min_x) // grid_width)
                grid_j = int((y - min_y) // grid_height)
                grid_elements = grid[grid_i][grid_j]
                grid_elements.append((index, x, y))

                # if grid is full
                if len(grid_elements) == self.cluster_size:
                    indexes_in_grid = [index for index, x, y in grid[grid_i][grid_j]]
                    clusters.append(indexes_in_grid)
                    grid[grid_i][grid_j] = []  # empty grid

            remaining_datapoints = []
            for grid_i in grid:
                for grid_j in grid[grid_i]:
                    remaining_datapoints.extend(grid[grid_i][grid_j])

            num_grids = num_grids // 2 + 1 if num_grids > 2 else num_grids // 2
            if num_grids == 0:
                break

            grid_width = (max_x - min_x) / num_grids
            grid_height = (max_y - min_y) / num_grids

            datapoints_to_cluster = remaining_datapoints

        if not self.drop_last:
            remaining_indexes = [index for index, x, y in remaining_datapoints]
            clusters.append(remaining_indexes)

        return clusters

    @staticmethod
    def find_max_min(datapoints):
        max_x = max(datapoints, key=lambda p: p[0])[0] + 0.01
        min_x = min(datapoints, key=lambda p: p[0])[0] - 0.01
        max_y = max(datapoints, key=lambda p: p[1])[1] + 0.01
        min_y = min(datapoints, key=lambda p: p[1])[1] - 0.01
        return max_x, min_x, max_y, min_y


class Sorted2DBatchSamplerOnTheFly:

    def __init__(self, dataset, batch_size, drop_last=False, max_length=100):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        concat_source = lambda source: sum(source, [])
        datapoints = [(len(concat_source(source)), len(target))
                      for source, target in self.dataset
                      if len(concat_source(source)) + len(target) < max_length] # filter out too long sentences
        if len(datapoints) != len(self.dataset):
            print('Filtered out {n} sentences'.format(n=len(self.dataset) - len(datapoints)))

        clustering = FixedSizeClustering(datapoints=datapoints,
                                         cluster_size=batch_size,
                                         drop_last=drop_last)
        self.clusters = clustering.find_clusters()

        random.seed(0)
        random.shuffle(self.clusters)

    def __iter__(self):
        for cluster in self.clusters:
            yield cluster

    def __len__(self):
        return len(self.clusters)


class Sorted2DBatchSampler:

    def __init__(self, phase, version='v1'):

        self.phase = phase
        parameters_filepath = join(BASE_DIR, 'parameters', 'sorted_2d_batch_sampler', version, phase + '.pkl')
        with open(parameters_filepath, 'rb') as file:
            self.clusters = pickle.load(file)

    def __iter__(self):
        for cluster in self.clusters:
            yield cluster

    def __len__(self):
        return len(self.clusters)

    @staticmethod
    def concat_source(source):
        return sum(source, [])  # can be changed later

    @staticmethod
    def prepare_batch_sampler(phase, batch_size, drop_last=False, max_length=100, version='v1'):

        collection = DB.get_collection('novels_sources_targets')
        cursor = collection.find({'phase': phase})

        datapoints = []
        for document in cursor:
            source_indexed = document['indexed']['source']
            target_indexed = document['indexed']['target']
            source_concated = Sorted2DBatchSampler.concat_source(source_indexed)

            source_length = len(source_concated)
            target_length = len(target_indexed)
            if source_length + target_length > max_length:
                print('A sentence filtered out from dataset')
                continue   # filter out too long sentences

            datapoints.append((source_length, target_length))

        clustering = FixedSizeClustering(datapoints=datapoints,
                                         cluster_size=batch_size,
                                         drop_last=drop_last)
        clusters = clustering.find_clusters()
        random.seed(0)
        random.shuffle(clusters)

        parameters_dir = join(BASE_DIR, 'parameters', 'sorted_2d_batch_sampler', version)
        if not exists(parameters_dir):
            makedirs(parameters_dir)
        parameters_filepath = join(parameters_dir, phase + '.pkl')
        with open(parameters_filepath, 'wb') as file:
            pickle.dump(clusters, file)


if __name__ == '__main__':

    datapoints = [(1.0, 2.0),
                  (1.0, 3.0),
                  (2.0, 2.7),
                  (2.4, 1.7),
                  (2.7, 4.5),
                  (3.0, 1.0),
                  (4.1, 2.2),
                  (4.7, 2.6),
                  (4.8, 1.1),
                  (4.9, 2.3)]

    clustering = FixedSizeClustering(datapoints, cluster_size=3, drop_last=False)
    print(clustering.find_clusters())

    Sorted2DBatchSampler.prepare_batch_sampler('train', batch_size=32, drop_last=False, max_length=100, version='v1')
    Sorted2DBatchSampler.prepare_batch_sampler('val', batch_size=100, drop_last=False, max_length=100, version='v1')
    Sorted2DBatchSampler.prepare_batch_sampler('test', batch_size=32, drop_last=False, max_length=100, version='v1')
    batch_sampler = Sorted2DBatchSampler('train')
    print(next(iter(batch_sampler)))
    # import pandas as pd
    # import numpy as np
    # from matplotlib import pyplot as plt
    # % matplotlib inline
    #
    # datapoints = np.random.randint(100, size=(100, 2))
    #
    # clustering = FixedSizeClustering(datapoints, cluster_size=3, drop_last=True)
    # print(clustering.find_clusters())
    #
    # df = pd.DataFrame(datapoints, columns=['x', 'y'])
    #
    # for i, seq in enumerate(clustering.find_clusters()):
    #     for e in seq:
    #         df.loc[e, 'cluster'] = i
    #
    # df.plot(kind='scatter', x='x', y='y', c='cluster', figsize=(17, 10), colorbar=False, colormap='cool', s=50)