import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import pycls.datasets.utils as ds_utils
from tqdm import tqdm

class DCoM:
    """
    AL algorithm that selects the next centroids based on point density and model confidence, which measured
    using minimum margin.
    """

    def __init__(self, cfg, lSet, uSet, budgetSize, lSet_deltas, max_delta):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.all_features = ds_utils.load_features(self.ds_name, self.seed)
        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.max_delta = max_delta
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)  # indices of lSet and then uSet in all_features
        self.rel_features = self.all_features[self.relevant_indices]  # features of lSet and then uSet
        self.lSet_deltas = [np.float(delta) for delta in lSet_deltas]
        self.lSet_deltas_dict = dict(zip(np.arange(len(self.lSet)), self.lSet_deltas))
        self.delta_avg = np.average(self.lSet_deltas)

    def construct_graph_excluding_lSet(self, delta=None, batch_size=700):
        """
        Creates a directed graph where:
        x -> y if l2(x, y) < delta.
        Considers all images, but does not reference or delete edges in lSet.

        The graph is represented by a list of edges (a sparse matrix) and stored in a DataFrame.
        """
        if delta is None:
            delta = self.delta_avg

        xs, ys, ds = [], [], []
        print(f'Start constructing graph using delta={delta}')
        # distance computations are done in GPU
        cuda_feats = torch.tensor(self.rel_features).cuda()
        for i in range(len(self.rel_features) // batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(cur_feats, cuda_feats)
            mask = dist < delta
            # saving edges using indices list - saves memory.
            x, y = mask.nonzero().T
            xs.append(x.cpu() + batch_size * i)
            ys.append(y.cpu())
            ds.append(dist[mask].cpu())

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()

        df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
        print(f'Before delete lSet neighbors: Graph contains {len(df)} edges.')
        return df

    def construct_graph(self, delta=None, batch_size=700):
        """
         Creates a directed graph where:
         x -> y if l2(x, y) < delta, and deletes the covered points using lSet_deltas.

         Deletes edges to the covered samples (samples that are covered by lSet balls)
         and deletes all the edges from lSet.

         The graph is represented by a list of edges (a sparse matrix) and stored in a DataFrame.
         """
        if delta is None:
            delta = self.delta_avg

        df = self.construct_graph_excluding_lSet(delta, batch_size)

        # removing incoming edges to all cover from the existing labeled set
        edges_from_lSet_in_ball = np.isin(df.x, np.arange(len(self.lSet))) & (df.d < df.x.map(self.lSet_deltas_dict))
        covered_samples = df.y[edges_from_lSet_in_ball].unique()

        edges_to_covered_samples = np.isin(df.y, covered_samples)
        all_edges_from_lSet = np.isin(df.x, np.arange(len(self.lSet)))

        mask = all_edges_from_lSet | edges_to_covered_samples  # all the points inside the balls
        df_filtered = df[~mask]

        print(f'Finished constructing graph using delta={delta}')
        print(f'Graph contains {len(df_filtered)} edges.')
        return df_filtered, covered_samples

    def select_samples(self, model, train_data, data_obj):
        """
        Performs DCoM active sampling section.
        """
        def get_competence_score(coverage):
            """
            Implementation of the logistic function weighting.
            """
            k = self.cfg.ACTIVE_LEARNING.K_LOGISTIC  # the logistic growth rate or steepness of the curve
            a = self.cfg.ACTIVE_LEARNING.A_LOGISTIC  # the logistic function center
            p = (1 + np.exp(-k * (1 - a)))
            competence_score = p / (1 + np.exp(-k * (coverage - a)))
            print(f'a = {a}, k = {k}, p = {round(p, 4)}')
            print("The coverage over the graph is: ", coverage)
            print("The competence_score is: ", round(competence_score, 3))
            return round(competence_score, 3)

        print(f"\n==================== Start DCoM Active Sampling ====================")
        # Calculate the current coverage
        selected = []
        fully_graph = self.construct_graph_excluding_lSet(self.max_delta)
        current_coverage = DCoM.calculate_coverage(fully_graph, self.lSet, self.lSet_deltas_dict, len(self.relevant_indices))
        del fully_graph

        competence_score = get_competence_score(current_coverage)

        margin = self.calculate_margin(model, train_data, data_obj)
        margin[0: len(self.lSet)] = 0 # We define the margin score to be 1-margin (as described in our paper)

        cur_df, covered_samples = self.construct_graph(self.delta_avg)

        for i in range(self.budgetSize): # The active selection
            coverage = len(covered_samples) / len(self.relevant_indices)

            if len(cur_df) == 0:
                ranks = np.zeros(len(self.relevant_indices))
            else:
                # calculate density for each point
                ranks = self.calculate_density(cur_df)

            cur_selection = DCoM.normalize_and_maximize(ranks, margin, 1, lambda r, e: competence_score * e + (1 - competence_score) * r)[0]
            print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tCoverage is {coverage:.3f}. \tCurr choice is {cur_selection}. \tcompetence_score={competence_score}')

            new_covered_samples = cur_df.y[(cur_df.x == cur_selection)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'

            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]  # Delete all the edges to the covered samples
            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            margin[cur_selection] = 0
            selected.append(cur_selection)

        activeSet = self.relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        assert len(np.intersect1d(self.lSet, activeSet)) == 0, 'all samples should be new'
        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        return activeSet, remainSet

    def new_centroids_deltas(self, lSet_labels, pseudo_labels, budget, batch_size=500, all_labels=[]):
        """
        Performs binary search of the next delta values.
        """
        def calc_threshold(coverage):
            assert 0 <= coverage <= 1, f'coverage is not between 0 to 1: {coverage}'
            return 0.2 * coverage + 0.4

        def check_purity(df, cent_label, delta):
            # find all the neighbors
            edges_from_lSet = (df.x == cent_idx) & (df.d < delta)
            neighbors_idx = df.y[edges_from_lSet]

            # take their neighbors and compute the ball purity (Are the labels the same as the chosen point?)
            neighbors_pseudo_labels = list(map(pseudo_labels.__getitem__, neighbors_idx))
            neighbors_real_labels = list(map(all_labels.__getitem__, neighbors_idx))

            if len(neighbors_idx):
                pseudo_purity = sum(np.array(neighbors_pseudo_labels == cent_label)) / len(neighbors_idx)
                real_purity = sum(np.array(neighbors_real_labels == cent_label)) / len(neighbors_idx)
                print(f'real_purity: {real_purity}, pseudo_purity: {pseudo_purity}')
                return pseudo_purity
            return 0

        new_deltas = []
        df = self.construct_graph_excluding_lSet(self.max_delta, batch_size)

        fully_df = self.construct_graph_excluding_lSet(self.max_delta)
        covered_samples = fully_df.y[np.isin(fully_df.x, np.arange(len(self.lSet))) & (
                fully_df.d < fully_df.x.map(self.lSet_deltas_dict))].unique()
        coverage = len(covered_samples) / len(self.relevant_indices)

        purity_threshold = calc_threshold(coverage)
        print("Current threshold: ", purity_threshold)

        for cent_idx, centroid in enumerate(self.lSet):
            if cent_idx < len(self.lSet) - budget:  # Not new points
                continue

            print(f'start calculation for cent_idx: {cent_idx}')
            low_del_val = 0
            max_del_val = self.max_delta
            mid_del_val = (low_del_val + max_del_val) / 2
            last_purity = 0
            last_delta = mid_del_val

            while abs(low_del_val - max_del_val) > self.cfg.ACTIVE_LEARNING.DELTA_RESOLUTION:
                curr_purity = check_purity(df, cent_label=lSet_labels[cent_idx], delta=mid_del_val)
                print("centroid: ", centroid, ", idx: ", cent_idx, ". delta = ", mid_del_val, " and purity = ", curr_purity)

                if last_delta < mid_del_val and last_purity == purity_threshold and curr_purity < purity_threshold:
                    mid_del_val = last_delta
                    break

                if curr_purity < purity_threshold:
                    # if smaller than the threshold - try smaller delta
                    max_del_val = mid_del_val
                elif curr_purity >= purity_threshold:  # if bigger than threshold -> try bigger delta
                    low_del_val = mid_del_val

                last_purity = curr_purity
                last_delta = mid_del_val
                mid_del_val = (low_del_val + max_del_val) / 2

            curr_purity = check_purity(df, cent_label=lSet_labels[cent_idx], delta=mid_del_val)
            print("the chosen delta: ", mid_del_val, "and its purity: ", curr_purity)
            print("---------------------------------------------------------------------------")
            new_deltas.append(str(round(mid_del_val, 2)))

        self.lSet_deltas = [np.float(delta) for delta in new_deltas]
        self.lSet_deltas_dict = dict(zip(np.arange(len(self.lSet)), self.lSet_deltas))
        print("All new deltas: ", new_deltas, '\n')
        return new_deltas

    def calculate_density(self, df):
        rank_mapping = pd.DataFrame(df.groupby('x')['y'].count())
        all_indices_df = pd.DataFrame(index=np.arange(len(self.relevant_indices)))
        result_df = pd.merge(all_indices_df, rank_mapping, left_index=True, right_index=True, how='left').fillna(0)
        return np.array(result_df)

    def calculate_margin(self, model, train_data, data_obj):
        oldmode = model.training
        model.eval()

        print(f'Start calculating points margin.')
        all_images_idx = self.relevant_indices
        images_loader = data_obj.getSequentialDataLoader(indexes=all_images_idx,
                                                batch_size=self.cfg.TRAIN.BATCH_SIZE, data=train_data)
        clf = model.cuda()
        ranks = []
        n_loader = len(images_loader)

        for i, (x_u, _) in enumerate(tqdm(images_loader, desc="All images Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)
                temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                temp_u_rank, _ = torch.sort(temp_u_rank, descending=True)
                difference = temp_u_rank[:, 0] - temp_u_rank[:, 1]

                # for code consistency across uncertainty, entropy methods i.e., picking datapoints with max value
                difference = -1 * difference
                ranks.append(difference.detach().cpu().numpy())
        ranks = np.concatenate(ranks, axis=0)
        print(f"u_ranks.shape: {ranks.shape}")

        model.train(oldmode)

        margin_result = np.array(ranks).reshape(-1, 1)
        scaler = MinMaxScaler()
        normalized_margin_result = scaler.fit_transform(margin_result)
        final_margin_result = np.array(normalized_margin_result.flatten().tolist())
        return final_margin_result

    @staticmethod
    def normalize_and_maximize(param1_list, param2_list, amount, target_func):
        """
        Perform pre-processing on each list and apply the target function on them.
        """
        param1_arr = np.array(param1_list).reshape(-1, 1)
        param2_arr = np.array(param2_list).reshape(-1, 1)

        # Min-Max normalization using scikit-learn's MinMaxScaler
        scaler = MinMaxScaler()
        param1_normalized = scaler.fit_transform(param1_arr)
        param2_normalized = scaler.fit_transform(param2_arr)

        # Calculate the product using the provided target_func
        product_array = target_func(param1_normalized.flatten(), param2_normalized.flatten())

        sorted_indices = np.argsort(product_array)[::-1]

        return sorted_indices[:amount]

    @staticmethod
    def calculate_coverage(fully_df, lSet, lSet_deltas_dict, total_data_len):
        """
        Return the current probability coverage.
        """
        covered_samples = fully_df.y[np.isin(fully_df.x, np.arange(len(lSet))) & (
                fully_df.d < fully_df.x.map(lSet_deltas_dict))].unique()  # lSet send arrow to them
        return len(covered_samples) / total_data_len
