# based on https://github.com/WendyBaiYunwei/FSL-QuickBoost
import random
import pickle
import json
import time
import os
import numpy as np
import torch
# from numpy.linalg import norm
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from core.utils import accuracy

class FSL_FORSET():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            # print(f"key: {key}, value: {value}")
            setattr(self, key, value)
        # TODO: Read fellow kwargs from config
        self.CLASS_SIZE = 64 # miniImagenet has 64 classes for training
        self.sample_size = 50
        self.iterations = 2
        self.IMGS_PER_CLASS = 600
        self.EMBEDDING_SIZE = 512 # resnet 18 has output embedding size of 512

    def train(self):
        train_x, train_y = self._generate_data_form_encoder()
        train_x = train_x.astype(np.float16)
        train_y = train_y.astype(np.float16)

        print('start RF training')
        start = time.time()
        self.classifier = RandomForestRegressor(**self.rf_kwargs)
        self.classifier.fit(train_x, train_y)
        end = time.time()
        print(f"done RF training. time taken is {end - start:.4f} s")

    def save_model(self, path):
        file = os.path.join(path, 'model_best.pkl')
        with open(file, 'wb') as f:
            pickle.dump(self.classifier, f)
        print(f"QuickBoost FSL-FOREST model saved in {file}")

    def load_model(self, path):
        file = os.path.join(path, 'model_best.pkl')
        print(f"Loading FSL-FOREST model form {file} ...")
        with open(file, 'rb') as f:
            self.classifier = pickle.load(f)

        print("Loading pretrain embedding and name2idx.json ...")
        with open(self.pretrain_embedding_test_path, 'rb') as f:
            self.embeddings_test = pickle.load(f)
            self.embeddings_test = self.embeddings_test.astype(np.float16)
        with open(self.pretrain_name2idx_test_path, 'r') as f:
                self.name2idx_test = json.load(f)

    def test(self, batch, model_outputs = None):
        image, global_target, image_names = batch
        image = image.to(self.device)

        support_images, query_images, support_targets, query_targets, support_names, query_names = self.split_by_episode(image, image_names, mode = 2)

        episode_size = query_targets.size(0)
        output_list = []
        # TODO: testing ensemble with other model
        if model_outputs is not None:
            model_output_chunks = torch.chunk(model_outputs, chunks = episode_size, dim = 0)

        for i in range(episode_size):
            forest_output = self._get_batch_rels(support_names[i], query_names[i], self.test_shot).to(self.device).reshape(-1, self.test_way)
            final_output = self._minmax_normalize(forest_output, dim = 1)

            if model_outputs is not None:
                model_output = self._minmax_normalize(model_output_chunks[i], dim = 1)

                final_output = torch.stack([model_output, final_output], dim=2)
                final_output = torch.mean(final_output, dim=2)

            output_list.append(final_output)
            
        output = torch.cat(output_list, dim = 0)
        acc = accuracy(output, query_targets.reshape(-1))
        return output, acc

    def _generate_data_form_encoder(self):
        train_x = []
        train_y = []

        print("Generating QuickBoost RF data from pretrain embedding ...")
        random.seed(self.seed)
        with open(self.pretrain_embedding_path, 'rb') as f:
            self.embeddings = pickle.load(f)
        with open(self.pretrain_embedding_test_path, 'rb') as f:
            self.embeddings_test = pickle.load(f)
            self.embeddings_test = self.embeddings_test.astype(np.float16)
        train_x, train_y = self._generate_data(0, self.CLASS_SIZE, self.sample_size, self.iterations, self.IMGS_PER_CLASS, self.EMBEDDING_SIZE)
        
        print(f"QuickBoost RF data successfully generated, data shape: X-{train_x.shape}, Y-{train_y.shape}")
        return train_x, train_y

    def _generate_data(self, class_start, class_end, sample_size, iterations, imgs_per_class, embedding_size):
        x_list = []
        y_list = []
        full_indices = set([i for i in range(class_start * imgs_per_class, class_end * imgs_per_class)])
        for class_idx in range(class_start, class_end):
            same_class_idxs = set([i for i in range(class_idx * imgs_per_class, (class_idx + 1) * imgs_per_class)])
            diff_class_idxs = full_indices - same_class_idxs
            for i in range(iterations):
                embedding_idx = class_idx * imgs_per_class + i
                current_image = self.embeddings[embedding_idx]
                current_image = current_image / np.linalg.norm(current_image) 
                cur = np.expand_dims(current_image, axis = 0)
                cur = np.repeat(cur, sample_size, 0)
                
                # get same class images' indices
                same_class_imgs = random.sample(same_class_idxs, k = sample_size)
                
                same_class_embs = []
                for img_i in same_class_imgs:
                    same_class_emb = self.embeddings[img_i]
                    same_class_emb = same_class_emb / np.linalg.norm(same_class_emb) 
                    same_class_embs.append(same_class_emb)
                same_class_imgs = np.stack(same_class_embs).squeeze()
                diff = (cur - same_class_imgs) ** 2
                x_list.append(diff)
                # label same class image pairs as '1'
                labels = np.repeat(np.ones(1), sample_size, 0)
                y_list.append(labels)

                # get different class images' indices
                diff_class_idxs = random.sample(diff_class_idxs, k = sample_size)
                diff_img_embs = self.embeddings[diff_class_idxs].squeeze()
                for diff_i in range(len(diff_img_embs)):
                    diff_img_embs[diff_i] =\
                        diff_img_embs[diff_i] / np.linalg.norm(diff_img_embs[diff_i]) 
                diff = (cur - diff_img_embs) ** 2 
                x_list.append(diff)
                # label same class image pairs as '0'
                labels = np.repeat(np.zeros(1), sample_size, 0)
                y_list.append(labels)

        x_list = np.stack(x_list).reshape(-1, embedding_size)
        y_list = np.stack(y_list).reshape(-1, 1)

        return x_list, y_list
    
    def _get_batch_rels(self, support_names, qry_names, shot_size = 1):
        relations_rf = []
        spt_embs = [] # support embeddings
        classpt_embs = []
        for i, support_name in enumerate(support_names):
            if i % shot_size == 0 and i > 0:
                avg_spt_emb = np.mean(spt_embs, axis = 0)
                avg_spt_emb = avg_spt_emb / np.linalg.norm(avg_spt_emb)
                classpt_embs.append(avg_spt_emb)
                spt_embs.clear()
            spt_embedding = self.embeddings_test[self.name2idx_test[support_name]]
            spt_embedding = spt_embedding / np.linalg.norm(spt_embedding)
            spt_embs.append(spt_embedding)
        avg_spt_emb = np.mean(spt_embs, axis = 0)
        avg_spt_emb = avg_spt_emb / np.linalg.norm(avg_spt_emb)
        classpt_embs.append(avg_spt_emb)
        spt_embs.clear()

        for qry_name in qry_names:
            qry_emb = self.embeddings_test[self.name2idx_test[qry_name]]
            qry_emb = qry_emb / np.linalg.norm(qry_emb)
            for classI in range(self.test_way):
                diff = (classpt_embs[classI] - qry_emb) ** 2
                relations_rf.append(diff)  
        relations_rf = np.stack(relations_rf).reshape(-1, self.EMBEDDING_SIZE)
        preds = self.classifier.predict(relations_rf)
        preds = preds.reshape(len(qry_names), -1)
        return torch.from_numpy(preds)

    def split_by_episode(self, images, names, mode = 1):
        # TODO: other models don't need to manual set this, try to find why
        if mode == 2: # for test
            shot_num = self.test_shot
            way_num = self.test_way
            query_num = self.test_query
        else:
            shot_num = self.shot_num
            way_num = self.way_num
            query_num = self.query_num

        episode_size = images.size(0) // (
            way_num * (shot_num + query_num)
        )

        local_targets = (
            torch.arange(way_num, dtype=torch.long)
            .view(1, -1, 1)
            .repeat(episode_size, 1, shot_num + query_num)
            .view(-1)
        )
        local_labels = (
            local_targets
            .to(self.device)
            .contiguous()
            .view(episode_size, way_num, shot_num + query_num)
        )

        _, c, h, w = images.shape
        images = images.to(self.device).contiguous().view(
            episode_size,
            way_num,
            shot_num + query_num,
            c,
            h,
            w,
        )
        support_images = images[:, :, : shot_num, :, ...].contiguous().view(
            episode_size,
            way_num * shot_num,
            c,
            h,
            w,
        )
        query_images = images[:, :, shot_num :, :, ...].contiguous().view(
            episode_size,
            way_num * query_num,
            c,
            h,
            w,
        )
        support_target = local_labels[:, :, : shot_num].reshape(
            episode_size, way_num * shot_num
        )
        query_target = local_labels[:, :, shot_num :].reshape(
            episode_size, way_num * query_num
        )

        names = np.array(names).reshape(episode_size, way_num, shot_num + query_num)
        support_names = names[:, :, : shot_num].reshape(episode_size, way_num * shot_num)
        query_names = names[:, :, shot_num :].reshape(episode_size, way_num * query_num)

        return support_images, query_images, support_target, query_target, support_names, query_names

    def _minmax_normalize(self, x, dim, eps=1e-8):
        """Min-Max 归一化函数"""
        min_val = x.min(dim=dim, keepdim=True)[0]
        max_val = x.max(dim=dim, keepdim=True)[0]
        return (x - min_val) / (max_val - min_val + eps)

def quickboost(**kwargs):
    """Constructs a FSL-FOREST model."""
    model = FSL_FORSET(**kwargs)
    return model
