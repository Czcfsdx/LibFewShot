# based on https://github.com/WendyBaiYunwei/FSL-QuickBoost
import random
import pickle
import json
import time
import os
import numpy as np
# import torch
# from numpy.linalg import norm
from torch import from_numpy
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor

class FSL_FORSET():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
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

        # with open(config.data_dir + 'name2idx_test.json', 'r') as f:
        #     self.name2idx_test = json.load(f)

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
        with open(file, 'rb') as f:
            self.classifier = pickle.load(f)
        print(f"QuickBoost FSL-FOREST model loaded form {file}")

    def test(self):
        print("QuickBoost start testing")
        exit()
    
    def _generate_data_form_encoder(self):
        train_x = []
        train_y = []

        print("Generating QuickBoost RF data from pretrained encoder ...")
        random.seed(self.seed)
        with open(self.pretrained_encoder_path, 'rb') as f:
            self.embeddings = pickle.load(f)
        with open(self.pretrained_encoder_test_path, 'rb') as f:
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
                cosine_sims = []
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
        relations = []
        spt_embs = [] # support embeddings
        classpt_embs = []
        for i, support_name in enumerate(support_names):
            if i % shot_size == 0 and i > 0:
                avg_spt_emb = np.mean(spt_embs, axis = 0)
                avg_spt_emb = avg_spt_emb / np.linalg.norm(avg_spt_emb)
                classpt_embs.append(avg_spt_emb)
                spt_embs.clear()
            tokens = support_name.split('/')
            support_name = tokens[-1]
            spt_embedding = self.embeddings_test[self.name2idx_test[support_name]]
            spt_embedding = spt_embedding / np.linalg.norm(spt_embedding)
            spt_embs.append(spt_embedding)
        avg_spt_emb = np.mean(spt_embs, axis = 0)
        avg_spt_emb = avg_spt_emb / np.linalg.norm(avg_spt_emb)
        classpt_embs.append(avg_spt_emb)
        spt_embs.clear()

        for qry_name in qry_names:
            tokens = qry_name.split('/')
            qry_name = tokens[-1]
            qry_emb = self.embeddings_test[self.name2idx_test[qry_name]]
            qry_emb = qry_emb / np.linalg.norm(qry_emb)
            for classI in range(5):
                diff = (classpt_embs[classI] - qry_emb) ** 2
                relations_rf.append(diff)  
        relations_rf = np.stack(relations_rf).reshape(-1, 512)
        preds = self.classifier.predict(relations_rf)
        preds = preds.reshape(len(qry_names), -1)
        return from_numpy(preds)

def quickboost(**kwargs):
    """Constructs a FSL-FOREST model."""
    model = FSL_FORSET(**kwargs)
    return model
