# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import sys 
sys.path.append("C:\/Users\/Ryan\/Desktop\/workspace\/AI_Train\/code")
from detection import models


class CustomRetinaNetConfig(models.RetinaNetConfig):
    def __init__(self):
        super().__init__()

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="data",
            root="airdoor",
            ann_file="train_arch.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="data",
            root="airdoor",
            ann_file="test_arch.json",
            remove_images_without_annotations=False,
        )
        self.num_classes = 1
        
        # ------------------------ training cfg ---------------------- #
        self.train_image_short_size = 800
        self.max_epoch = 10
        self.lr_decay_stages = [24, 32]
        self.nr_images_epoch = 1000
        self.warm_iters = 100
        self.log_interval = 10



Net = models.RetinaNet
Cfg = CustomRetinaNetConfig
