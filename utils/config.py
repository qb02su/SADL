class Config:
    def __init__(self,task):
        # Default npy directory names - modify these within specific task blocks if needed
        # Training data can come from two sources
        

        if "la" in task: # SSL
            self.base_dir = './Datasets/LASeg/'
            self.save_dir = './LA_data'
            self.patch_size = (112, 112, 80)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50
            self.batch_size = 4

        elif "synapse" in task: # IBSSL
            self.base_dir = './Datasets/Synapse'
            self.save_dir = './Synapse_data'
            self.patch_size = (64, 128, 128)
            self.num_cls = 14
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
            self.batch_size = 2

        elif "mmwhs" in task:
            self.base_dir = './Datasets/MM-WHS'
            self.save_dir = './MMWHS_data'
            self.patch_size = (128, 128, 128)
            self.num_cls = 5
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
            self.batch_size = 1

        elif "udamms" in task:
            self.base_dir = './Datasets/MM-WHS'
            self.save_dir = './MNMS_data_2d'
            self.patch_size = (32, 128, 128)
            self.num_cls = 4
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 80
            self.batch_size = 1

        elif "mms2d" in task:
            self.base_dir = './Datasets/mnms_split_data/Labeled'
            self.save_dir = './Mms'
            self.patch_size = (224, 224)
            self.num_cls = 4
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50
            self.batch_size = 16


        elif "mnms" in task: # SemiDG
            if "2d" in task:
                self.base_dir = './Datasets/mnms_split_data/Labeled'
                self.save_dir = './MNMS_data_2d'
                self.patch_size = (32, 128, 128)
                self.num_cls = 4
                self.num_channels = 1
                self.n_filters = 32
                self.early_stop_patience = 50
                self.batch_size = 32
            else:
                self.base_dir = './Datasets/MNMs/Labeled'
                self.save_dir = './MNMS_data'
                self.patch_size = (32, 128, 128)
                self.num_cls = 4
                self.num_channels = 1
                self.n_filters = 32
                self.early_stop_patience = 80
                self.batch_size = 4
        elif "brat" in task:
            self.base_dir = './Datasets/BraTS_2020'
            self.save_dir = './BraTS_2020'
            self.patch_size = (144, 144, 144)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
            self.batch_size = 1

        elif "feta" in task:
            self.base_dir = './Datasets/feta'
            self.save_dir = './feta_data'
            self.patch_size = (128, 128, 128)
            self.num_cls = 8
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
            self.batch_size = 2

        elif "ixi" in task:
            self.train_npy_dir_1 = './Datasets/ixi/npy/train'
            self.train_npy_dir_2 = './Datasets/tk/npy'
            self.val_npy_dir = './Datasets/ixi/npy/val'
            self.test_npy_dir = './Datasets/ixi/npy/test'
            self.patch_size = (128, 128, 128)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
            self.batch_size = 1


        else:
            raise NameError("Please provide correct task name, see config.py")
