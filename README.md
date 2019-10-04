# Palm_NonLinear

Source Code for IEEE BTAS 2019 paper:
Palmprint Recognition Using Realistic Animation Aided Data Augmentation.

Dependencies:
1. Pytorch
2. TensorboardX
3. Numpy
4. Scipy
5. PIL
6. OpenCV

The CPD code in the project is from https://github.com/siavashk/pycpd


# Steps to train:
1. Set the data path, the anim_frames path for the dataset in config file.
2. Run cpd_transform.py script from the root folder of the project to generate the augmented dataset.
3. Change the TRAIN_IMG_DIR in config file to cpd_data/CASIA/flip/ROI/train/ if training with CASIA augmented data for example.
4. Change the network to desired model: example inceptionv3 or alexnet
5. Change the OUTPUT_PATH appropriately to save the model being trained.
6. Run main.py script from the root folder to train the model.

# Config settings:
1. Set STANDARD_AUGMENT to True and TRANSLATION_ONLY to False for full affine data augmentation.
2. Set TRANSLATION_ONLY to True when using non-linearly augmented training data to combine them (NltTrans).
3. Please refer to the paper to set other parameters.
