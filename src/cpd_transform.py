import yaml
import glob
from functools import partial
import matplotlib.pyplot as plt
from pycpd import deformable_registration
import numpy as np
from datetime import datetime
import argparse
import os
import cv2
from PIL import Image, ImageFilter
from scipy.ndimage import filters as scF


class CPDTransform:

    def __init__(self):
        self.draw_cpd = False
        self.debug_plot = False
        self.show_edges = False
        self.show_original = False
        self.show_anim_frame = False
        self.apply_mask = False
        self.max_iterations = 100000
        self.filter_size = 11
        self.amplitude = 3.0
        self.frames =['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']
        self.deformations_anim = None

    def get_deformations(self, input_img):
        original_pos = None
        deformations_anim = []
        for frame in self.frames:
            fImg = Image.open(os.path.join(cfg['ANIM_FRAMES_PATH'], frame))
            # Resize it to the input frame size
            fImg = fImg.resize(input_img.size, Image.ANTIALIAS)
            if self.show_anim_frame:
                fImg.show()
            (fxImg, fEdgePos) = self.detect_edges(fImg, self.filter_size)

            if original_pos is None:
                original_pos = fEdgePos

            if self.show_edges:
                mImg = Image.blend(fImg.convert("RGBA"), fxImg.convert("RGBA"), alpha=0.5)
                mImg.show()

            X = fEdgePos.copy()
            Y = original_pos.copy()

            M01 = X[::4]
            N01 = Y[::4]
            reg2 = deformable_registration(alpha=2.0, beta=2.0, **{'X': M01.copy(), 'Y': N01.copy()},
                                           max_iterations=self.max_iterations, tolerance=1e-8)
            if self.draw_cpd:
                fig = plt.figure()
                fig.add_axes([0, 0, 1, 1])
                ax = fig.axes[0]
            else:
                ax = None
            callback = partial(self.callback_func, ax=ax)
            reg2.register(callback)
            M02 = N01
            N02 = reg2.transform_point_cloud(N01)
            deformations_anim.append([M02.astype(int), N02.astype(int)])
            # break
        return  deformations_anim

    def gen(self, cfg, image_path, out_path):
        input_img = Image.open(image_path)
        if self.deformations_anim is None:
            self.deformations_anim = self.get_deformations(input_img)

        warp_input_img = input_img
        basedir = os.path.dirname(image_path)
        fname = image_path.replace(basedir, "")
        fname = fname.split(".")
        i = 0
        fp = out_path + "/" + fname[0] + "_" + str(0) + '.jpg'
        #warp_input_img.save(fp=fp)
        i = i + 1
        for deformation in self.deformations_anim:
            X = deformation[0]
            Y = deformation[1]
            Y = X + ((Y - X) * self.amplitude)
            roi_tps = self.warpImage(Y, X, warp_input_img)
            fp = out_path + "/" + fname[0] + "_" + str(i) + '.jpg'
            print(fp)
            roi_tps.save(fp=fp)
            i = i + 1
            if self.debug_plot:
                plt.figure()
                #plt.imshow(roi_tps, origin='upper')
                plt.scatter(x=X[:, 0], y=X[:, 1], color='red', label='Target', alpha=0.5)
                plt.scatter(x=Y[:, 0], y=Y[:, 1], color='blue', label='Source', alpha=0.5)
                plt.ylim([0, input_img.size[1]])
                plt.xlim([0, input_img.size[0]])
                plt.show()

    def detect_edges(self, img, c):
        img = np.array(img.convert("L")).astype('uint8')
        img = scF.median_filter(img, size=c)
        img = scF.sobel(img)
        imx = Image.fromarray(img)
        imx = imx.filter(ImageFilter.SMOOTH_MORE)
        img = imx.filter(ImageFilter.FIND_EDGES)
        imz = np.array(img)
        imz[imz < 245] = 0
        imz[imz >= 245] = 255
        iEdgePos = np.argwhere(imz > 0)
        return Image.fromarray(imz), iEdgePos

    def callback_func(self, iteration, error, X, Y, ax):
        if self.draw_cpd:
            plt.cla()
            ax.scatter(Y[:, 1], Y[:, 0], color='blue', label='Source', alpha=0.5)
            ax.scatter(X[:, 1], X[:, 0], color='red', label='Target', alpha=0.5)
            plt.text(0.87, 0.92,
                     'Iteration: {:d}\nError: {:06.6f}'.format(iteration, error),
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax.transAxes,
                     fontsize='x-large')
            ax.legend(loc='upper left', fontsize='x-large')
            plt.draw()
            plt.pause(0.001)

    def warpImage(self, source, target, img):
        open_cv_image = np.array(img)
        tps = cv2.createThinPlateSplineShapeTransformer(regularizationParameter=2.0)
        source = source.reshape(-1, len(source), 2)
        target = target.reshape(-1, len(target), 2)

        if(source.shape != target.shape):
            raise Exception("Not a valid transform...")

        matches = list()
        for i in range(0, len(source[0])):
            matches.append(cv2.DMatch(i, i, 0))

        tps.estimateTransformation(target, source, matches)
        new_img = tps.warpImage(open_cv_image)
        return Image.fromarray(new_img)

    def get_mask(self, img):
        img = np.array(img.convert("L")).astype('uint8')
        img[img > 0.0] = 255
        return Image.fromarray(img)


def make_out_dir(cfg, o_path):
    if o_path is not None:
        out_path = os.path.join(o_path, "cpd_out", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        out_path = os.path.join(cfg['CPD_OUT'])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


def call_cpd(cfg, i_path, out_path ):
    print("Processing file: ", i_path)
    cpdTransform = CPDTransform()
    cpdTransform.gen(cfg, i_path, out_path)
    del cpdTransform

def getbasefilename(path):
    kpname = os.path.basename(path)
    fname = kpname.split(".")
    fname = fname[0].split("_")
    return fname[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group()
    group.add_argument("--image", "-i", help="Path to input image")
    group.add_argument("--out", "-o", help="Path to output images")
    opts = parser.parse_args()
    i_path = opts.image
    o_path = opts.out

    with open('config/cpd.yaml', 'r') as f_in:
        cfg = yaml.load(f_in)
        f_path = cfg["CPD_IN"]
    print(cfg)

    if i_path is not None:
        out_path = make_out_dir(cfg, o_path)
        call_cpd(cfg, i_path, out_path)
    elif f_path is not None:
        print("Processing folder: ", f_path)
        out_path = make_out_dir(cfg, None)
        processed_file_paths = glob.glob(out_path + "/*")
        processed_files = set()
        for file_path in processed_file_paths:
            file = getbasefilename(file_path)
            processed_files.add(file)
            #print(file)

        input_files = glob.glob(f_path + "/*")
        for i_path in input_files:
            file = getbasefilename(i_path)
            if file in processed_files:
                print("Ignoring: ", file)
                continue
            call_cpd(cfg, i_path, out_path)
