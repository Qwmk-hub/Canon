import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import random

input_root = '/home/icl/WORKSPACE/Canon/data/test'
output_root = '/home/icl/WORKSPACE/Canon/data/feature_aug'

os.makedirs(output_root, exist_ok=True)

# 단일 변환 함수들
def rotate(img):  
    angle = random.randint(-5, 5)  # Randomized angle between -10 and +10
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

def scale(img):
    fx = random.uniform(0.85, 1.2)  # Randomized fx between 0.85 and 1.2
    fy = random.uniform(0.85, 1.2)  # Randomized fy between 0.85 and 1.2
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

def affine(img):
    rows, cols = img.shape[:2]
    pts1 = np.float32([[10, 10], [100, 10], [10, 100]])
    pts2 = np.float32([[10 + random.uniform(-1, 1), 20 + random.uniform(-1, 1)],
                        [110 + random.uniform(-1, 1), 20 + random.uniform(-1, 1)],
                        [20 + random.uniform(-1, 1), 100 + random.uniform(-1, 1)]])  # Randomized points
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (cols, rows), borderValue=(0, 0, 0))

def gaussian_blur(img):  
    ksize = random.choice([(3, 3), (5, 5), (7, 7)])  # Randomized kernel size
    return cv2.GaussianBlur(img, ksize, 0)

def change_brightness(img):
    img = img.astype(np.float32)  # Convert to float32
    value = random.randint(-60, 60)  # Random value between -60 and +60
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.uint8)  # Convert back to uint8

def motion_blur(img):
    size = random.randint(5, 30)  # Random size
    direction = random.choice(['horizontal', 'vertical'])  # Randomly apply horizontal or vertical blur
    kernel = np.zeros((size, size))
    if direction == 'horizontal':
        kernel[int((size - 1) / 2), :] = np.ones(size)
    else:
        kernel[:, int((size - 1) / 2)] = np.ones(size)
    kernel /= size
    return cv2.filter2D(img, -1, kernel)

def hazy_effect(img):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    return cv2.addWeighted(img, 0.7, blur, 0.5, 0)

def sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

# 사용할 augmentation 함수 리스트 (flip 제외!)
aug_funcs = [
    rotate,
    scale,
    affine,
    change_brightness,
    motion_blur,
    hazy_effect,  # Added hazy_effect
    sharpen,      # Added sharpen
    gaussian_blur  # Re-enabled gaussian_blur
]

# 사용자 설정: 각 이미지 당 증강 생성 수
num_augmented_2 = 1  # 2개 변환으로 생성할 이미지 수
num_augmented_3 = 1  # 3개 변환으로 생성할 이미지 수
# num_augmented_all = 1  # 모든 변환으로 생성할 이미지 수

for class_folder in os.listdir(input_root):
    class_input_path = os.path.join(input_root, class_folder)
    class_output_path = os.path.join(output_root, class_folder)
    os.makedirs(class_output_path, exist_ok=True)

    img_paths = glob(os.path.join(class_input_path, "*.jpg"))

    for img_path in tqdm(img_paths, desc=f"Augmenting {class_folder}"):
        img = cv2.imread(img_path)
        basename = os.path.splitext(os.path.basename(img_path))[0]

        # # 원본 저장
        # cv2.imwrite(os.path.join(class_output_path, f"{basename}.jpg"), img)

        # 2개 변환으로 생성
        for i in range(num_augmented_2):
            aug_img = img.copy()
            ops = random.sample(aug_funcs, k=2)  # 랜덤 2개 조합
            for op in ops:
                aug_img = op(aug_img)

            save_name = f"{basename}_aug2_{i}.jpg"
            cv2.imwrite(os.path.join(class_output_path, save_name), aug_img)

        # 3개 변환으로 생성
        for i in range(num_augmented_3):
            aug_img = img.copy()
            ops = random.sample(aug_funcs, k=3)  # 랜덤 3개 조합
            for op in ops:
                aug_img = op(aug_img)

            save_name = f"{basename}_aug3_{i}.jpg"
            cv2.imwrite(os.path.join(class_output_path, save_name), aug_img)

        # # 모든 변환으로 생성
        # aug_img = img.copy()
        # ops = random.sample(aug_funcs, k=len(aug_funcs))  # 모든 변환 랜덤 순서
        # for op in ops:
        #     aug_img = op(aug_img)

        save_name = f"{basename}_aug_all.jpg"
        cv2.imwrite(os.path.join(class_output_path, save_name), aug_img)
