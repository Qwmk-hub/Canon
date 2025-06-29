import torch
from torchvision import models, transforms
from PIL import Image
import os
from model.mobilenetv2 import get_mobilenet
from model.efficientnet import get_efficientnet_b1, get_efficientnet_b2
from torch.nn.functional import softmax
import time

# 설정
input_folder = "data/1cycle_image"  # 분류할 이미지 폴더
model_path = "outputs/best_1_model_b1.pth"
class_order = ["target_1", "target_2", "target_3", "target_4"]

# 1. 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = get_mobilenet(num_classes=5)
model = get_efficientnet_b1(num_classes=4)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# 2. 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3. 클래스 인덱스 → 이름 매핑 (ImageFolder에서 class_to_idx 기준)
class_names = ["target_1", "target_2", "target_3", "target_4"]

# 4. 이미지 순차적으로 예측
predicted_sequence = []
image_files = sorted(os.listdir(input_folder))

total_inference_time = 0

for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    start_time = time.time()

    with torch.no_grad():
        output = model(input_tensor)
        probs = softmax(output, dim=1)
        max_prob = probs.max().item()
        pred_idx = probs.argmax(1).item()

        end_time = time.time()  # ✅ 끝 시간 기록
        inference_time = end_time - start_time
        total_inference_time += inference_time  # ✅ 누적

        if max_prob >= 0.450 and pred_idx < len(class_names): # best_1,2 : 0.45 / best_3 : 0.93
            pred_label = class_names[pred_idx]
        else:
            pred_label = "pass"
        predicted_sequence.append(pred_label)

# 예측된 시퀀스 출력
print("🔍 예측된 시퀀스:")
for img_name, label in zip(image_files, predicted_sequence):
    print(f"{img_name}: {label}")

# 5. 순차적인 target 확인 (정확한 순서 유지 필요)
target_idx = 0
for label in predicted_sequence:
    if label == class_order[target_idx]:
        target_idx += 1
        if target_idx == len(class_order):
            break
    elif label in class_order[target_idx+1:]:
        # 순서 어긋난 경우 → 바로 FAIL
        target_idx = -1
        break

# 6. 결과 출력
if target_idx == len(class_order):
    print("✅ PASS: 모든 타겟이 순차적으로 감지됨!")
else:
    print("❌ FAIL: 순서대로 감지되지 않음.")

avg_time = total_inference_time / len(image_files)
print(f"\n⏱️ 전체 Inference 시간: {total_inference_time:.4f}초")
print(f"⏱️ 이미지당 평균 Inference 시간: {avg_time:.4f}초")
