import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="FGAI Emotion Analysis",
    page_icon="🎭",
    layout="wide"
)

# =========================
# 감정 클래스
# =========================
CLASS_NAMES = ['anger', 'happy', 'panic', 'sadness']

# =========================
# 디바이스 설정
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# 모델 로드
# =========================
@st.cache_resource
def load_model():

    model = timm.create_model(
        'convnext_base',
        pretrained=False,
        num_classes=4
    )

    model_path = 'models/convnext.pth'

    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )

    model.to(device)
    model.eval()

    return model

model = load_model()

# =========================
# 이미지 전처리
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# 감정 예측 함수
# =========================
def predict_emotion(image):

    image = image.convert('RGB')

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)

    probs = probs.cpu().numpy()[0]

    pred_idx = np.argmax(probs)

    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]

    return pred_class, confidence, probs

# =========================
# 영상 분석 함수
# =========================
def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    emotions = []
    times = []
    confidences = []

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    frame_interval = int(fps)

    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(rgb)

            pred_class, confidence, probs = predict_emotion(pil_img)

            current_time = frame_count / fps

            emotions.append(pred_class)
            confidences.append(confidence)
            times.append(current_time)

        frame_count += 1

    cap.release()

    return times, emotions, confidences

# =========================
# CSS 스타일
# =========================
st.markdown("""
<style>

.main {
    background-color: #F7F9FC;
}

h1 {
    color: #1E3A8A;
}

h2 {
    color: #2563EB;
}

.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# 제목
# =========================
st.title("🎭 FGAI 시스템")
st.subheader("FGI 감정 흐름 분석 AI 플랫폼")

st.markdown("---")

# =========================
# 사이드바
# =========================
st.sidebar.title("📂 메뉴")

menu = st.sidebar.radio(
    "기능 선택",
    [
        "이미지 감정 분석",
        "영상 감정 흐름 분석"
    ]
)

# =========================
# 이미지 감정 분석
# =========================
if menu == "이미지 감정 분석":

    st.header("🖼️ 이미지 감정 분석")

    uploaded_file = st.file_uploader(
        "이미지를 업로드하세요",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption='업로드 이미지',
                use_container_width=True
            )

        pred_class, confidence, probs = predict_emotion(image)

        with col2:

            st.success(f"예측 감정: {pred_class}")
            st.info(f"신뢰도: {confidence:.2%}")

            st.subheader("감정 확률")

            prob_df = pd.DataFrame({
                'Emotion': CLASS_NAMES,
                'Probability': probs
            })

            st.bar_chart(
                prob_df.set_index('Emotion')
            )

# =========================
# 영상 감정 흐름 분석
# =========================
if menu == "영상 감정 흐름 분석":

    st.header("🎬 영상 감정 흐름 분석")

    uploaded_video = st.file_uploader(
        "영상을 업로드하세요",
        type=['mp4', 'mov', 'avi']
    )

    if uploaded_video is not None:

        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())

        st.video(temp_video.name)

        with st.spinner("영상 분석 중입니다..."):

            times, emotions, confidences = analyze_video(
                temp_video.name
            )

        st.success("영상 분석 완료")

        # =========================
        # 결과 테이블
        # =========================
        st.subheader("📊 감정 분석 결과")

        result_df = pd.DataFrame({
            'Time': times,
            'Emotion': emotions,
            'Confidence': confidences
        })

        st.dataframe(result_df)

        # =========================
        # Emotion Timeline
        # =========================
        st.subheader("📈 Emotion Timeline")

        emotion_map = {
            'anger': 0,
            'happy': 1,
            'panic': 2,
            'sadness': 3
        }

        emotion_values = [
            emotion_map[e] for e in emotions
        ]

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            times,
            emotion_values,
            marker='o'
        )

        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(CLASS_NAMES)

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Emotion')
        ax.set_title('Emotion Timeline')

        st.pyplot(fig)

        # =========================
        # 감정 변화 분석
        # =========================
        st.subheader("⚡ 감정 변화 분석")

        changes = []

        for i in range(1, len(emotions)):

            if emotions[i] != emotions[i - 1]:

                changes.append({
                    'Time': times[i],
                    'Before': emotions[i - 1],
                    'After': emotions[i]
                })

        if len(changes) > 0:

            change_df = pd.DataFrame(changes)

            st.dataframe(change_df)

        else:

            st.info("감정 변화가 감지되지 않았습니다.")

# =========================
# 하단 설명
# =========================
st.markdown("---")

st.markdown("""
### 📌 프로젝트 설명

FGAI 시스템은 단순 감정 분류를 넘어,

시간 흐름 기반 감정 변화와 그룹 다이나믹을 분석하는

FGI 감정 분석 AI 플랫폼입니다.
""")