import cv2
import mediapipe as mp

# MediaPipe 포즈 모델 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def analyze_pose(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 찾을 수 없음.")
        return None

    # BGR → RGB 변환 (MediaPipe는 RGB를 씀)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 포즈 감지
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print("사람을 감지하지 못했어요. 전신이 잘 나온 사진인지 확인하세요.")
        return None

    # 감지된 관절 개수 출력
    landmarks = results.pose_landmarks.landmark
    print(f"관절 감지 성공! 총 {len(landmarks)}개 랜드마크 검출")

    # 몇 가지 주요 관절 좌표 출력
    print(f"왼쪽 어깨: x={landmarks[11].x:.3f}, y={landmarks[11].y:.3f}")
    print(f"오른쪽 어깨: x={landmarks[12].x:.3f}, y={landmarks[12].y:.3f}")
    print(f"왼쪽 엉덩이: x={landmarks[23].x:.3f}, y={landmarks[23].y:.3f}")
    print(f"오른쪽 엉덩이: x={landmarks[24].x:.3f}, y={landmarks[24].y:.3f}")
    print(f"왼쪽 무릎  신뢰도: {landmarks[25].visibility:.2f}")
    print(f"오른쪽 무릎 신뢰도: {landmarks[26].visibility:.2f}")
    print(f"왼쪽 발목  신뢰도: {landmarks[27].visibility:.2f}")
    print(f"오른쪽 발목 신뢰도: {landmarks[28].visibility:.2f}")

    # 스켈레톤 그려서 결과 이미지 저장
    annotated = image.copy()
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
    output_path = "samples/result.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"결과 이미지 저장됨: {output_path}")

    return landmarks

# 직접 실행할 때 테스트
if __name__ == "__main__":
    analyze_pose("samples/ramon-test.jpg")