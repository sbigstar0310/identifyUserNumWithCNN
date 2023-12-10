import tensorflow as tf
import cv2
import numpy as np

# 모델 로드
model = tf.keras.models.load_model("mnist-myModel.h5")

# 카메라 열기
video = cv2.VideoCapture(0)  # 0은 기본 카메라, 웹캠을 사용하는 경우 0 이상의 값을 사용

while True:
    # 프레임 읽기
    ret, frame = video.read()

    # 이미지의 높이와 너비 가져오기
    height, width, _ = frame.shape

    # mnist 데이터는 정방향의 사각형에 숫자가 꽉 차있다.
    # 그러한 데이터 셋에 맞추기 위해 직접 정방향 사각형 프레임으로 자른다.
    # 자를 크기 정의
    crop_size = 280  # 이미지 크기가 (28 * 28) 이므로 280으로 지정함.
    # 중앙 좌표 계산
    center_y, center_x = height // 2, width // 2
    # 이미지를 중앙에서 crop_size의 반만큼 위, 아래, 왼쪽, 오른쪽으로 자르기
    top = center_y - crop_size // 2
    bottom = center_y + crop_size // 2
    left = center_x - crop_size // 2
    right = center_x + crop_size // 2
    # 이미지 자르기
    cropped_frame = frame[top:bottom, left:right]

    if ret:
        cv2.waitKey(10)

        # 프레임 전처리
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (28, 28))

        # Invert the colors 0 -> 1, 1 -> 0
        inverted_frame = cv2.bitwise_not(resized_frame)

        # 모델 입력 형태에 맞게 전처리
        input_data = inverted_frame.reshape((1, 28, 28, 1)).astype("float32")

        im = input_data / 255.0
    else:
        print("no image from video")
        exit()

    # 숫자 예측
    prediction = model.predict(im)
    prob = tf.nn.softmax(prediction).numpy()
    predicted_digit = np.argmax(prob)

    # 화면에 숫자 표시
    cv2.putText(
        frame,
        f"Predicted Digit: {predicted_digit}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # 가운데 인식을 도와줄 격자 사각형 레이아웃 그리기
    rect_width, rect_height = 280, 280  # make sure same with cropped_frame
    rect_x = (width - rect_width) // 2
    rect_y = (height - rect_height) // 2

    # Create a blank image with the same size as the frame
    layout = np.zeros_like(frame)
    # Draw a white rectangle on the layout image
    cv2.rectangle(
        layout,
        (rect_x, rect_y),
        (rect_x + rect_width, rect_y + rect_height),
        (0, 0, 255),
        2,
    )
    # Overlay the layout on top of the original frame
    frameWithLayout = cv2.addWeighted(frame, 1, layout, 1, 0)
    # Display the combined frame
    cv2.imshow("Number Recognition", frameWithLayout)

    # 종료 조건
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# 종료 시 리소스 해제
video.release()
cv2.destroyAllWindows()
