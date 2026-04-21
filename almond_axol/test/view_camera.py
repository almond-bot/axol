import cv2
import pyzed.sl as sl

zed = sl.Camera()
init = sl.InitParameters()
init.input.set_from_stream("192.168.10.1", 30000)

err = zed.open(init)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"Failed to connect: {err}")
    exit(1)

print("Connected!")
image = sl.Mat()
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()[:, :, :3]
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

zed.close()
