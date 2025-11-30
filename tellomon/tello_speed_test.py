from djitellopy import Tello
import time

def main():
    tello = Tello()

    print("[INFO] Connecting to Tello...")
    tello.connect()

    # 배터리 확인
    print(f"[INFO] Battery: {tello.get_battery()}%")

    print("[INFO] Reading speed values... (Ctrl+C to stop)")

    try:
        while True:
            vx = tello.get_speed_x()   # cm/s
            vy = tello.get_speed_y()
            vz = tello.get_speed_z()

            print(f"Speed -> X:{vx:4d}  Y:{vy:4d}  Z:{vz:4d} (cm/s)")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("[INFO] Stop reading speeds.")
    finally:
        tello.end()

if __name__ == "__main__":
    main()
