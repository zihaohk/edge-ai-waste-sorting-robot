from dataclasses import dataclass
from ultralytics import YOLO
import cv2
import time
import os
import sys


# Ensure DISPLAY is available before importing GUI/video backends on Linux.
os.environ.setdefault("DISPLAY", ":0")


# ===================================
# Robot Communication Setting
# ===================================
try:
    from Library_Robot.lib3360 import motor, servo
except Exception:
    import importlib.util
    import pathlib

    lib_path = pathlib.Path(__file__).resolve().parent / "Library_Robot" / "lib3360.py"
    spec = importlib.util.spec_from_file_location("lib3360", str(lib_path))
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    motor = getattr(lib, "motor")
    servo = getattr(lib, "servo")

PORT = None
if os.getenv("SENDSERIAL_PORT"):
    PORT = os.getenv("SENDSERIAL_PORT")
elif len(sys.argv) > 1:
    PORT = sys.argv[1]
# ===================================


# ===================================
# AI Model Setting
# ===================================
MODEL_PATH = os.getenv("AIR_MODEL_PATH", "/home/jetson/Documents/IC_AI_03/AI_Model_game01/v9/best.engine")
IMG_SIZE = (640, 640)
CONF_THRES = 0.8
FRAME_WIDTH = 800
FRAME_HEIGHT = 600

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    raise SystemExit("Error: Could not open webcam.")
# ===================================


# ===================================
# Game 01 Setting
# ===================================
SUGAR_CLASS = "coca cola can"
ZERO_CLASS = "coca cola can zero sugar"
LEFT_BEACON_CLASS = "green ball"
MAX_DROP_COUNT = 3

AUTO_LINE_X1 = 340
AUTO_LINE_X2 = 460

GRAB_NEAR_WIDTH_THRESHOLD = 740
GRAB_SLOW_WIDTH_THRESHOLD = 180
ANY_CAN_NEAR_WIDTH_THRESHOLD = 500
BEACON_ALIGN_WIDTH_THRESHOLD = 100
BEACON_RELEASE_WIDTH_THRESHOLD = 220
BEACON_SLOW_WIDTH_THRESHOLD = 200
DROP_SHIFT_TURN_LEFT_SEC = 0.7
DROP_SHIFT_TURN_RIGHT_SEC = 0.9
DROP_SHIFT_BACK_TURN_LEFT_SEC = DROP_SHIFT_TURN_LEFT_SEC 
DROP_SHIFT_BACK_TURN_RIGHT_SEC = DROP_SHIFT_TURN_RIGHT_SEC / 1.2
DROP_SHIFT_FORWARD_SEC = 2
DROP_FINAL_FORWARD_SEC = 2
DROP_COLUMN_TURN_SIGNS = (-1, 0, 1)

GRAB_APPROACH_FAST_SPEED = 20000
GRAB_APPROACH_SLOW_SPEED = 11000
BEACON_APPROACH_FAST_SPEED = 20000
BEACON_APPROACH_SLOW_SPEED = 15000
TURN_SPEED = 12000
SUGAR_CORRECTION_TURN_SPEED = 12000
BEACON_CORRECTION_TURN_SPEED = TURN_SPEED
RETREAT_SPEED = 20000
SEARCH_CAN_TURN_SPEED = 8000
DIAGONAL_INNER_RATIO = 0.55

HEARTBEAT_SEC = 0.12
MIN_EFFECTIVE_SPEED = 10000

GRIPPER_OPEN_S1 = 1000
GRIPPER_CLOSE_S1 = 1550
GRIPPER_S2 = 1400

GRAB_SETTLE_SEC = 0.60
GRAB_RECHECK_DELAY_SEC = 0.20
ENABLE_REGRAB = False
REGRAB_WIDTH_DROP_RATIO = 0.6
REGRAB_WIDTH_DROP_PX = 120
RELEASE_SETTLE_SEC = 0.50
SUGAR_CORRECTION_TURN_SEC = 0.005
BEACON_CORRECTION_TURN_SEC = 0.05
SUGAR_CORRECTION_SETTLE_SEC = 0.08
BEACON_CORRECTION_SETTLE_SEC = 0.08
SUGAR_SLOW_FORWARD_NUDGE_SEC = 0.10
RETREAT_AFTER_GRAB_SEC = 4
RETREAT_AFTER_DROP_SEC = 4
TURN_LEFT_TO_BEACON_SEC = 1.60
TURN_RIGHT_TO_CANS_SEC = 1.50
WAIT_BEFORE_SEARCH_SUGAR_CAN_SEC = 0.5

SUGAR_CLASS_LOWER = SUGAR_CLASS.lower()
ZERO_CLASS_LOWER = ZERO_CLASS.lower()
LEFT_BEACON_CLASS_LOWER = LEFT_BEACON_CLASS.lower()

STATE_SEARCH_SUGAR_CAN = "SEARCH_SUGAR_CAN"
STATE_APPROACH_SUGAR_CAN = "APPROACH_SUGAR_CAN"
STATE_GRAB_SUGAR_CAN = "GRAB_SUGAR_CAN"
STATE_RETREAT_AFTER_GRAB = "RETREAT_AFTER_GRAB"
STATE_TURN_LEFT_TO_BEACON = "TURN_LEFT_TO_BEACON"
STATE_SEARCH_LEFT_BEACON = "SEARCH_LEFT_BEACON"
STATE_APPROACH_BEACON = "APPROACH_BEACON"
STATE_APPROACH_MID_DROP_BY_BEACON = "APPROACH_MID_DROP_BY_BEACON"
STATE_TURN_TO_DROP_COLUMN = "TURN_TO_DROP_COLUMN"
STATE_DRIVE_TO_DROP_COLUMN = "DRIVE_TO_DROP_COLUMN"
STATE_TURN_BACK_FROM_DROP_COLUMN = "TURN_BACK_FROM_DROP_COLUMN"
STATE_APPROACH_DROP_AFTER_SHIFT = "APPROACH_DROP_AFTER_SHIFT"
STATE_RELEASE_CAN = "RELEASE_CAN"
STATE_RETREAT_AFTER_DROP = "RETREAT_AFTER_DROP"
STATE_TURN_RIGHT_TO_CANS = "TURN_RIGHT_TO_CANS"
STATE_WAIT_BEFORE_SEARCH_SUGAR_CAN = "WAIT_BEFORE_SEARCH_SUGAR_CAN"
STATE_FINISHED = "FINISHED"

DROP_LABELS = ("LEFT", "MID", "RIGHT")
# ===================================


@dataclass(frozen=True)
class Detection:
    class_name: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int
    width: int
    height: int


current_drive = (0, 0, 0, 0, "STOP")
last_sent_time = 0.0

current_state = STATE_SEARCH_SUGAR_CAN
state_deadline = None
state_started_at = 0.0

gripper_open = False
drop_index = 0
grab_reference_width = 0
sugar_close_priority_locked = False
locked_sugar_target = None
beacon_locked = False
locked_beacon_target = None
correction_scope = None
correction_drive = None
correction_deadline = None
correction_settle_deadline = None
correction_stop_label = None
correction_followup_drive = None
correction_followup_deadline = None


def normalize_motor_speed(speed):
    speed = int(speed)
    if speed <= 0:
        return 0
    return max(speed, MIN_EFFECTIVE_SPEED)


def normalize_drive(drive):
    m1, m2, d1, d2, label = drive
    return (
        normalize_motor_speed(m1),
        normalize_motor_speed(m2),
        int(d1),
        int(d2),
        label,
    )


def send_drive(drive, now):
    global current_drive, last_sent_time
    drive = normalize_drive(drive)
    if drive != current_drive:
        m1, m2, d1, d2, _ = drive
        motor(m1, m2, dir1=d1, dir2=d2, mode="once", port=PORT)
        current_drive = drive
        last_sent_time = now


def resend_drive_if_needed(now):
    global last_sent_time
    if now - last_sent_time >= HEARTBEAT_SEC:
        m1, m2, d1, d2, _ = current_drive
        motor(m1, m2, dir1=d1, dir2=d2, mode="once", port=PORT)
        last_sent_time = now


def force_stop(now, label="STOP"):
    global current_drive, last_sent_time
    motor(0, 0, dir1=0, dir2=0, mode="once", port=PORT)
    current_drive = (0, 0, 0, 0, label)
    last_sent_time = now


def clear_correction():
    global correction_scope, correction_drive, correction_deadline, correction_settle_deadline, correction_stop_label
    global correction_followup_drive, correction_followup_deadline
    correction_scope = None
    correction_drive = None
    correction_deadline = None
    correction_settle_deadline = None
    correction_stop_label = None
    correction_followup_drive = None
    correction_followup_deadline = None


def clear_beacon_lock():
    global beacon_locked, locked_beacon_target
    beacon_locked = False
    locked_beacon_target = None


def clear_sugar_lock():
    global sugar_close_priority_locked, locked_sugar_target
    sugar_close_priority_locked = False
    locked_sugar_target = None


def lock_sugar_target(target):
    global sugar_close_priority_locked, locked_sugar_target
    if target is not None:
        sugar_close_priority_locked = True
        locked_sugar_target = target


def lock_beacon_target(target):
    global beacon_locked, locked_beacon_target
    if target is not None:
        beacon_locked = True
        locked_beacon_target = target


def get_tracked_beacon_target(current_target):
    global locked_beacon_target
    if current_target is not None:
        locked_beacon_target = current_target
    if beacon_locked and locked_beacon_target is not None:
        return locked_beacon_target
    return current_target


def get_tracked_sugar_target(candidates):
    global locked_sugar_target
    if locked_sugar_target is None or not candidates:
        return None

    # Keep following the same can by image position, instead of switching
    # targets whenever the other can looks slightly larger in one frame.
    candidates.sort(
        key=lambda det: (
            abs(det.center_x - locked_sugar_target.center_x),
            abs(det.center_y - locked_sugar_target.center_y),
            abs(det.width - locked_sugar_target.width),
            det.center_x,
            -det.conf,
        )
    )
    locked_sugar_target = candidates[0]
    return locked_sugar_target


def set_state(new_state, now, deadline=None):
    global current_state, state_started_at, state_deadline
    current_state = new_state
    state_started_at = now
    state_deadline = deadline
    if new_state not in (STATE_APPROACH_SUGAR_CAN, STATE_GRAB_SUGAR_CAN):
        clear_sugar_lock()
    if new_state == STATE_SEARCH_SUGAR_CAN:
        clear_beacon_lock()
    clear_correction()


def open_gripper():
    global gripper_open
    if not gripper_open:
        servo(GRIPPER_OPEN_S1, GRIPPER_S2, mode="hex")
        gripper_open = True


def close_gripper():
    global gripper_open
    if gripper_open:
        servo(GRIPPER_CLOSE_S1, GRIPPER_S2, mode="hex")
        gripper_open = False


def stop_drive(label):
    return 0, 0, 0, 0, label


def forward_drive(speed, label):
    return speed, speed, 1, 1, label


def backward_drive(speed, label):
    return speed, speed, 0, 0, label


def left_drive(speed, label):
    return speed, speed, 0, 1, label


def right_drive(speed, label):
    return speed, speed, 1, 0, label


def parse_detections(result, names):
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    for box, conf, cls_id in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = map(int, box)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        detections.append(
            Detection(
                class_name=str(names[int(cls_id)]).lower(),
                conf=float(conf),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                center_x=center_x,
                center_y=center_y,
                width=max(0, x2 - x1),
                height=max(0, y2 - y1),
            )
        )

    return detections


def find_leftmost_sugar_can(detections, keep_locked_target=False):
    sugar_detections = [det for det in detections if det.class_name == SUGAR_CLASS_LOWER]
    if not sugar_detections:
        return None

    if keep_locked_target and sugar_close_priority_locked:
        tracked_target = get_tracked_sugar_target(sugar_detections)
        if tracked_target is not None:
            return tracked_target

    sugar_detections.sort(key=lambda det: (det.center_x, -det.conf))
    return sugar_detections[0]


def overlap_length(a1, a2, b1, b2):
    return max(0, min(a2, b2) - max(a1, b1))


def find_locked_sugar_proxy_target(detections):
    global locked_sugar_target
    if locked_sugar_target is None:
        return None

    proxy_candidates = []
    for det in detections:
        x_overlap = overlap_length(det.x1, det.x2, locked_sugar_target.x1, locked_sugar_target.x2)
        y_overlap = overlap_length(det.y1, det.y2, locked_sugar_target.y1, locked_sugar_target.y2)
        min_width = min(det.width, locked_sugar_target.width)
        min_height = min(det.height, locked_sugar_target.height)

        if min_width <= 0 or min_height <= 0:
            continue
        if x_overlap < int(min_width * 0.35):
            continue
        if y_overlap < int(min_height * 0.35):
            continue

        proxy_candidates.append(det)

    if not proxy_candidates:
        return None

    proxy_candidates.sort(
        key=lambda det: (
            -(overlap_length(det.x1, det.x2, locked_sugar_target.x1, locked_sugar_target.x2) *
              overlap_length(det.y1, det.y2, locked_sugar_target.y1, locked_sugar_target.y2)),
            abs(det.center_x - locked_sugar_target.center_x),
            abs(det.center_y - locked_sugar_target.center_y),
            abs(det.width - locked_sugar_target.width),
            -det.conf,
        )
    )
    locked_sugar_target = proxy_candidates[0]
    return locked_sugar_target


def choose_reacquire_drive():
    if locked_sugar_target is None:
        return stop_drive("SUGAR_REACQUIRE")

    if locked_sugar_target.center_x < AUTO_LINE_X1:
        return left_drive(SEARCH_CAN_TURN_SPEED, "SUGAR_REACQUIRE_LEFT")
    if locked_sugar_target.center_x > AUTO_LINE_X2:
        return right_drive(SEARCH_CAN_TURN_SPEED, "SUGAR_REACQUIRE_RIGHT")
    return stop_drive("SUGAR_REACQUIRE")


def find_best_detection_by_class(detections, class_name_lower):
    matches = [det for det in detections if det.class_name == class_name_lower]
    if not matches:
        return None

    matches.sort(key=lambda det: (det.conf, det.width), reverse=True)
    return matches[0]


def is_can_detection(det):
    return "can" in det.class_name


def find_single_near_can(detections, near_width_threshold):
    can_detections = [det for det in detections if is_can_detection(det)]
    if len(can_detections) != 1:
        return None

    only_can = can_detections[0]
    if only_can.width < near_width_threshold:
        return None
    return only_can


def get_drop_turn_sign(current_drop_index):
    return DROP_COLUMN_TURN_SIGNS[min(current_drop_index, len(DROP_COLUMN_TURN_SIGNS) - 1)]


def get_drop_turn_duration(turn_sign):
    if turn_sign < 0:
        return DROP_SHIFT_TURN_LEFT_SEC
    if turn_sign > 0:
        return DROP_SHIFT_TURN_RIGHT_SEC
    return 0.0


def get_drop_back_turn_duration(turn_sign):
    if turn_sign < 0:
        return DROP_SHIFT_BACK_TURN_LEFT_SEC
    if turn_sign > 0:
        return DROP_SHIFT_BACK_TURN_RIGHT_SEC
    return 0.0


def build_drop_turn_drive(turn_sign, speed, label_prefix):
    if turn_sign < 0:
        return left_drive(speed, f"{label_prefix}_LEFT")
    if turn_sign > 0:
        return right_drive(speed, f"{label_prefix}_RIGHT")
    return stop_drive(f"{label_prefix}_MID")


def choose_two_stage_speed(current_width, slow_width_threshold, slow_speed, fast_speed):
    if current_width >= slow_width_threshold:
        return slow_speed
    return fast_speed


def start_correction(scope, drive, now, duration, settle_sec, stop_label, followup_drive=None, followup_sec=0.0):
    global correction_scope, correction_drive, correction_deadline, correction_settle_deadline, correction_stop_label
    global correction_followup_drive, correction_followup_deadline
    correction_scope = scope
    correction_drive = normalize_drive(drive)
    correction_deadline = now + duration
    correction_settle_deadline = correction_deadline + settle_sec
    correction_stop_label = stop_label
    if followup_drive is not None and followup_sec > 0:
        correction_followup_drive = normalize_drive(followup_drive)
        correction_followup_deadline = correction_settle_deadline + followup_sec
    else:
        correction_followup_drive = None
        correction_followup_deadline = None
    return correction_drive


def get_active_correction(scope, now):
    if correction_scope != scope or correction_drive is None or correction_deadline is None:
        return None
    if now < correction_deadline:
        return correction_drive
    if correction_settle_deadline is not None and now < correction_settle_deadline:
        return stop_drive(correction_stop_label or f"{scope}_CORRECTION_SETTLE")
    if correction_followup_drive is not None and correction_followup_deadline is not None and now < correction_followup_deadline:
        return correction_followup_drive
    clear_correction()
    return None


startup_now = time.monotonic()
open_gripper()
force_stop(startup_now, "STARTUP_STOP")
clear_beacon_lock()
set_state(STATE_SEARCH_SUGAR_CAN, startup_now)


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        infer_start = time.time()
        results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRES)
        annotated = results[0].plot()
        fps = 1.0 / max(time.time() - infer_start, 1e-6)
        now = time.monotonic()

        detections = parse_detections(results[0], model.names)
        keep_locked_sugar_target = current_state in (STATE_APPROACH_SUGAR_CAN, STATE_GRAB_SUGAR_CAN)
        sugar_target = find_leftmost_sugar_can(
            detections,
            keep_locked_target=keep_locked_sugar_target,
        )
        sugar_target_is_proxy = False
        if sugar_target is None and keep_locked_sugar_target and sugar_close_priority_locked:
            sugar_target = find_locked_sugar_proxy_target(detections)
            sugar_target_is_proxy = sugar_target is not None
        beacon_target = find_best_detection_by_class(detections, LEFT_BEACON_CLASS_LOWER)
        tracked_beacon_target = get_tracked_beacon_target(beacon_target)
        zero_count = sum(1 for det in detections if det.class_name == ZERO_CLASS_LOWER)

        desired_drive = current_drive
        current_target_text = "none"
        sugar_text = "none"
        sugar_pick_text = "LEFTMOST"
        beacon_text = "none"
        active_can_target = sugar_target
        active_can_target_label = SUGAR_CLASS

        if sugar_target_is_proxy and active_can_target is not None:
            active_can_target_label = f"LOCKED_PROXY:{active_can_target.class_name}"
            sugar_pick_text = "LEFT_LOCKED_PROXY"
        elif keep_locked_sugar_target and sugar_close_priority_locked:
            sugar_pick_text = "LEFT_LOCKED"
        elif keep_locked_sugar_target:
            sugar_pick_text = "LEFTMOST_WAIT_CLOSE"

        if active_can_target is not None:
            sugar_text = (
                f"{active_can_target.class_name} conf={active_can_target.conf:.2f} "
                f"center=({active_can_target.center_x},{active_can_target.center_y}) "
                f"width={active_can_target.width}"
            )

        if tracked_beacon_target is not None:
            beacon_text = (
                f"{LEFT_BEACON_CLASS} "
                f"center=({tracked_beacon_target.center_x},{tracked_beacon_target.center_y}) "
                f"width={tracked_beacon_target.width}"
            )

        if current_state == STATE_SEARCH_SUGAR_CAN:
            current_target_text = active_can_target_label if active_can_target is not None else "none"
            if active_can_target is None:
                desired_drive = right_drive(SEARCH_CAN_TURN_SPEED, "SEARCH_SUGAR_CAN")
            else:
                lock_sugar_target(active_can_target)
                desired_drive = stop_drive("SUGAR_FOUND")
                set_state(STATE_APPROACH_SUGAR_CAN, now)

        elif current_state == STATE_APPROACH_SUGAR_CAN:
            current_target_text = active_can_target_label
            sugar_correction_drive = get_active_correction("SUGAR", now)
            if sugar_correction_drive is not None:
                desired_drive = sugar_correction_drive
            elif active_can_target is None:
                if sugar_close_priority_locked:
                    desired_drive = choose_reacquire_drive()
                else:
                    desired_drive = stop_drive("SUGAR_LOST")
                    set_state(STATE_SEARCH_SUGAR_CAN, now)
            elif active_can_target.center_x < AUTO_LINE_X1:
                slow_mode_followup = active_can_target.width >= GRAB_SLOW_WIDTH_THRESHOLD
                desired_drive = start_correction(
                    "SUGAR",
                    left_drive(SUGAR_CORRECTION_TURN_SPEED, "SUGAR_LEFT"),
                    now,
                    SUGAR_CORRECTION_TURN_SEC,
                    SUGAR_CORRECTION_SETTLE_SEC,
                    "SUGAR_CORRECTION_SETTLE",
                    followup_drive=forward_drive(GRAB_APPROACH_SLOW_SPEED, "SUGAR_FORWARD_NUDGE") if slow_mode_followup else None,
                    followup_sec=SUGAR_SLOW_FORWARD_NUDGE_SEC if slow_mode_followup else 0.0,
                )
            elif active_can_target.center_x > AUTO_LINE_X2:
                slow_mode_followup = active_can_target.width >= GRAB_SLOW_WIDTH_THRESHOLD
                desired_drive = start_correction(
                    "SUGAR",
                    right_drive(SUGAR_CORRECTION_TURN_SPEED, "SUGAR_RIGHT"),
                    now,
                    SUGAR_CORRECTION_TURN_SEC,
                    SUGAR_CORRECTION_SETTLE_SEC,
                    "SUGAR_CORRECTION_SETTLE",
                    followup_drive=forward_drive(GRAB_APPROACH_SLOW_SPEED, "SUGAR_FORWARD_NUDGE") if slow_mode_followup else None,
                    followup_sec=SUGAR_SLOW_FORWARD_NUDGE_SEC if slow_mode_followup else 0.0,
                )
            elif active_can_target.width < GRAB_NEAR_WIDTH_THRESHOLD:
                sugar_speed = choose_two_stage_speed(
                    active_can_target.width,
                    GRAB_SLOW_WIDTH_THRESHOLD,
                    GRAB_APPROACH_SLOW_SPEED,
                    GRAB_APPROACH_FAST_SPEED,
                )
                desired_drive = forward_drive(sugar_speed, "APPROACH_SUGAR")
            else:
                grab_reference_width = active_can_target.width
                close_gripper()
                desired_drive = stop_drive("GRAB_SUGAR_CAN")
                set_state(STATE_GRAB_SUGAR_CAN, now, now + GRAB_SETTLE_SEC)

        elif current_state == STATE_GRAB_SUGAR_CAN:
            current_target_text = active_can_target_label
            desired_drive = stop_drive("GRAB_SUGAR_CAN")
            if ENABLE_REGRAB and active_can_target is not None and now - state_started_at >= GRAB_RECHECK_DELAY_SEC:
                regrab_width_threshold = max(
                    int(grab_reference_width * REGRAB_WIDTH_DROP_RATIO),
                    grab_reference_width - REGRAB_WIDTH_DROP_PX,
                )
                if active_can_target.width <= regrab_width_threshold:
                    open_gripper()
                    desired_drive = stop_drive("REGRAB_SUGAR_CAN")
                    set_state(STATE_APPROACH_SUGAR_CAN, now)
                elif state_deadline is not None and now >= state_deadline:
                    desired_drive = backward_drive(RETREAT_SPEED, "RETREAT_AFTER_GRAB")
                    set_state(STATE_RETREAT_AFTER_GRAB, now, now + RETREAT_AFTER_GRAB_SEC)
            elif state_deadline is not None and now >= state_deadline:
                desired_drive = backward_drive(RETREAT_SPEED, "RETREAT_AFTER_GRAB")
                set_state(STATE_RETREAT_AFTER_GRAB, now, now + RETREAT_AFTER_GRAB_SEC)

        elif current_state == STATE_RETREAT_AFTER_GRAB:
            current_target_text = ZERO_CLASS
            desired_drive = backward_drive(RETREAT_SPEED, "RETREAT_AFTER_GRAB")
            if state_deadline is not None and now >= state_deadline:
                desired_drive = left_drive(TURN_SPEED, "TURN_LEFT_TO_BEACON")
                set_state(STATE_TURN_LEFT_TO_BEACON, now, now + TURN_LEFT_TO_BEACON_SEC)

        elif current_state == STATE_TURN_LEFT_TO_BEACON:
            current_target_text = LEFT_BEACON_CLASS
            if beacon_target is not None:
                lock_beacon_target(beacon_target)
                desired_drive = stop_drive("BEACON_FOUND")
                set_state(STATE_APPROACH_BEACON, now)
            else:
                desired_drive = left_drive(TURN_SPEED, "TURN_LEFT_TO_BEACON")
                if state_deadline is not None and now >= state_deadline:
                    set_state(STATE_SEARCH_LEFT_BEACON, now)

        elif current_state == STATE_SEARCH_LEFT_BEACON:
            current_target_text = LEFT_BEACON_CLASS
            if beacon_target is not None:
                lock_beacon_target(beacon_target)
                desired_drive = stop_drive("BEACON_FOUND")
                set_state(STATE_APPROACH_BEACON, now)
            else:
                desired_drive = left_drive(TURN_SPEED, "SEARCH_LEFT_BEACON")

        elif current_state == STATE_APPROACH_BEACON:
            current_target_text = LEFT_BEACON_CLASS
            beacon_correction_drive = get_active_correction("BEACON", now)
            if beacon_correction_drive is not None:
                desired_drive = beacon_correction_drive
            elif tracked_beacon_target is None:
                desired_drive = stop_drive("BEACON_LOST")
                set_state(STATE_SEARCH_LEFT_BEACON, now)
            elif tracked_beacon_target.center_x < AUTO_LINE_X1:
                desired_drive = start_correction(
                    "BEACON",
                    left_drive(BEACON_CORRECTION_TURN_SPEED, "BEACON_LEFT"),
                    now,
                    BEACON_CORRECTION_TURN_SEC,
                    BEACON_CORRECTION_SETTLE_SEC,
                    "BEACON_CORRECTION_SETTLE",
                )
            elif tracked_beacon_target.center_x > AUTO_LINE_X2:
                desired_drive = start_correction(
                    "BEACON",
                    right_drive(BEACON_CORRECTION_TURN_SPEED, "BEACON_RIGHT"),
                    now,
                    BEACON_CORRECTION_TURN_SEC,
                    BEACON_CORRECTION_SETTLE_SEC,
                    "BEACON_CORRECTION_SETTLE",
                )
            elif tracked_beacon_target.width < BEACON_ALIGN_WIDTH_THRESHOLD:
                beacon_speed = choose_two_stage_speed(
                    tracked_beacon_target.width,
                    BEACON_SLOW_WIDTH_THRESHOLD,
                    BEACON_APPROACH_SLOW_SPEED,
                    BEACON_APPROACH_FAST_SPEED,
                )
                desired_drive = forward_drive(beacon_speed, "APPROACH_BEACON")
            else:
                drop_turn_sign = get_drop_turn_sign(drop_index)
                if drop_turn_sign == 0:
                    desired_drive = forward_drive(BEACON_APPROACH_SLOW_SPEED, "APPROACH_MID_DROP_BY_BEACON")
                    set_state(STATE_APPROACH_MID_DROP_BY_BEACON, now)
                else:
                    desired_drive = build_drop_turn_drive(drop_turn_sign, TURN_SPEED, "DROP_SHIFT_OUT")
                    set_state(STATE_TURN_TO_DROP_COLUMN, now, now + get_drop_turn_duration(drop_turn_sign))

        elif current_state == STATE_APPROACH_MID_DROP_BY_BEACON:
            current_target_text = LEFT_BEACON_CLASS
            beacon_correction_drive = get_active_correction("BEACON", now)
            if beacon_correction_drive is not None:
                desired_drive = beacon_correction_drive
            elif tracked_beacon_target is None:
                desired_drive = forward_drive(BEACON_APPROACH_SLOW_SPEED, "MID_DROP_REACQUIRE")
            elif tracked_beacon_target.width >= BEACON_RELEASE_WIDTH_THRESHOLD:
                open_gripper()
                desired_drive = stop_drive("RELEASE_CAN")
                set_state(STATE_RELEASE_CAN, now, now + RELEASE_SETTLE_SEC)
            elif tracked_beacon_target.center_x < AUTO_LINE_X1:
                desired_drive = start_correction(
                    "BEACON",
                    left_drive(BEACON_CORRECTION_TURN_SPEED, "BEACON_LEFT"),
                    now,
                    BEACON_CORRECTION_TURN_SEC,
                    BEACON_CORRECTION_SETTLE_SEC,
                    "BEACON_CORRECTION_SETTLE",
                )
            elif tracked_beacon_target.center_x > AUTO_LINE_X2:
                desired_drive = start_correction(
                    "BEACON",
                    right_drive(BEACON_CORRECTION_TURN_SPEED, "BEACON_RIGHT"),
                    now,
                    BEACON_CORRECTION_TURN_SEC,
                    BEACON_CORRECTION_SETTLE_SEC,
                    "BEACON_CORRECTION_SETTLE",
                )
            else:
                desired_drive = forward_drive(BEACON_APPROACH_SLOW_SPEED, "APPROACH_MID_DROP_BY_BEACON")

        elif current_state == STATE_TURN_TO_DROP_COLUMN:
            current_target_text = LEFT_BEACON_CLASS
            drop_turn_sign = get_drop_turn_sign(drop_index)
            desired_drive = build_drop_turn_drive(drop_turn_sign, TURN_SPEED, "DROP_SHIFT_OUT")
            if state_deadline is not None and now >= state_deadline:
                desired_drive = forward_drive(BEACON_APPROACH_SLOW_SPEED, "DROP_SHIFT_FORWARD")
                set_state(STATE_DRIVE_TO_DROP_COLUMN, now, now + DROP_SHIFT_FORWARD_SEC)

        elif current_state == STATE_DRIVE_TO_DROP_COLUMN:
            current_target_text = LEFT_BEACON_CLASS
            desired_drive = forward_drive(BEACON_APPROACH_SLOW_SPEED, "DROP_SHIFT_FORWARD")
            if state_deadline is not None and now >= state_deadline:
                return_turn_sign = -get_drop_turn_sign(drop_index)
                if return_turn_sign == 0:
                    desired_drive = forward_drive(BEACON_APPROACH_SLOW_SPEED, "APPROACH_DROP_AFTER_SHIFT")
                    set_state(STATE_APPROACH_DROP_AFTER_SHIFT, now, now + DROP_FINAL_FORWARD_SEC)
                else:
                    desired_drive = build_drop_turn_drive(return_turn_sign, TURN_SPEED, "DROP_SHIFT_BACK")
                    set_state(STATE_TURN_BACK_FROM_DROP_COLUMN, now, now + get_drop_back_turn_duration(return_turn_sign))

        elif current_state == STATE_TURN_BACK_FROM_DROP_COLUMN:
            current_target_text = LEFT_BEACON_CLASS
            return_turn_sign = -get_drop_turn_sign(drop_index)
            desired_drive = build_drop_turn_drive(return_turn_sign, TURN_SPEED, "DROP_SHIFT_BACK")
            if state_deadline is not None and now >= state_deadline:
                desired_drive = forward_drive(BEACON_APPROACH_SLOW_SPEED, "APPROACH_DROP_AFTER_SHIFT")
                set_state(STATE_APPROACH_DROP_AFTER_SHIFT, now, now + DROP_FINAL_FORWARD_SEC)

        elif current_state == STATE_APPROACH_DROP_AFTER_SHIFT:
            current_target_text = LEFT_BEACON_CLASS
            desired_drive = forward_drive(BEACON_APPROACH_SLOW_SPEED, "APPROACH_DROP_AFTER_SHIFT")
            if state_deadline is not None and now >= state_deadline:
                open_gripper()
                desired_drive = stop_drive("RELEASE_CAN")
                set_state(STATE_RELEASE_CAN, now, now + RELEASE_SETTLE_SEC)

        elif current_state == STATE_RELEASE_CAN:
            current_target_text = LEFT_BEACON_CLASS
            desired_drive = stop_drive("RELEASE_CAN")
            if state_deadline is not None and now >= state_deadline:
                drop_index += 1
                desired_drive = backward_drive(RETREAT_SPEED, "RETREAT_AFTER_DROP")
                set_state(STATE_RETREAT_AFTER_DROP, now, now + RETREAT_AFTER_DROP_SEC)

        elif current_state == STATE_RETREAT_AFTER_DROP:
            current_target_text = LEFT_BEACON_CLASS
            desired_drive = backward_drive(RETREAT_SPEED, "RETREAT_AFTER_DROP")
            if state_deadline is not None and now >= state_deadline:
                if drop_index >= MAX_DROP_COUNT:
                    desired_drive = stop_drive("FINISHED")
                    set_state(STATE_FINISHED, now)
                else:
                    desired_drive = right_drive(TURN_SPEED, "TURN_RIGHT_TO_CANS")
                    set_state(STATE_TURN_RIGHT_TO_CANS, now, now + TURN_RIGHT_TO_CANS_SEC)

        elif current_state == STATE_TURN_RIGHT_TO_CANS:
            current_target_text = SUGAR_CLASS
            desired_drive = right_drive(TURN_SPEED, "TURN_RIGHT_TO_CANS")
            if state_deadline is not None and now >= state_deadline:
                desired_drive = stop_drive("WAIT_BEFORE_SEARCH_SUGAR_CAN")
                set_state(STATE_WAIT_BEFORE_SEARCH_SUGAR_CAN, now, now + WAIT_BEFORE_SEARCH_SUGAR_CAN_SEC)

        elif current_state == STATE_WAIT_BEFORE_SEARCH_SUGAR_CAN:
            current_target_text = SUGAR_CLASS
            desired_drive = stop_drive("WAIT_BEFORE_SEARCH_SUGAR_CAN")
            if state_deadline is not None and now >= state_deadline:
                desired_drive = right_drive(SEARCH_CAN_TURN_SPEED, "SEARCH_SUGAR_CAN")
                set_state(STATE_SEARCH_SUGAR_CAN, now)

        elif current_state == STATE_FINISHED:
            current_target_text = "none"
            desired_drive = stop_drive("FINISHED")

        else:
            desired_drive = stop_drive("UNKNOWN_STATE")
            set_state(STATE_FINISHED, now)

        send_drive(desired_drive, now)
        resend_drive_if_needed(now)

        cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"STATE: {current_state}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"STATE_T: {now - state_started_at:.1f}s",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"ROUND: {min(drop_index + 1, MAX_DROP_COUNT)}/{MAX_DROP_COUNT}  DROPPED: {drop_index}/{MAX_DROP_COUNT}",
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"CMD: {current_drive[4]}",
            (10, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"TARGET: {current_target_text}",
            (10, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"SUGAR: {sugar_text}  ZERO_COUNT: {zero_count}",
            (10, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"SUGAR_PICK: {sugar_pick_text}",
            (10, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"BEACON: {beacon_text}",
            (10, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        grip_text = "OPEN" if gripper_open else "CLOSED"
        cv2.putText(
            annotated,
            f"GRIPPER: {grip_text}",
            (10, 305),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        drop_label = DROP_LABELS[min(drop_index, len(DROP_LABELS) - 1)] if drop_index < MAX_DROP_COUNT else "DONE"
        cv2.putText(
            annotated,
            f"DROP COLUMN: {drop_label}",
            (10, 335),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.line(annotated, (AUTO_LINE_X1, 0), (AUTO_LINE_X1, 130), (255, 0, 0), 1)
        cv2.line(annotated, (AUTO_LINE_X2, 0), (AUTO_LINE_X2, 130), (255, 0, 0), 1)
        cv2.putText(annotated, f"{AUTO_LINE_X1}", (AUTO_LINE_X1 - 50, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"{AUTO_LINE_X2}", (AUTO_LINE_X2 + 10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)

        if active_can_target is not None:
            cv2.circle(annotated, (active_can_target.center_x, active_can_target.center_y), 6, (0, 255, 255), -1)

        if beacon_target is not None:
            cv2.rectangle(
                annotated,
                (beacon_target.x1, beacon_target.y1),
                (beacon_target.x2, beacon_target.y2),
                (0, 255, 0),
                2,
            )
            cv2.circle(annotated, (beacon_target.center_x, beacon_target.center_y), 6, (0, 255, 0), -1)

        cv2.putText(
            annotated,
            "AUTO GAME01: Q quit",
            (10, FRAME_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Game01 Auto State Machine", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    shutdown_now = time.monotonic()
    force_stop(shutdown_now, "STOP")
    cap.release()
    cv2.destroyAllWindows()
