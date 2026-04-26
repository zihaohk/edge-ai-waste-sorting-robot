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
CONF_THRES = 0.6
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
# Game 02 Setting
# ===================================
CAN_CLASSES = {"coca cola can", "sprite"}
BOX_CLASSES = {"vita lemon tea", "vita soybean drink"}
VALID_TARGET_CLASSES = CAN_CLASSES | BOX_CLASSES

FAMILY_CAN = "can"
FAMILY_BOX = "box"
LEFT_SIDE = "left"
RIGHT_SIDE = "right"

LEFT_BEACON_CLASS = "green ball"
RIGHT_BEACON_CLASS = "red ball"

MAX_DROP_PER_SIDE = 2
MAX_TOTAL_DROP_COUNT = 4

GRAB_APPROACH_FAST_SPEED = 20000
GRAB_APPROACH_SLOW_SPEED = 11000
BEACON_APPROACH_FAST_SPEED = 20000
BEACON_APPROACH_SLOW_SPEED = 15000
TURN_SPEED = 12000
TARGET_CORRECTION_TURN_SPEED = TURN_SPEED
BEACON_CORRECTION_TURN_SPEED = TURN_SPEED
RETREAT_SPEED = 20000
SEARCH_TARGET_TURN_SPEED = 8000

BEACON_AUTO_LINE_X1 = 310
BEACON_AUTO_LINE_X2 = 410
BEACON_ALIGN_WIDTH_THRESHOLD = 120
BEACON_RELEASE_WIDTH_THRESHOLD = 220
BEACON_SLOW_WIDTH_THRESHOLD = 200

DROP_SHIFT_TURN_LEFT_SEC = 0.7
DROP_SHIFT_TURN_RIGHT_SEC = 0.9
DROP_SHIFT_BACK_TURN_LEFT_SEC = DROP_SHIFT_TURN_LEFT_SEC
DROP_SHIFT_BACK_TURN_RIGHT_SEC = DROP_SHIFT_TURN_RIGHT_SEC / 1.2
DROP_SHIFT_FORWARD_SEC = 2
DROP_FINAL_FORWARD_SEC = 1.5
DROP_COLUMN_TURN_SIGNS = (-1, 1)
DROP_LABELS = ("LEFT", "RIGHT")

HEARTBEAT_SEC = 0.12
MIN_EFFECTIVE_SPEED = 10000

GRIPPER_OPEN_S1 = 1000
GRIPPER_CLOSE_S1 = 1550
GRIPPER_S2 = 1400

GRAB_SETTLE_SEC = 0.60
GRAB_RECHECK_DELAY_SEC = 0.20
ENABLE_REGRAB = False
RELEASE_SETTLE_SEC = 0.50
BEACON_CORRECTION_TURN_SEC = 0.05
BEACON_CORRECTION_SETTLE_SEC = 0.08
RETREAT_AFTER_DROP_SEC = 4.5
TURN_TO_TARGET_BEACON_SEC = 1.60
TURN_BACK_TO_ITEMS_SEC = 1.5
WAIT_BEFORE_SEARCH_TARGET_SEC = 0.5
STARTUP_WAIT_SEC = 0.5
EXIT_STOP_SETTLE_SEC = 0.2
EXIT_RETREAT_SEC = 1.0

STATE_STARTUP_WAIT = "STARTUP_WAIT"
STATE_SEARCH_TARGET = "SEARCH_TARGET"
STATE_APPROACH_TARGET = "APPROACH_TARGET"
STATE_GRAB_TARGET = "GRAB_TARGET"
STATE_RETREAT_AFTER_GRAB = "RETREAT_AFTER_GRAB"
STATE_TURN_TO_TARGET_BEACON = "TURN_TO_TARGET_BEACON"
STATE_SEARCH_TARGET_BEACON = "SEARCH_TARGET_BEACON"
STATE_APPROACH_TARGET_BEACON = "APPROACH_TARGET_BEACON"
STATE_TURN_TO_DROP_COLUMN = "TURN_TO_DROP_COLUMN"
STATE_DRIVE_TO_DROP_COLUMN = "DRIVE_TO_DROP_COLUMN"
STATE_TURN_BACK_FROM_DROP_COLUMN = "TURN_BACK_FROM_DROP_COLUMN"
STATE_APPROACH_DROP_AFTER_SHIFT = "APPROACH_DROP_AFTER_SHIFT"
STATE_RELEASE_TARGET = "RELEASE_TARGET"
STATE_RETREAT_AFTER_DROP = "RETREAT_AFTER_DROP"
STATE_TURN_BACK_TO_ITEMS = "TURN_BACK_TO_ITEMS"
STATE_WAIT_BEFORE_SEARCH_TARGET = "WAIT_BEFORE_SEARCH_TARGET"
STATE_FINISHED = "FINISHED"
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


@dataclass(frozen=True)
class GrabConfig:
    auto_line_x1: int
    auto_line_x2: int
    grab_near_width_threshold: int
    grab_slow_width_threshold: int
    regrab_width_drop_ratio: float
    regrab_width_drop_px: int
    correction_turn_sec: float
    correction_settle_sec: float
    retreat_after_grab_sec: float
    slow_forward_nudge_sec: float = 0.0


CAN_GRAB_CONFIG = GrabConfig(
    auto_line_x1=320,
    auto_line_x2=440,
    grab_near_width_threshold=740,
    grab_slow_width_threshold=180,
    regrab_width_drop_ratio=0.6,
    regrab_width_drop_px=120,
    correction_turn_sec=0.005,
    correction_settle_sec=0.08,
    retreat_after_grab_sec=4,
    slow_forward_nudge_sec=0.10,
)

BOX_GRAB_CONFIG = GrabConfig(
    auto_line_x1=320,
    auto_line_x2=450,
    grab_near_width_threshold=610,
    grab_slow_width_threshold=180,
    regrab_width_drop_ratio=0.5,
    regrab_width_drop_px=150,
    correction_turn_sec=0.02,
    correction_settle_sec=0.08,
    retreat_after_grab_sec=4.0,
    slow_forward_nudge_sec=0.0,
)


current_drive = (0, 0, 0, 0, "STOP")
last_sent_time = 0.0

current_state = STATE_SEARCH_TARGET
state_deadline = None
state_started_at = 0.0

gripper_open = False
grab_reference_width = 0

target_locked = False
locked_target = None
beacon_locked = False
locked_beacon_target = None

correction_scope = None
correction_drive = None
correction_deadline = None
correction_settle_deadline = None
correction_stop_label = None
correction_followup_drive = None
correction_followup_deadline = None

carried_target_class = None
carried_target_family = None
target_beacon_class = None
target_side = None

left_drop_count = 0
right_drop_count = 0


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


def clear_target_lock():
    global target_locked, locked_target
    target_locked = False
    locked_target = None


def clear_beacon_lock():
    global beacon_locked, locked_beacon_target
    beacon_locked = False
    locked_beacon_target = None


def clear_carried_target_context():
    global carried_target_class, carried_target_family, target_beacon_class, target_side
    carried_target_class = None
    carried_target_family = None
    target_beacon_class = None
    target_side = None


def lock_target(target):
    global target_locked, locked_target
    if target is not None:
        target_locked = True
        locked_target = target


def lock_beacon_target(target):
    global beacon_locked, locked_beacon_target
    if target is not None:
        beacon_locked = True
        locked_beacon_target = target


def get_tracked_target(candidates):
    global locked_target
    if locked_target is None or not candidates:
        return None

    candidates.sort(
        key=lambda det: (
            abs(det.center_x - locked_target.center_x),
            abs(det.center_y - locked_target.center_y),
            abs(det.width - locked_target.width),
            det.center_x,
            -det.conf,
        )
    )
    locked_target = candidates[0]
    return locked_target


def get_tracked_beacon_target(current_target):
    global locked_beacon_target
    if current_target is not None:
        locked_beacon_target = current_target
    if beacon_locked and locked_beacon_target is not None:
        return locked_beacon_target
    return current_target


def get_target_family(class_name):
    if class_name in CAN_CLASSES:
        return FAMILY_CAN
    if class_name in BOX_CLASSES:
        return FAMILY_BOX
    return None


def get_target_side(family):
    if family == FAMILY_CAN:
        return LEFT_SIDE
    if family == FAMILY_BOX:
        return RIGHT_SIDE
    return None


def get_target_beacon_for_side(side):
    if side == LEFT_SIDE:
        return LEFT_BEACON_CLASS
    if side == RIGHT_SIDE:
        return RIGHT_BEACON_CLASS
    return None


def get_grab_config(family):
    if family == FAMILY_CAN:
        return CAN_GRAB_CONFIG
    if family == FAMILY_BOX:
        return BOX_GRAB_CONFIG
    return CAN_GRAB_CONFIG


def set_carried_target_context(class_name):
    global carried_target_class, carried_target_family, target_beacon_class, target_side
    carried_target_class = class_name
    carried_target_family = get_target_family(class_name)
    target_side = get_target_side(carried_target_family)
    target_beacon_class = get_target_beacon_for_side(target_side)


def get_side_drop_count(side):
    if side == LEFT_SIDE:
        return left_drop_count
    if side == RIGHT_SIDE:
        return right_drop_count
    return 0


def get_total_drop_count():
    return left_drop_count + right_drop_count


def is_finished():
    return (
        (left_drop_count >= MAX_DROP_PER_SIDE and right_drop_count >= MAX_DROP_PER_SIDE)
        or get_total_drop_count() >= MAX_TOTAL_DROP_COUNT
    )


def get_available_target_classes():
    available = set()
    if left_drop_count < MAX_DROP_PER_SIDE:
        available.update(CAN_CLASSES)
    if right_drop_count < MAX_DROP_PER_SIDE:
        available.update(BOX_CLASSES)
    return available


def get_current_drop_turn_sign(side):
    current_drop_count = get_side_drop_count(side)
    index = min(current_drop_count, len(DROP_COLUMN_TURN_SIGNS) - 1)
    return DROP_COLUMN_TURN_SIGNS[index]


def get_current_drop_label(side):
    current_drop_count = get_side_drop_count(side)
    if current_drop_count >= len(DROP_LABELS):
        return "DONE"
    return DROP_LABELS[current_drop_count]


def set_state(new_state, now, deadline=None):
    global current_state, state_started_at, state_deadline
    current_state = new_state
    state_started_at = now
    state_deadline = deadline
    if new_state not in (STATE_APPROACH_TARGET, STATE_GRAB_TARGET):
        clear_target_lock()
    if new_state in (STATE_WAIT_BEFORE_SEARCH_TARGET, STATE_SEARCH_TARGET, STATE_FINISHED):
        clear_beacon_lock()
        clear_carried_target_context()
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


def find_leftmost_target(detections, keep_locked_target=False):
    available_target_classes = get_available_target_classes()
    target_detections = [det for det in detections if det.class_name in available_target_classes]
    if not target_detections:
        return None

    if keep_locked_target and target_locked:
        tracked_target = get_tracked_target(target_detections)
        if tracked_target is not None:
            return tracked_target

    target_detections.sort(key=lambda det: (det.center_x, -det.conf, -det.width))
    return target_detections[0]


def overlap_length(a1, a2, b1, b2):
    return max(0, min(a2, b2) - max(a1, b1))


def find_locked_target_proxy(detections):
    global locked_target
    if locked_target is None:
        return None

    proxy_candidates = []
    for det in detections:
        x_overlap = overlap_length(det.x1, det.x2, locked_target.x1, locked_target.x2)
        y_overlap = overlap_length(det.y1, det.y2, locked_target.y1, locked_target.y2)
        min_width = min(det.width, locked_target.width)
        min_height = min(det.height, locked_target.height)

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
            -(overlap_length(det.x1, det.x2, locked_target.x1, locked_target.x2) *
              overlap_length(det.y1, det.y2, locked_target.y1, locked_target.y2)),
            abs(det.center_x - locked_target.center_x),
            abs(det.center_y - locked_target.center_y),
            abs(det.width - locked_target.width),
            -det.conf,
        )
    )
    locked_target = proxy_candidates[0]
    return locked_target


def choose_reacquire_drive(grab_config):
    if locked_target is None:
        return stop_drive("TARGET_REACQUIRE")

    if locked_target.center_x < grab_config.auto_line_x1:
        return left_drive(SEARCH_TARGET_TURN_SPEED, "TARGET_REACQUIRE_LEFT")
    if locked_target.center_x > grab_config.auto_line_x2:
        return right_drive(SEARCH_TARGET_TURN_SPEED, "TARGET_REACQUIRE_RIGHT")
    return stop_drive("TARGET_REACQUIRE")


def find_best_detection_by_class(detections, class_name_lower):
    matches = [det for det in detections if det.class_name == class_name_lower]
    if not matches:
        return None

    matches.sort(key=lambda det: (det.conf, det.width), reverse=True)
    return matches[0]


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


def get_turn_to_beacon_drive(side, label_prefix):
    if side == LEFT_SIDE:
        return left_drive(TURN_SPEED, f"{label_prefix}_LEFT")
    return right_drive(TURN_SPEED, f"{label_prefix}_RIGHT")


def get_turn_back_to_items_drive(side, label_prefix):
    if side == LEFT_SIDE:
        return right_drive(TURN_SPEED, f"{label_prefix}_RIGHT")
    return left_drive(TURN_SPEED, f"{label_prefix}_LEFT")


def get_display_target_label(target, is_proxy):
    if target is None:
        return carried_target_class or "none"
    if is_proxy:
        return f"LOCKED_PROXY:{target.class_name}"
    return target.class_name


def run_exit_retreat_sequence():
    open_gripper()
    time.sleep(RELEASE_SETTLE_SEC)

    stop_now = time.monotonic()
    force_stop(stop_now, "EXIT_STOP")
    time.sleep(EXIT_STOP_SETTLE_SEC)

    retreat_deadline = time.monotonic() + EXIT_RETREAT_SEC
    while True:
        now = time.monotonic()
        if now >= retreat_deadline:
            break
        send_drive(backward_drive(RETREAT_SPEED, "EXIT_RETREAT"), now)
        resend_drive_if_needed(now)
        time.sleep(HEARTBEAT_SEC / 2)

    force_stop(time.monotonic(), "EXIT_FINAL_STOP")


startup_now = time.monotonic()
open_gripper()
force_stop(startup_now, "STARTUP_STOP")
set_state(STATE_STARTUP_WAIT, startup_now, startup_now + STARTUP_WAIT_SEC)
exit_requested = False


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        infer_start = time.time()
        results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRES)
        # Show every detection from the model, then overlay game-specific highlights.
        annotated = results[0].plot()
        fps = 1.0 / max(time.time() - infer_start, 1e-6)
        now = time.monotonic()

        detections = parse_detections(results[0], model.names)

        keep_locked_target = current_state in (STATE_APPROACH_TARGET, STATE_GRAB_TARGET)
        target = find_leftmost_target(detections, keep_locked_target=keep_locked_target)
        target_is_proxy = False
        if target is None and keep_locked_target and target_locked:
            target = find_locked_target_proxy(detections)
            target_is_proxy = target is not None

        active_grab_config = get_grab_config(carried_target_family)

        beacon_target = None
        tracked_beacon_target = None
        if current_state in (
            STATE_TURN_TO_TARGET_BEACON,
            STATE_SEARCH_TARGET_BEACON,
            STATE_APPROACH_TARGET_BEACON,
        ) and target_beacon_class is not None:
            beacon_target = find_best_detection_by_class(detections, target_beacon_class)
            tracked_beacon_target = get_tracked_beacon_target(beacon_target)

        desired_drive = current_drive
        current_target_text = get_display_target_label(target, target_is_proxy)
        target_text = "none"
        target_pick_text = "LEFTMOST"
        family_text = carried_target_family or "none"
        side_text = target_side or "none"
        beacon_text = target_beacon_class or "none"
        drop_label_text = get_current_drop_label(target_side) if target_side is not None else "none"

        if target_is_proxy and target is not None:
            target_pick_text = "LEFT_LOCKED_PROXY"
        elif keep_locked_target and target_locked:
            target_pick_text = "LEFT_LOCKED"

        if target is not None:
            target_text = (
                f"{target.class_name} conf={target.conf:.2f} "
                f"center=({target.center_x},{target.center_y}) "
                f"width={target.width}"
            )
        elif carried_target_class is not None:
            target_text = f"{carried_target_class} (locked)"

        if tracked_beacon_target is not None and target_beacon_class is not None:
            beacon_text = (
                f"{target_beacon_class} "
                f"center=({tracked_beacon_target.center_x},{tracked_beacon_target.center_y}) "
                f"width={tracked_beacon_target.width}"
            )

        if current_state == STATE_STARTUP_WAIT:
            current_target_text = "none"
            target_pick_text = "WAIT_STARTUP"
            desired_drive = stop_drive("STARTUP_WAIT")
            if state_deadline is not None and now >= state_deadline:
                set_state(STATE_SEARCH_TARGET, now)

        elif current_state == STATE_SEARCH_TARGET:
            current_target_text = target.class_name if target is not None else "none"
            target_pick_text = "LEFTMOST"
            if target is None:
                desired_drive = right_drive(SEARCH_TARGET_TURN_SPEED, "SEARCH_TARGET")
            else:
                set_carried_target_context(target.class_name)
                family_text = carried_target_family or "none"
                side_text = target_side or "none"
                beacon_text = target_beacon_class or "none"
                drop_label_text = get_current_drop_label(target_side)
                active_grab_config = get_grab_config(carried_target_family)
                lock_target(target)
                desired_drive = stop_drive("TARGET_FOUND")
                set_state(STATE_APPROACH_TARGET, now)

        elif current_state == STATE_APPROACH_TARGET:
            current_target_text = carried_target_class or current_target_text
            target_correction_drive = get_active_correction("TARGET", now)
            if target_correction_drive is not None:
                desired_drive = target_correction_drive
            elif target is None:
                if target_locked:
                    desired_drive = choose_reacquire_drive(active_grab_config)
                else:
                    desired_drive = stop_drive("TARGET_LOST")
                    set_state(STATE_SEARCH_TARGET, now)
            elif target.center_x < active_grab_config.auto_line_x1:
                slow_mode_followup = (
                    active_grab_config.slow_forward_nudge_sec > 0
                    and target.width >= active_grab_config.grab_slow_width_threshold
                )
                desired_drive = start_correction(
                    "TARGET",
                    left_drive(TARGET_CORRECTION_TURN_SPEED, "TARGET_LEFT"),
                    now,
                    active_grab_config.correction_turn_sec,
                    active_grab_config.correction_settle_sec,
                    "TARGET_CORRECTION_SETTLE",
                    followup_drive=forward_drive(GRAB_APPROACH_SLOW_SPEED, "TARGET_FORWARD_NUDGE") if slow_mode_followup else None,
                    followup_sec=active_grab_config.slow_forward_nudge_sec if slow_mode_followup else 0.0,
                )
            elif target.center_x > active_grab_config.auto_line_x2:
                slow_mode_followup = (
                    active_grab_config.slow_forward_nudge_sec > 0
                    and target.width >= active_grab_config.grab_slow_width_threshold
                )
                desired_drive = start_correction(
                    "TARGET",
                    right_drive(TARGET_CORRECTION_TURN_SPEED, "TARGET_RIGHT"),
                    now,
                    active_grab_config.correction_turn_sec,
                    active_grab_config.correction_settle_sec,
                    "TARGET_CORRECTION_SETTLE",
                    followup_drive=forward_drive(GRAB_APPROACH_SLOW_SPEED, "TARGET_FORWARD_NUDGE") if slow_mode_followup else None,
                    followup_sec=active_grab_config.slow_forward_nudge_sec if slow_mode_followup else 0.0,
                )
            elif target.width < active_grab_config.grab_near_width_threshold:
                target_speed = choose_two_stage_speed(
                    target.width,
                    active_grab_config.grab_slow_width_threshold,
                    GRAB_APPROACH_SLOW_SPEED,
                    GRAB_APPROACH_FAST_SPEED,
                )
                desired_drive = forward_drive(target_speed, "APPROACH_TARGET")
            else:
                grab_reference_width = target.width
                close_gripper()
                desired_drive = stop_drive("GRAB_TARGET")
                set_state(STATE_GRAB_TARGET, now, now + GRAB_SETTLE_SEC)

        elif current_state == STATE_GRAB_TARGET:
            current_target_text = carried_target_class or current_target_text
            desired_drive = stop_drive("GRAB_TARGET")
            if ENABLE_REGRAB and target is not None and now - state_started_at >= GRAB_RECHECK_DELAY_SEC:
                regrab_width_threshold = max(
                    int(grab_reference_width * active_grab_config.regrab_width_drop_ratio),
                    grab_reference_width - active_grab_config.regrab_width_drop_px,
                )
                if target.width <= regrab_width_threshold:
                    open_gripper()
                    desired_drive = stop_drive("REGRAB_TARGET")
                    set_state(STATE_APPROACH_TARGET, now)
                elif state_deadline is not None and now >= state_deadline:
                    desired_drive = backward_drive(RETREAT_SPEED, "RETREAT_AFTER_GRAB")
                    set_state(STATE_RETREAT_AFTER_GRAB, now, now + active_grab_config.retreat_after_grab_sec)
            elif state_deadline is not None and now >= state_deadline:
                desired_drive = backward_drive(RETREAT_SPEED, "RETREAT_AFTER_GRAB")
                set_state(STATE_RETREAT_AFTER_GRAB, now, now + active_grab_config.retreat_after_grab_sec)

        elif current_state == STATE_RETREAT_AFTER_GRAB:
            current_target_text = carried_target_class or "none"
            desired_drive = backward_drive(RETREAT_SPEED, "RETREAT_AFTER_GRAB")
            if state_deadline is not None and now >= state_deadline:
                desired_drive = get_turn_to_beacon_drive(target_side, "TURN_TO_TARGET_BEACON")
                set_state(STATE_TURN_TO_TARGET_BEACON, now, now + TURN_TO_TARGET_BEACON_SEC)

        elif current_state == STATE_TURN_TO_TARGET_BEACON:
            current_target_text = target_beacon_class or "none"
            if beacon_target is not None:
                lock_beacon_target(beacon_target)
                desired_drive = stop_drive("BEACON_FOUND")
                set_state(STATE_APPROACH_TARGET_BEACON, now)
            else:
                desired_drive = get_turn_to_beacon_drive(target_side, "TURN_TO_TARGET_BEACON")
                if state_deadline is not None and now >= state_deadline:
                    set_state(STATE_SEARCH_TARGET_BEACON, now)

        elif current_state == STATE_SEARCH_TARGET_BEACON:
            current_target_text = target_beacon_class or "none"
            if beacon_target is not None:
                lock_beacon_target(beacon_target)
                desired_drive = stop_drive("BEACON_FOUND")
                set_state(STATE_APPROACH_TARGET_BEACON, now)
            else:
                desired_drive = get_turn_to_beacon_drive(target_side, "SEARCH_TARGET_BEACON")

        elif current_state == STATE_APPROACH_TARGET_BEACON:
            current_target_text = target_beacon_class or "none"
            beacon_correction_drive = get_active_correction("BEACON", now)
            if beacon_correction_drive is not None:
                desired_drive = beacon_correction_drive
            elif tracked_beacon_target is None:
                desired_drive = stop_drive("BEACON_LOST")
                set_state(STATE_SEARCH_TARGET_BEACON, now)
            elif tracked_beacon_target.center_x < BEACON_AUTO_LINE_X1:
                desired_drive = start_correction(
                    "BEACON",
                    left_drive(BEACON_CORRECTION_TURN_SPEED, "BEACON_LEFT"),
                    now,
                    BEACON_CORRECTION_TURN_SEC,
                    BEACON_CORRECTION_SETTLE_SEC,
                    "BEACON_CORRECTION_SETTLE",
                )
            elif tracked_beacon_target.center_x > BEACON_AUTO_LINE_X2:
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
                desired_drive = forward_drive(beacon_speed, "APPROACH_TARGET_BEACON")
            else:
                drop_turn_sign = get_current_drop_turn_sign(target_side)
                desired_drive = build_drop_turn_drive(drop_turn_sign, TURN_SPEED, "DROP_SHIFT_OUT")
                set_state(STATE_TURN_TO_DROP_COLUMN, now, now + get_drop_turn_duration(drop_turn_sign))

        elif current_state == STATE_TURN_TO_DROP_COLUMN:
            current_target_text = target_beacon_class or "none"
            drop_turn_sign = get_current_drop_turn_sign(target_side)
            desired_drive = build_drop_turn_drive(drop_turn_sign, TURN_SPEED, "DROP_SHIFT_OUT")
            if state_deadline is not None and now >= state_deadline:
                desired_drive = forward_drive(BEACON_APPROACH_SLOW_SPEED, "DROP_SHIFT_FORWARD")
                set_state(STATE_DRIVE_TO_DROP_COLUMN, now, now + DROP_SHIFT_FORWARD_SEC)

        elif current_state == STATE_DRIVE_TO_DROP_COLUMN:
            current_target_text = target_beacon_class or "none"
            desired_drive = forward_drive(BEACON_APPROACH_SLOW_SPEED, "DROP_SHIFT_FORWARD")
            if state_deadline is not None and now >= state_deadline:
                return_turn_sign = -get_current_drop_turn_sign(target_side)
                desired_drive = build_drop_turn_drive(return_turn_sign, TURN_SPEED, "DROP_SHIFT_BACK")
                set_state(STATE_TURN_BACK_FROM_DROP_COLUMN, now, now + get_drop_back_turn_duration(return_turn_sign))

        elif current_state == STATE_TURN_BACK_FROM_DROP_COLUMN:
            current_target_text = target_beacon_class or "none"
            return_turn_sign = -get_current_drop_turn_sign(target_side)
            desired_drive = build_drop_turn_drive(return_turn_sign, TURN_SPEED, "DROP_SHIFT_BACK")
            if state_deadline is not None and now >= state_deadline:
                desired_drive = forward_drive(BEACON_APPROACH_SLOW_SPEED, "APPROACH_DROP_AFTER_SHIFT")
                set_state(STATE_APPROACH_DROP_AFTER_SHIFT, now, now + DROP_FINAL_FORWARD_SEC)

        elif current_state == STATE_APPROACH_DROP_AFTER_SHIFT:
            current_target_text = target_beacon_class or "none"
            desired_drive = forward_drive(BEACON_APPROACH_SLOW_SPEED, "APPROACH_DROP_AFTER_SHIFT")
            if state_deadline is not None and now >= state_deadline:
                open_gripper()
                desired_drive = stop_drive("RELEASE_TARGET")
                set_state(STATE_RELEASE_TARGET, now, now + RELEASE_SETTLE_SEC)

        elif current_state == STATE_RELEASE_TARGET:
            current_target_text = target_beacon_class or "none"
            desired_drive = stop_drive("RELEASE_TARGET")
            if state_deadline is not None and now >= state_deadline:
                if target_side == LEFT_SIDE:
                    left_drop_count += 1
                elif target_side == RIGHT_SIDE:
                    right_drop_count += 1
                desired_drive = backward_drive(RETREAT_SPEED, "RETREAT_AFTER_DROP")
                set_state(STATE_RETREAT_AFTER_DROP, now, now + RETREAT_AFTER_DROP_SEC)

        elif current_state == STATE_RETREAT_AFTER_DROP:
            current_target_text = target_beacon_class or "none"
            desired_drive = backward_drive(RETREAT_SPEED, "RETREAT_AFTER_DROP")
            if state_deadline is not None and now >= state_deadline:
                if is_finished():
                    desired_drive = stop_drive("FINISHED")
                    set_state(STATE_FINISHED, now)
                else:
                    desired_drive = get_turn_back_to_items_drive(target_side, "TURN_BACK_TO_ITEMS")
                    set_state(STATE_TURN_BACK_TO_ITEMS, now, now + TURN_BACK_TO_ITEMS_SEC)

        elif current_state == STATE_TURN_BACK_TO_ITEMS:
            current_target_text = carried_target_class or "none"
            desired_drive = get_turn_back_to_items_drive(target_side, "TURN_BACK_TO_ITEMS")
            if state_deadline is not None and now >= state_deadline:
                desired_drive = stop_drive("WAIT_BEFORE_SEARCH_TARGET")
                set_state(STATE_WAIT_BEFORE_SEARCH_TARGET, now, now + WAIT_BEFORE_SEARCH_TARGET_SEC)

        elif current_state == STATE_WAIT_BEFORE_SEARCH_TARGET:
            current_target_text = "none"
            desired_drive = stop_drive("WAIT_BEFORE_SEARCH_TARGET")
            if state_deadline is not None and now >= state_deadline:
                desired_drive = right_drive(SEARCH_TARGET_TURN_SPEED, "SEARCH_TARGET")
                set_state(STATE_SEARCH_TARGET, now)

        elif current_state == STATE_FINISHED:
            current_target_text = "none"
            desired_drive = stop_drive("FINISHED")

        else:
            desired_drive = stop_drive("UNKNOWN_STATE")
            set_state(STATE_FINISHED, now)

        send_drive(desired_drive, now)
        resend_drive_if_needed(now)

        line_x1 = None
        line_x2 = None
        if current_state in (STATE_APPROACH_TARGET, STATE_GRAB_TARGET):
            line_x1 = active_grab_config.auto_line_x1
            line_x2 = active_grab_config.auto_line_x2
        elif current_state in (
            STATE_TURN_TO_TARGET_BEACON,
            STATE_SEARCH_TARGET_BEACON,
            STATE_APPROACH_TARGET_BEACON,
        ):
            line_x1 = BEACON_AUTO_LINE_X1
            line_x2 = BEACON_AUTO_LINE_X2

        if line_x1 is not None and line_x2 is not None:
            cv2.line(annotated, (line_x1, 0), (line_x1, FRAME_HEIGHT), (255, 255, 0), 2)
            cv2.line(annotated, (line_x2, 0), (line_x2, FRAME_HEIGHT), (255, 255, 0), 2)

        if target is not None:
            cv2.rectangle(
                annotated,
                (target.x1, target.y1),
                (target.x2, target.y2),
                (0, 255, 255),
                3,
            )
            cv2.circle(annotated, (target.center_x, target.center_y), 6, (0, 255, 255), -1)

        if tracked_beacon_target is not None:
            beacon_color = (0, 255, 0) if target_side == LEFT_SIDE else (0, 0, 255)
            cv2.rectangle(
                annotated,
                (tracked_beacon_target.x1, tracked_beacon_target.y1),
                (tracked_beacon_target.x2, tracked_beacon_target.y2),
                beacon_color,
                3,
            )
            cv2.circle(annotated, (tracked_beacon_target.center_x, tracked_beacon_target.center_y), 6, beacon_color, -1)

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
            f"CMD: {current_drive[4]}",
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"TARGET: {current_target_text}",
            (10, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"TARGET_INFO: {target_text}",
            (10, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"TARGET_PICK: {target_pick_text}",
            (10, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"FAMILY: {family_text}  SIDE: {side_text}",
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
        cv2.putText(
            annotated,
            f"LEFT_DROPPED: {left_drop_count}/{MAX_DROP_PER_SIDE}  RIGHT_DROPPED: {right_drop_count}/{MAX_DROP_PER_SIDE}",
            (10, 305),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"DROP_ROUTE: {drop_label_text}",
            (10, 335),
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
            (10, 365),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Game02", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            exit_requested = True
            break

finally:
    if exit_requested:
        run_exit_retreat_sequence()
    else:
        shutdown_now = time.monotonic()
        force_stop(shutdown_now, "EXIT_STOP")
    cap.release()
    cv2.destroyAllWindows()
