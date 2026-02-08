import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import cv2

from android_tool import AndroidTool
from argparses import args as global_args
from argparses import attack_actions_detail, info_actions_detail
from onnxRunner import OnnxRunner


@dataclass
class ActionState:
    move_action: int = 0
    angle: int = 0
    info_action: int = 0
    attack_action: int = 0
    action_type: int = 0
    arg1: int = 0
    arg2: int = 0
    arg3: int = 0

    def to_list(self):
        return [
            self.move_action,
            self.angle,
            self.info_action,
            self.attack_action,
            self.action_type,
            self.arg1,
            self.arg2,
            self.arg3,
        ]

    def clear_instant_actions(self):
        self.info_action = 0
        self.attack_action = 0
        self.action_type = 0
        self.arg1 = 0
        self.arg2 = 0
        self.arg3 = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Record human demonstrations for wzry_ai.")
    parser.add_argument("--out_dir", type=str, default="data/demos", help="Base output directory.")
    parser.add_argument("--fps", type=float, default=8.0, help="Recording FPS.")
    parser.add_argument("--max_seconds", type=int, default=0, help="Stop after N seconds (0 for no limit).")
    parser.add_argument(
        "--require_started",
        action="store_true",
        help="Only record frames when start.onnx detects 'started'.",
    )
    return parser.parse_args()


def _now_ts():
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _load_start_checker():
    model_path = "models/start.onnx"
    if not os.path.exists(model_path):
        return None
    return OnnxRunner(model_path, classes=["started"])


def _print_controls(hero_type):
    print("\n================= 控制说明 =================")
    print("移动: W/A/S/D (释放移动: 空格)")
    print("攻击: J (普通攻击), K/L/U (技能1/2/3)")
    if hero_type == 4:
        print("四技能英雄额外技能: I (技能4)")
    print("小兵/塔: 1 (攻击小兵), 2 (攻击塔)")
    print("回城/恢复: R (回城), T (恢复)")
    print("信息类(信号/升级): G/H (发起进攻/开始撤退), 6/7/8/9 (升级技能)")
    print("退出: Q")
    print("============================================\n")


def _read_key_windows():
    import msvcrt

    if not msvcrt.kbhit():
        return None
    key = msvcrt.getch()
    if key in (b"\x00", b"\xe0"):
        key = msvcrt.getch()
        return f"__arrow__{key.decode(errors='ignore')}"
    try:
        return key.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _read_key():
    if os.name == "nt":
        return _read_key_windows()
    return None


def _update_action_from_key(key, action_state, hero_type):
    if key is None:
        return None
    key = key.lower()
    move_angles = {
        "w": 270,
        "a": 180,
        "s": 90,
        "d": 0,
    }
    if key in move_angles:
        action_state.move_action = 1
        action_state.angle = move_angles[key]
        return "move"
    if key == " ":
        action_state.move_action = 0
        return "stop"
    if key == "j":
        action_state.attack_action = 1
        return "attack"
    if key == "k":
        action_state.attack_action = 8
        return "skill1"
    if key == "l":
        action_state.attack_action = 9
        return "skill2"
    if key == "u":
        action_state.attack_action = 10
        return "skill3"
    if key == "i" and hero_type == 4:
        action_state.attack_action = 11
        return "skill4"
    if key == "1":
        action_state.attack_action = 2
        return "attack_minion"
    if key == "2":
        action_state.attack_action = 3
        return "attack_tower"
    if key == "r":
        action_state.attack_action = 4
        return "recall"
    if key == "t":
        action_state.attack_action = 5
        return "recover"
    if key == "g":
        action_state.info_action = 3
        return "signal_attack"
    if key == "h":
        action_state.info_action = 4
        return "signal_retreat"
    if key == "6":
        action_state.info_action = 6
        return "upgrade1"
    if key == "7":
        action_state.info_action = 7
        return "upgrade2"
    if key == "8":
        action_state.info_action = 8
        return "upgrade3"
    if key == "9" and hero_type == 4:
        action_state.info_action = 9
        return "upgrade4"
    return None


def _dispatch_action(tool, action_state):
    if action_state.move_action == 1:
        tool.action_move({"action": 1, "angle": action_state.angle})
    if action_state.info_action != 0:
        tool.action_info({"action": action_state.info_action})
    if action_state.attack_action != 0:
        tool.action_attack(
            {
                "action": action_state.attack_action,
                "action_type": action_state.action_type,
                "arg1": action_state.arg1,
                "arg2": action_state.arg2,
                "arg3": action_state.arg3,
            }
        )


def main():
    args = parse_args()
    hero_type = global_args.hero_type

    _print_controls(hero_type)
    start_checker = _load_start_checker() if args.require_started else None

    session_dir = os.path.join(args.out_dir, f"demo_{_now_ts()}")
    frames_dir = os.path.join(session_dir, "frames")
    _ensure_dir(frames_dir)

    meta = {
        "created_at": _now_ts(),
        "fps": args.fps,
        "hero_type": hero_type,
        "require_started": args.require_started,
        "attack_actions_detail": attack_actions_detail,
        "info_actions_detail": info_actions_detail,
        "controls": {
            "move": {"w": 270, "a": 180, "s": 90, "d": 0},
            "stop": "space",
            "attack": "j",
            "skill1": "k",
            "skill2": "l",
            "skill3": "u",
            "skill4": "i",
        },
    }
    with open(os.path.join(session_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    action_log_path = os.path.join(session_dir, "actions.jsonl")
    tool = AndroidTool()
    action_state = ActionState()

    frame_idx = 0
    start_time = time.time()
    last_frame_time = 0.0
    print(f"开始录制: {session_dir}")
    print("等待开始...按 Q 退出")

    try:
        while True:
            if args.max_seconds and time.time() - start_time >= args.max_seconds:
                print("已达到最大录制时长，退出。")
                break

            key = _read_key()
            if key in ("q", "Q"):
                print("收到退出指令，结束录制。")
                break

            _update_action_from_key(key, action_state, hero_type)

            now = time.time()
            if now - last_frame_time < 1.0 / args.fps:
                time.sleep(0.001)
                continue
            last_frame_time = now

            frame = tool.take_screenshot()
            if frame is None:
                continue

            if start_checker is not None:
                label = start_checker.get_max_label(frame)
                if label != "started":
                    continue

            frame_name = f"{frame_idx:06d}.jpg"
            frame_path = os.path.join(frames_dir, frame_name)
            cv2.imwrite(frame_path, frame)

            _dispatch_action(tool, action_state)

            record = {
                "frame": frame_name,
                "timestamp": now,
                "action": action_state.to_list(),
                "action_state": asdict(action_state),
            }
            with open(action_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            action_state.clear_instant_actions()
            frame_idx += 1

    except KeyboardInterrupt:
        print("录制中断，正在退出。")

    print(f"录制结束，共保存 {frame_idx} 帧。")


if __name__ == "__main__":
    if os.name != "nt":
        print("当前录制脚本在 Windows 上支持键盘捕获。")
        print("请在 Windows 环境运行该脚本。")
        sys.exit(1)
    main()
