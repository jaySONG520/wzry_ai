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
    parser.add_argument(
        "--keymap",
        type=str,
        default="record_demo_keymap.json",
        help="Path to keymap JSON file.",
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


def _print_controls(hero_type, keymap):
    print("\n================= 控制说明 =================")
    stop_label = keymap["stop"] if keymap.get("stop") else "无"
    print(f"移动: {keymap['move']} (释放移动: {stop_label})")
    print(
        f"攻击: {keymap['attack']} (普通攻击), "
        f"{keymap['skill1']}/{keymap['skill2']}/{keymap['skill3']} (技能1/2/3)"
    )
    if hero_type == 4:
        print(f"四技能英雄额外技能: {keymap['skill4']} (技能4)")
    print(f"小兵/塔: {keymap['attack_minion']} (攻击小兵), {keymap['attack_tower']} (攻击塔)")
    print(f"回城/恢复: {keymap['recall']} (回城), {keymap['recover']} (恢复)")
    print(f"召唤师/装备技能: {keymap['summoner']} (召唤师), {keymap['equipment']} (装备)")
    print(
        f"信息类(信号/升级): {keymap['signal_attack']}/{keymap['signal_retreat']}/{keymap['signal_collect']} "
        f"(发起进攻/开始撤退/请求集合), {keymap['upgrade1']}/{keymap['upgrade2']}/"
        f"{keymap['upgrade3']}/{keymap['upgrade4']} (升级技能)"
    )
    print(f"购买装备: {keymap['buy1']} (装备1), {keymap['buy2']} (装备2)")
    if keymap.get("record_start"):
        print(f"开始录制: {keymap['record_start']}")
    if keymap.get("record_stop"):
        print(f"停止录制: {keymap['record_stop']}")
    if keymap.get("quit"):
        print(f"退出: {keymap['quit']}")
    else:
        print("退出: 无 (可用 Ctrl+C)")
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


def _update_action_from_key(key, action_state, hero_type, keymap):
    if key is None:
        return None
    key = key.lower()
    move_angles = {
        keymap["move"][0].lower(): 270,
        keymap["move"][1].lower(): 180,
        keymap["move"][2].lower(): 90,
        keymap["move"][3].lower(): 0,
    }
    if key in move_angles:
        action_state.move_action = 1
        action_state.angle = move_angles[key]
        return "move"
    if keymap.get("stop") and key == keymap["stop"].lower():
        action_state.move_action = 0
        return "stop"
    if key == keymap["attack"].lower():
        action_state.attack_action = 1
        return "attack"
    if key == keymap["skill1"].lower():
        action_state.attack_action = 8
        return "skill1"
    if key == keymap["skill2"].lower():
        action_state.attack_action = 9
        return "skill2"
    if key == keymap["skill3"].lower():
        action_state.attack_action = 10
        return "skill3"
    if key == keymap["skill4"].lower() and hero_type == 4:
        action_state.attack_action = 11
        return "skill4"
    if key == keymap["attack_minion"].lower():
        action_state.attack_action = 2
        return "attack_minion"
    if key == keymap["attack_tower"].lower():
        action_state.attack_action = 3
        return "attack_tower"
    if key == keymap["recall"].lower():
        action_state.attack_action = 4
        return "recall"
    if key == keymap["recover"].lower():
        action_state.attack_action = 5
        return "recover"
    if key == keymap["equipment"].lower():
        action_state.attack_action = 6
        return "equipment"
    if key == keymap["summoner"].lower():
        action_state.attack_action = 7
        return "summoner"
    if key == keymap["signal_attack"].lower():
        action_state.info_action = 3
        return "signal_attack"
    if key == keymap["signal_retreat"].lower():
        action_state.info_action = 4
        return "signal_retreat"
    if key == keymap["signal_collect"].lower():
        action_state.info_action = 5
        return "signal_collect"
    if key == keymap["upgrade1"].lower():
        action_state.info_action = 6
        return "upgrade1"
    if key == keymap["upgrade2"].lower():
        action_state.info_action = 7
        return "upgrade2"
    if key == keymap["upgrade3"].lower():
        action_state.info_action = 8
        return "upgrade3"
    if key == keymap["upgrade4"].lower() and hero_type == 4:
        action_state.info_action = 9
        return "upgrade4"
    if key == keymap["buy1"].lower():
        action_state.info_action = 1
        return "buy1"
    if key == keymap["buy2"].lower():
        action_state.info_action = 2
        return "buy2"
    return None


def _load_keymap(path):
    default = {
        "move": ["w", "a", "s", "d"],
        "stop": "",
        "attack": "mouse_right",
        "skill1": "q",
        "skill2": "e",
        "skill3": "r",
        "skill4": "i",
        "attack_minion": "z",
        "attack_tower": "x",
        "recall": "b",
        "recover": "c",
        "equipment": "6",
        "summoner": "f",
        "signal_attack": "h",
        "signal_retreat": "j",
        "signal_collect": "k",
        "upgrade1": "1",
        "upgrade2": "2",
        "upgrade3": "3",
        "upgrade4": "6",
        "buy1": "4",
        "buy2": "5",
        "record_start": "=",
        "record_stop": "-",
        "quit": "",
    }
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return default
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key, value in default.items():
        if key not in data:
            data[key] = value
    if not isinstance(data["move"], list) or len(data["move"]) != 4:
        data["move"] = default["move"]
    return data


def _mouse_right_pressed():
    if os.name != "nt":
        return False
    import ctypes

    return bool(ctypes.windll.user32.GetAsyncKeyState(0x02) & 0x8000)


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

    keymap = _load_keymap(args.keymap)
    _print_controls(hero_type, keymap)
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
        "keymap": keymap,
    }
    with open(os.path.join(session_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    action_log_path = os.path.join(session_dir, "actions.jsonl")
    tool = AndroidTool()
    action_state = ActionState()

    frame_idx = 0
    start_time = time.time()
    last_frame_time = 0.0
    recording_enabled = True
    if keymap.get("record_start") or keymap.get("record_stop"):
        recording_enabled = False
    print(f"录制目录: {session_dir}")
    if recording_enabled:
        print("已开始录制。")
    else:
        print("等待开始录制...按开始键启动。")

    try:
        while True:
            if args.max_seconds and time.time() - start_time >= args.max_seconds:
                print("已达到最大录制时长，退出。")
                break

            key = _read_key()
            if keymap.get("quit") and key and key.lower() == keymap["quit"].lower():
                print("收到退出指令，结束录制。")
                break
            if keymap.get("record_start") and key and key.lower() == keymap["record_start"].lower():
                if not recording_enabled:
                    recording_enabled = True
                    print("已开始录制。")
                continue
            if keymap.get("record_stop") and key and key.lower() == keymap["record_stop"].lower():
                if recording_enabled:
                    recording_enabled = False
                    print("已停止录制。")
                continue

            _update_action_from_key(key, action_state, hero_type, keymap)
            if keymap["attack"] == "mouse_right" and _mouse_right_pressed():
                action_state.attack_action = 1

            now = time.time()
            if now - last_frame_time < 1.0 / args.fps:
                time.sleep(0.001)
                continue
            last_frame_time = now
            if not recording_enabled:
                continue

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
