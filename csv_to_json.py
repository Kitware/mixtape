#!/usr/bin/env python3
"""
csv_to_json.py

Convert an SC2 (or similar) decomposed episode CSV + a video replay file
into a Mixtape External Episode JSON with per-step base64-encoded PNG frames.

Dependencies (install if needed):
  pip install pandas opencv-python

Usage examples:
  csv_to_json.py \
    --csv decomposed_ep_1.csv \
    --video replay.mp4 \
    --out episode_from_csv_and_video.json \
    --env "SC2 Two Bridge" \
    --algo "Masked PPO" \
    --rewards combat_r nav_r term_r \
    --action-col A_verb \
    --action-map '{"0":1,"1":2,"2":0}' \
    --frame-col frame

  # If your CSV has a 'time_sec' column instead of 'frame':
  csv_to_json.py \
    --csv decomposed_ep_1.csv \
    --video replay.mp4 \
    --out episode_from_csv_and_video.json \
    --env "SC2 Two Bridge" \
    --algo "Masked PPO" \
    --rewards combat_r nav_r term_r \
    --action-col A_verb \
    --fps 30 \
    --time-col time_sec

Notes:
- You must choose exactly one of --frame-col or --time-col (with --fps).
- Action handling:
    * --action-col <name> picks the integer action id directly from CSV (or via --action-map remap).
    * If your CSV stores a categorical like 0=Move,1=Attack,2=No-Op, you can remap with --action-map.
      Example above maps A_verb 0->1(Move), 1->2(Attack), 2->0(No-Op).
- Observation vector is optional; you can include columns via --obs-cols pattern list or leave empty.
- The script uses "rewards": [...] (vector) consistently across steps.
- Images are pulled from the provided video at each step and base64-encoded PNGs.
"""

import argparse, json, base64, io, sys, math, re
from typing import List, Dict, Optional
import pandas as pd

try:
    import cv2
except Exception as e:
    cv2 = None

def b64_image_from_frame(frame_bgr, image_format: str = "png", jpeg_quality: int = 85) -> str:
    """Encode a BGR frame to base64 using PNG or JPEG."""
    import cv2 as _cv2
    fmt = (image_format or "png").lower()
    if fmt in ("jpg", "jpeg"):
        q = int(max(1, min(100, jpeg_quality)))
        ok, buf = _cv2.imencode(".jpg", frame_bgr, [_cv2.IMWRITE_JPEG_QUALITY, q])
    else:
        ok, buf = _cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError(f"OpenCV failed to encode frame to {fmt.upper()}")
    return base64.b64encode(buf.tobytes()).decode("ascii")

def read_frame(cap, frame_idx: int):
    # OpenCV can random-seek, but accuracy depends on codec. We still try; it's the standard approach.
    # Returns BGR frame or None if failed.
    if frame_idx < 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def maybe_resize(frame_bgr, max_w: Optional[int], max_h: Optional[int]):
    """Resize BGR frame preserving aspect ratio to fit within max_w x max_h.
    If neither max_w nor max_h provided, return original frame.
    """
    if max_w is None and max_h is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    scale = 1.0
    if max_w is not None and w > 0:
        scale = min(scale, float(max_w) / float(w))
    if max_h is not None and h > 0:
        scale = min(scale, float(max_h) / float(h))
    if scale >= 1.0:
        return frame_bgr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def to_float_or_none(val):
    try:
        f = float(val)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f

def parse_action(value, action_map: Optional[Dict[str,int]]) -> int:
    # Accept strings/ints/floats; cast to int; then optional remap
    try:
        v = int(value)
    except Exception:
        try:
            v = int(float(value))
        except Exception:
            raise ValueError(f"Cannot parse action value '{value}' as int")
    if action_map:
        # keys in JSON are strings; map if present, else pass through
        v = int(action_map.get(str(v), v))
    return v

def collect_observation(row: pd.Series, obs_cols: List[str]) -> Optional[List[float]]:
    if not obs_cols:
        return None
    obs = []
    for c in obs_cols:
        val = row.get(c, None)
        fv = to_float_or_none(val)
        if fv is None:
            obs.append(0.0)
        else:
            obs.append(fv)
    return obs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("episode", nargs="?", default=None, help="Episode name (e.g., 'Combat_Loss'). Auto-constructs paths if --csv/--video/--out not provided.")
    p.add_argument("--csv", default=None, help="Path to decomposed episode CSV (default: 'Unified CSVs/{episode}.csv')")
    p.add_argument("--video", default=None, help="Path to replay video file (default: 'Replay Files/{episode}.mov')")
    p.add_argument("--out", default=None, help="Output Mixtape JSON path (default: 'JSON Ingest Files/{episode}_FINAL.json')")
    p.add_argument("--env", default="SC2 Two Bridge", help="training.environment")
    p.add_argument("--algo", default="Masked PPO", help="training.algorithm")
    p.add_argument("--iterations", type=int, default=100, help="training.iterations")
    p.add_argument("--rewards", nargs="+", default=["nav_r", "combat_r", "term_r"], help="CSV column names for reward vector, in order (default: nav_r combat_r term_r)")
    p.add_argument("--action-col", default="A_verb", help="CSV column name holding action id / code (default: A_verb)")
    p.add_argument("--action-map", default=None, help='JSON dict string mapping incoming action to Mixtape action, e.g. {"0":1,"1":2,"2":0}')
    p.add_argument("--frame-col", default=None, help="CSV column with absolute frame index per row")
    p.add_argument("--time-col", default=None, help="CSV column with time in seconds per row (use with --fps)")
    p.add_argument("--row-as-frame", action="store_true", default=True, help="Treat each CSV row index as the video frame index (default: enabled)")
    p.add_argument("--fps", type=float, default=None, help="Video FPS (required if using --time-col)")
    p.add_argument("--obs-cols", nargs="*", default=[], help="Optional CSV columns to include as observation_space (floats)")
    p.add_argument("--action-labels", nargs="*", default=None, help='Optional list like: 0:No-Op 1:Move 2:Attack (space-separated "id:label")')
    p.add_argument("--max-width", type=int, default=None, help="Optional: downscale extracted frames to this max width (preserves aspect ratio)")
    p.add_argument("--max-height", type=int, default=None, help="Optional: downscale extracted frames to this max height (preserves aspect ratio)")
    p.add_argument("--image-format", choices=["png", "jpg", "jpeg"], default="jpeg", help="Image format for embedded frames (default: jpeg)")
    p.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality 1-100 when --image-format is jpg/jpeg (default: 85)")
    p.add_argument("--stretch-to-video", action="store_true", default=True, help="Linearly map rows across video duration (default: enabled)")
    p.add_argument("--per-unit-tags", action="store_true", default=True, help="Use per-unit tag columns (default: enabled)")
    args = p.parse_args()
    
    # Auto-construct paths from episode name if not provided
    if args.episode:
        if args.csv is None:
            args.csv = f"Unified CSVs/{args.episode}.csv"
        if args.video is None:
            args.video = f"Replay Files/{args.episode}.mov"
        if args.out is None:
            args.out = f"JSON Ingest Files/{args.episode}_FINAL.json"
    
    # Validate required paths
    if not args.csv or not args.video or not args.out:
        print("ERROR: Either provide 'episode' name or specify --csv, --video, and --out explicitly.", file=sys.stderr)
        sys.exit(2)

    # Validate frame/time options unless using --row-as-frame
    if not args.row_as_frame:
        if (args.frame_col is None) == (args.time_col is None):
            print("ERROR: specify exactly one of --frame-col or --time-col (with --fps), or use --row-as-frame.", file=sys.stderr)
            sys.exit(2)
        if args.time_col is not None and (args.fps is None or args.fps <= 0):
            print("ERROR: --fps must be provided and > 0 when using --time-col.", file=sys.stderr)
            sys.exit(2)

    # Load data
    df = pd.read_csv(args.csv)

    # Detect per-unit tag columns if --per-unit-tags enabled
    per_unit_tags = None
    enemy_unit_tags = None
    if args.per_unit_tags:
        cols = df.columns.tolist()
        action_tags = sorted([c for c in cols if c.startswith('action_tag_')])
        hp_tags = sorted([c for c in cols if c.startswith('friend_hp_tag_')])
        nav_rew_tags = sorted([c for c in cols if c.startswith('nav_rew_tag_')])
        combat_rew_tags = sorted([c for c in cols if c.startswith('combat_rew_tag_')])
        enemy_hp_tags = sorted([c for c in cols if c.startswith('enemy_hp_tag_')])

        if not action_tags or not hp_tags:
            print("ERROR: --per-unit-tags specified but no action_tag_* or friend_hp_tag_* columns found", file=sys.stderr)
            sys.exit(2)

        # Extract tag IDs from column names
        tag_ids = [c.replace('action_tag_', '') for c in action_tags]
        per_unit_tags = {
            'tag_ids': tag_ids,
            'action_cols': action_tags,
            'hp_cols': hp_tags,
            'nav_rew_cols': nav_rew_tags if nav_rew_tags else None,
            'combat_rew_cols': combat_rew_tags if combat_rew_tags else None
        }
        if enemy_hp_tags:
            enemy_tag_ids = [c.replace('enemy_hp_tag_', '') for c in enemy_hp_tags]
            enemy_unit_tags = {
                'tag_ids': enemy_tag_ids,
                'hp_cols': enemy_hp_tags
            }
        print(f"Detected {len(tag_ids)} per-unit tags: {tag_ids}")
        if enemy_unit_tags:
            print(f"Detected {len(enemy_unit_tags['tag_ids'])} enemy per-unit tags: {enemy_unit_tags['tag_ids']}")

    # Prepare video
    if cv2 is None:
        print("ERROR: OpenCV (cv2) is required. Install with: pip install opencv-python", file=sys.stderr)
        sys.exit(2)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: failed to open video: {args.video}", file=sys.stderr)
        sys.exit(2)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"Video opened: {args.video}, {total_frames} frames at {video_fps:.2f} FPS")

    # Action remap
    action_map = None
    if args.action_map:
        try:
            action_map = json.loads(args.action_map)
        except Exception as e:
            print(f"ERROR: --action-map must be valid JSON (e.g. '{{\"0\":1,\"1\":2}}'): {e}", file=sys.stderr)
            sys.exit(2)

    # Action labels → action_mapping (optional)
    action_mapping = None
    if args.action_labels:
        action_mapping = {}
        for tok in args.action_labels:
            if ":" not in tok:
                print(f"WARNING: ignoring malformed action-label '{tok}', expected 'id:Label'")
                continue
            k, v = tok.split(":", 1)
            k = k.strip()
            v = v.strip()
            # must be int-convertible key and string label
            int(k)  # validate
            action_mapping[k] = v
        if not action_mapping:
            action_mapping = None

    # Collect steps
    steps = []
    # Track which units are alive across the entire episode (stateful death tracking)
    num_units = 5
    alive_units = set(range(1, num_units + 1))  # Start with all units alive: {1,2,3,4,5}
    
    for idx, row in df.iterrows():
        # Step index
        number = int(idx)

        # Determine frame index for this row
        if args.row_as_frame:
            # Map row index to frame index
            if args.stretch_to_video and total_frames > 0 and len(df) > 1:
                # Distribute rows across full video length [0, total_frames-1]
                frame_idx = int(round(number * (total_frames - 1) / float(len(df) - 1)))
            else:
                # Use the dataframe row index directly as the frame index
                frame_idx = number
        elif args.frame_col:
            if args.frame_col not in row:
                print(f"ERROR: frame column '{args.frame_col}' not found in CSV", file=sys.stderr)
                sys.exit(2)
            frame_idx = int(row[args.frame_col])
        else:
            # time-based
            if args.time_col not in row:
                print(f"ERROR: time column '{args.time_col}' not found in CSV", file=sys.stderr)
                sys.exit(2)
            t = float(row[args.time_col])
            frame_idx = int(round(t * args.fps))

        # Clamp to valid range
        if frame_idx < 0:
            frame_idx = 0
        if frame_idx >= total_frames:
            frame_idx = total_frames - 1 if total_frames > 0 else 0

        # Grab frame, optionally resize, and encode
        frame = read_frame(cap, frame_idx)
        if frame is None:
            print(f"WARNING: could not read frame {frame_idx} at row {idx}; image omitted")
            image_b64 = None
        else:
            # downscale if requested to reduce JSON size / ingest memory
            frame = maybe_resize(frame, args.max_width, args.max_height)
            image_b64 = b64_image_from_frame(frame, args.image_format, args.jpeg_quality)

        # Assemble rewards
        rewards_vec = []
        for col in args.rewards:
            if col not in row:
                print(f"ERROR: reward column '{col}' not found in CSV", file=sys.stderr)
                sys.exit(2)
            val = row[col]
            try:
                rewards_vec.append(float(val))
            except Exception:
                print(f"ERROR: reward column '{col}' contains non-numeric value '{val}'", file=sys.stderr)
                sys.exit(2)

        # Action
        if args.action_col not in row:
            print(f"ERROR: action column '{args.action_col}' not found in CSV", file=sys.stderr)
            sys.exit(2)
        action_id = parse_action(row[args.action_col], action_map)

        # Observation (optional)
        obs = collect_observation(row, args.obs_cols)

        agent_step = {
            "agent": "SC2Agent",
            "action": action_id,
            "rewards": rewards_vec,
            "observation_space": obs if obs is not None else [0.0] * 64  # 64-element dummy if no obs_cols
        }
        # --- Map CSV columns into additional telemetry fields used by the UI ---
        # value estimate
        if "value_estimate" in row:
            v = to_float_or_none(row["value_estimate"])
            if v is not None:
                agent_step["value_estimate"] = v

        # health (friendly)
        for col in ("friend_hp", "health"):
            if col in row:
                v = to_float_or_none(row[col])
                if v is not None:
                    agent_step["health"] = v
                break

        # enemy agent health (wrap single value into list)
        if "enemy_hp" in row:
            v = to_float_or_none(row["enemy_hp"])
            if v is not None:
                agent_step["enemy_agent_health"] = [v]

        # enemy unit health - build from enemy_hp if not present as separate column
        enemy_unit_health = []
        if "enemy_unit_health" in row and not pd.isna(row["enemy_unit_health"]):
            val = row["enemy_unit_health"]
            for x in str(val).split(","):
                x = x.strip()
                if not x:
                    continue
                fx = to_float_or_none(x)
                if fx is not None:
                    enemy_unit_health.append(fx)
        elif "enemy_hp" in row:
            # If no enemy_unit_health column, create synthetic unit healths from total enemy_hp
            # Assume 5 enemy units like in episode_1.json
            v = to_float_or_none(row["enemy_hp"])
            if v is not None:
                # Distribute health among 5 units (45.0 each if total is 225.0)
                unit_hp = v / 5.0
                enemy_unit_health = [unit_hp] * 5
        
        if enemy_unit_health:
            agent_step["enemy_unit_health"] = enemy_unit_health

        # custom metrics mapping
        custom = {}
        mapping_pairs = [
            ("nav_dist", "navigation_distance"),
            ("combat_dist", "combat_distance"),
            ("A_direction", "move_direction"),
            ("A_enemy_idx", "attacked_enemy_idx"),
            ("agent_r_hat", "predicted_rewards"),
            ("td_error", "td_error"),
            ("env_reward", "env_reward")
        ]
        for cname, cjson in mapping_pairs:
            if cname in row and not pd.isna(row[cname]) and str(row[cname]) != "":
                v = to_float_or_none(row[cname])
                if v is not None:
                    custom[cjson] = v

        # unit counts / one-hot style counts
        count_pairs = [
            ("a_move", "number_of_units_move"),
            ("a_attack", "number_of_units_attack"),
            ("a_noop", "number_of_units_noop"),
            ("a_alive", "number_of_units_alive"),
            ("A_selected", "number_of_units_selected"),
        ]
        for cname, cjson in count_pairs:
            if cname in row and not pd.isna(row[cname]) and str(row[cname]) != "":
                v = to_float_or_none(row[cname])
                if v is not None:
                    custom[cjson] = v

        if custom:
            agent_step["custom_metrics"] = custom

        # Build unit_steps from per-unit tag data OR aggregate counts
        unit_steps = []
        
        if per_unit_tags:
            # Use actual per-unit data from tag columns
            # Check for the step-0 edge case where per-unit HPs are all zero but total friend_hp > 0
            hp_vals = []
            for i in range(len(per_unit_tags['hp_cols'])):
                hp_vals.append(to_float_or_none(row.get(per_unit_tags['hp_cols'][i], 0)) or 0.0)
            all_unit_hp_zero = all((v == 0.0 for v in hp_vals))
            total_friend_hp = to_float_or_none(row.get("friend_hp") or row.get("health", 0)) or 0.0
            distribute_even_hp = (all_unit_hp_zero and total_friend_hp > 0.0)

            for i, tag_id in enumerate(per_unit_tags['tag_ids']):
                action_col = per_unit_tags['action_cols'][i]
                hp_col = per_unit_tags['hp_cols'][i]
                
                # Get action and health for this specific unit
                unit_action = int(to_float_or_none(row.get(action_col, 0)) or 0)
                unit_hp = to_float_or_none(row.get(hp_col, 0)) or 0.0
                if distribute_even_hp:
                    # Distribute the total friendly HP evenly across units for this step
                    unit_hp = total_friend_hp / float(len(per_unit_tags['hp_cols']))
                
                # Per-unit rewards if available
                unit_rewards = list(rewards_vec)  # Copy agent rewards
                if per_unit_tags['nav_rew_cols'] and per_unit_tags['combat_rew_cols']:
                    nav_rew = to_float_or_none(row.get(per_unit_tags['nav_rew_cols'][i], 0)) or 0.0
                    combat_rew = to_float_or_none(row.get(per_unit_tags['combat_rew_cols'][i], 0)) or 0.0
                    # Update nav and combat rewards (assuming order: nav, combat, term, ...)
                    if len(unit_rewards) >= 2:
                        unit_rewards[0] = nav_rew
                        unit_rewards[1] = combat_rew
                
                # Determine if unit is dead (HP = 0)
                is_dead = (unit_hp == 0.0)
                
                unit_step = {
                    "unit": f"marine_{i+1}",
                    "custom_metrics": {"tag_id": tag_id},
                    "rewards": unit_rewards,
                    "health": unit_hp,
                    "action": -1 if is_dead else unit_action  # -1 = Dead
                }
                unit_steps.append(unit_step)
        else:
            # FALLBACK: Synthetic unit steps from aggregate counts with stateful death tracking
            # Get current alive count from CSV
            n_move = int(to_float_or_none(row.get("a_move", 0)) or 0)
            n_attack = int(to_float_or_none(row.get("a_attack", 0)) or 0)
            n_noop = int(to_float_or_none(row.get("a_noop", 0)) or 0)
            n_alive_now = int(to_float_or_none(row.get("a_alive", num_units)) or num_units)
            
            # Track deaths: if fewer alive now than we have tracked, kill units
            if n_alive_now < len(alive_units):
                # Kill units from highest ID down (marine_5 dies first, then marine_4, etc.)
                units_to_kill = len(alive_units) - n_alive_now
                for _ in range(units_to_kill):
                    if alive_units:
                        dead_unit = max(alive_units)  # Kill highest numbered unit
                        alive_units.remove(dead_unit)
            
            # Build action list for currently alive units
            unit_actions = []
            for _ in range(n_move):
                unit_actions.append(1)  # Move
            for _ in range(n_attack):
                unit_actions.append(2)  # Attack
            for _ in range(n_noop):
                unit_actions.append(0)  # No-Op
            
            # Pad or truncate to match alive count
            while len(unit_actions) < len(alive_units):
                unit_actions.append(0)
            unit_actions = unit_actions[:len(alive_units)]
            
            # Calculate health for living units
            total_hp = to_float_or_none(row.get("friend_hp") or row.get("health", 0)) or 0
            unit_health = total_hp / len(alive_units) if (total_hp > 0 and len(alive_units) > 0) else 0.0
            
            # Create unit steps for ALL units (alive and dead)
            alive_list = sorted(alive_units)
            for unit_id in range(1, num_units + 1):
                if unit_id in alive_units:
                    # Alive unit: assign action from list
                    idx_in_alive = alive_list.index(unit_id)
                    action = unit_actions[idx_in_alive] if idx_in_alive < len(unit_actions) else 0
                    unit_step = {
                        "unit": f"marine_{unit_id}",
                        "custom_metrics": {},
                        "rewards": rewards_vec,
                        "health": unit_health,
                        "action": action
                    }
                else:
                    # Dead unit: action -1, health 0
                    unit_step = {
                        "unit": f"marine_{unit_id}",
                        "custom_metrics": {},
                        "rewards": rewards_vec,
                        "health": 0.0,
                        "action": -1
                    }
                unit_steps.append(unit_step)
        
        if unit_steps:
            agent_step["unit_steps"] = unit_steps

        step_obj = {"number": number, "agent_steps": [agent_step]}
        
        # Populate enemy_unit_health and enemy_unit_steps from enemy_hp_tag_* when available
        if enemy_unit_tags:
            enemy_unit_steps = []
            enemy_hp_vals = []
            
            for i, tag_id in enumerate(enemy_unit_tags['tag_ids']):
                hp_col = enemy_unit_tags['hp_cols'][i]
                unit_hp = to_float_or_none(row.get(hp_col, 0)) or 0.0
                enemy_hp_vals.append(unit_hp)
                
                # Create individual enemy unit object with tag_id (same structure as friendly units)
                enemy_unit_step = {
                    "unit": f"enemy_{i+1}",
                    "health": unit_hp,
                    "custom_metrics": {"tag_id": tag_id}
                }
                enemy_unit_steps.append(enemy_unit_step)
            
            # Only include if any non-zero to avoid conflicting with total at step-0
            if any(v != 0.0 for v in enemy_hp_vals):
                agent_step["enemy_unit_health"] = enemy_hp_vals
                agent_step["enemy_unit_steps"] = enemy_unit_steps
            else:
                # If all zeros but total enemy_hp > 0, distribute evenly (step-0 symmetry)
                total_enemy_hp = to_float_or_none(row.get("enemy_hp", 0)) or 0.0
                if total_enemy_hp > 0.0 and len(enemy_hp_vals) > 0:
                    per_enemy = total_enemy_hp / float(len(enemy_hp_vals))
                    agent_step["enemy_unit_health"] = [per_enemy] * len(enemy_hp_vals)
                    # Update enemy_unit_steps with distributed health
                    for j in range(len(enemy_unit_steps)):
                        enemy_unit_steps[j]["health"] = per_enemy
                    agent_step["enemy_unit_steps"] = enemy_unit_steps
        
        if image_b64 is not None:
            step_obj["image"] = image_b64

        steps.append(step_obj)

    cap.release()

    # Canonicalize reward names so UI can find 'navigation' and 'combat' regardless of CSV column naming
    def _canonical_reward_name(name: str) -> str:
        k = str(name).strip().lower()
        mapping = {
            'nav_r': 'navigation',
            'nav_reward': 'navigation',
            'navigation': 'navigation',
            'combat_r': 'combat',
            'combat_reward': 'combat',
            'combat': 'combat',
            'term_r': 'termination',
            'term': 'termination',
            'terminal': 'termination',
            'termination': 'termination',
        }
        return mapping.get(k, name)

    reward_mapping_out = [_canonical_reward_name(n) for n in args.rewards]

    # Build document
    doc = {
        "training": {
            "environment": args.env,
            "algorithm": args.algo,
            "parallel": False,
            "iterations": int(args.iterations),
            "config": {},
            "reward_mapping": reward_mapping_out
        },
        "inference": {
            "parallel": False,
            "config": {},
            "steps": steps
        }
    }
    
    # Always include action_mapping (required by UI)
    if action_mapping:
        doc["action_mapping"] = action_mapping
    else:
        # Default action mapping
        doc["action_mapping"] = {
            "-1": "Dead",
            "0": "No-Op",
            "1": "Move",
            "2": "Attack",
            "unit_mapping": {
                "-1": "Dead",
                "0": "No-Op",
                "1": "Move",
                "2": "Attack"
            }
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"Wrote {args.out} with {len(steps)} steps.")
    print("Ready to ingest with: docker compose run --rm -v \"$PWD:/data\" django ./manage.py ingest_episode /data/" + args.out)
    
if __name__ == "__main__":
    main()