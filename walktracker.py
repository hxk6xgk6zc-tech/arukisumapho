import cv2
import numpy as np
import pandas as pd
import json
from ultralytics import YOLO
from collections import defaultdict
import os
import torch
#!pip install ultralytics opencv-python pandas numpy
class CrowdAnalyzer:
    def __init__(self, model_path='yolov8x.pt'):
        """
        初期設定
        """
        # --- 1. デバイス設定 (Mac MPS / CPU) ---
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print("Using Apple MPS (Metal Performance Shaders) acceleration.")
        else:
            self.device = 'cpu'
            print("MPS not found. Using CPU.")

        # --- 2. モデルロード ---
        # CoreML (.mlpackage) があれば自動でそれを使います
        if model_path.endswith('.mlpackage'):
            print(f"Loading CoreML model: {model_path}")
        else:
            print(f"Loading PyTorch model: {model_path}")
        
        self.model = YOLO(model_path)

        # --- 3. スマホ検知（発光）の閾値設定 [調整可能変数] ---
        # 夜間の映像で「顔周りがどれくらい光っていたらスマホとみなすか」
        self.glow_brightness_threshold = 220  # 最大輝度 (0-255): これ以上明るい点があれば怪しい
        self.glow_contrast_threshold = 15     # 標準偏差: 背景が暗くて一点だけ明るい状態を検知
        
        # --- 4. 座標変換 (Homography) 設定 [現場に合わせて修正必須] ---
        # src: 動画上の4点 (ピクセル)
        # dst: 現実世界の4点 (メートル)
        # ※以下はダミー値です。必ず実際の動画に合わせて書き換えてください。
        self.src_points = np.array([
            [100, 200], [1820, 200], [1820, 980], [100, 980]
        ], dtype=np.float32)
        
        self.dst_points = np.array([
            [0, 0], [20, 0], [20, 15], [0, 15]
        ], dtype=np.float32)
        
        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

        # データ保持用
        self.track_history = defaultdict(lambda: {
            'start_time': None,
            'end_time': None,
            'coords': [],      # {t, x, y, is_phone}
            'phone_count': 0,  # スマホ検知されたフレーム数
            'total_frames': 0  # 追跡された総フレーム数
        })

    def transform_point(self, u, v):
        """画像座標(u, v) -> 地図座標(x, y)"""
        point = np.array([[[u, v]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.matrix)
        return transformed[0][0]

    def detect_smartphone_glow(self, frame, x1, y1, x2, y2):
        """
        バウンディングボックスの上半身エリアの輝度を解析し、
        スマホの照り返しかどうかを判定する
        """
        # 画像範囲チェック
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
        
        # 人物全体切り出し
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0: return False

        # 上半身（上1/3）に限定
        height = y2 - y1
        upper_body_roi = person_roi[0:int(height/3), :]
        if upper_body_roi.size == 0: return False

        # グレースケール変換
        gray = cv2.cvtColor(upper_body_roi, cv2.COLOR_BGR2GRAY)
        
        # 統計量計算
        max_val = np.max(gray)  # 最も明るいピクセル
        std_dev = np.std(gray)  # 明暗のばらつき（コントラスト）

        # 判定ロジック
        # 「最大輝度が閾値を超えている」かつ「全体がのっぺり明るいわけではない（コントラストがある）」
        if max_val > self.glow_brightness_threshold and std_dev > self.glow_contrast_threshold:
            return True
        
        return False

    def process_video(self, video_path, output_csv_path):
        print(f"\nProcessing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            timestamp = frame_idx / fps
            
            # --- 前処理: ガンマ補正（暗いシーンを見やすくするオプション） ---
            # 必要に応じてコメントアウトを外してください
            # gamma = 1.2
            # invGamma = 1.0 / gamma
            # table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
            # frame = cv2.LUT(frame, table)

            # --- YOLO推論 ---
            # 輝度解析のため、モザイク前の「生の画像」で推論・判定を行います
            results = self.model.track(
                frame, 
                persist=True, 
                tracker="bytetrack.yaml", 
                classes=[0], # 人物のみ
                device=self.device,
                verbose=False
            )
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    
                    # 1. スマホ判定 (Glow Detection)
                    is_using_phone = self.detect_smartphone_glow(frame, x1, y1, x2, y2)
                    
                    # 2. 座標変換 (足元)
                    foot_u = (x1 + x2) / 2
                    foot_v = y2 
                    real_x, real_y = self.transform_point(foot_u, foot_v)
                    
                    # 3. データ記録
                    tracker = self.track_history[track_id]
                    if tracker['start_time'] is None:
                        tracker['start_time'] = timestamp
                    tracker['end_time'] = timestamp
                    
                    tracker['total_frames'] += 1
                    if is_using_phone:
                        tracker['phone_count'] += 1

                    # 時系列データとして保存
                    tracker['coords'].append({
                        "t": round(timestamp, 3),
                        "x": round(float(real_x), 3),
                        "y": round(float(real_y), 3),
                        "p": 1 if is_using_phone else 0 # p=Phone flag
                    })

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")

        cap.release()
        
        # CSV出力
        self.export_csv(video_path, output_csv_path)

    def export_csv(self, video_path, output_path):
        data_rows = []
        video_name = os.path.basename(video_path)
        
        for track_id, data in self.track_history.items():
            # ノイズ除去（極端に短い軌跡は無視）
            if data['total_frames'] < 10: 
                continue

            # スマホ使用率（全フレームのうち、何割で光っていたか）
            phone_usage_ratio = data['phone_count'] / data['total_frames']

            row = {
                "ID": track_id,
                "Video_Name": video_name,
                "Start_Time": data['start_time'],
                "End_Time": data['end_time'],
                "Duration": round(data['end_time'] - data['start_time'], 2),
                "Phone_Usage_Ratio": round(phone_usage_ratio, 3), # 0.0 ~ 1.0
                # JSON形式で時系列データを格納
                "Trajectory_JSON": json.dumps(data['coords'])
            }
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        # カラム順序の整理
        cols = ["ID", "Video_Name", "Start_Time", "End_Time", "Duration", "Phone_Usage_Ratio", "Trajectory_JSON"]
        df = df[cols]
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Data saved to {output_path}")
        print(f"Total IDs processed: {len(df)}")

# --- メイン実行ブロック ---
if __name__ == "__main__":
    # 1. 解析対象の動画パス
    target_video = "night_sample.mp4" 
    output_csv = "night_analysis_result.csv"

    # 2. CoreMLモデルへの自動変換（初回のみ有効にしてください）
    # M1/M2/M3の場合は True 推奨
    use_coreml = True 
    base_model = 'yolov8x' # yolov8x, yolov8l, yolov8m から選択
    
    final_model_path = f"{base_model}.pt"
    if use_coreml:
        ml_path = f"{base_model}.mlpackage"
        if os.path.exists(ml_path):
            final_model_path = ml_path
        else:
            print("Exporting to CoreML for Apple Silicon optimization...")
            model = YOLO(f"{base_model}.pt")
            model.export(format='coreml')
            final_model_path = ml_path

    # 3. 実行
    if os.path.exists(target_video):
        analyzer = CrowdAnalyzer(model_path=final_model_path)
        
        # ★ここで閾値を微調整できます
        analyzer.glow_brightness_threshold = 200 # 少し感度を高める場合
        analyzer.glow_contrast_threshold = 15
        
        analyzer.process_video(target_video, output_csv)
    else:
        print(f"Video file not found: {target_video}")
        print("Please place a video file or update the path.")
