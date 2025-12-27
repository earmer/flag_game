import numpy as np

class CTFMatrixConverter:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        # ID 映射表
        self.ID = {
            "PLAYER": [0, 1, 2],
            "PLAYER_WITH_FLAG": [3, 4, 5],
            "OPPO_PLAYER": [6, 7, 8],
            "OPPO_PLAYER_WITH_FLAG": [9, 10, 11],
            "PRISON": 12,
            "HOME": 13,
            "HOME_WITH_FLAG": 14,
            "OPPO_HOME": 15,
            "BARRIER": 16,
            "BLANK": 17,
            "FLAG": 18,
            "OPPO_FLAG": 19
        }
        self.static_matrix = None

    def initialize_static_map(self, init_req):
        """处理初始化的地图信息（墙、障碍、区域）"""
        # 初始化为全空白
        self.static_matrix = np.full((self.height, self.width), self.ID["BLANK"], dtype=int)
        
        # 填充墙和障碍 (ID 13)
        for sub in ["walls", "obstacles"]:
            for item in init_req["map"].get(sub, []):
                self.static_matrix[item["y"]][item["x"]] = self.ID["BARRIER"]
        
        # 填充己方监狱 (ID 09)
        for item in init_req.get("myteamPrison", []):
            self.static_matrix[item["y"]][item["x"]] = self.ID["PRISON"]
            
        # 填充己方 Home (ID 10)
        for item in init_req.get("myteamTarget", []):
            self.static_matrix[item["y"]][item["x"]] = self.ID["HOME"]
            
        # 填充对方 Home (ID 12)
        for item in init_req.get("opponentTarget", []):
            self.static_matrix[item["y"]][item["x"]] = self.ID["OPPO_HOME"]

    def convert_to_matrix(self, status_req):
        """将实时状态叠加到地图上"""
        if self.static_matrix is None:
            return None
        
        # 复制静态地图
        matrix = self.static_matrix.copy()
        
        # 1. 处理我方旗帜（判断 Home 是否已有旗）
        for f in status_req.get("myteamFlag", []):
            x, y = f["posX"], f["posY"]
            if not f["canPickup"]: # 已背回 Home 的旗子
                matrix[y][x] = self.ID["HOME_WITH_FLAG"]
            else: # 还在原位或被掉落的我方旗子
                matrix[y][x] = self.ID["FLAG"]
        
        # 2. 处理对方旗帜 (ID 16)
        for f in status_req.get("opponentFlag", []):
            if f["canPickup"]:
                matrix[f["posY"]][f["posX"]] = self.ID["OPPO_FLAG"]
        
        # 3. 处理对方球员 (ID 06-08 或 09-11)
        for i, p in enumerate(status_req.get("opponentPlayer", [])):
            if i < 3:
                id_list = self.ID["OPPO_PLAYER_WITH_FLAG"] if p["hasFlag"] else self.ID["OPPO_PLAYER"]
                matrix[p["posY"]][p["posX"]] = id_list[i]

        # 4. 处理我方球员 (ID 00-02 或 03-05) - 优先级最高，覆盖在地图最上方
        for i, p in enumerate(status_req.get("myteamPlayer", [])):
            if i < 3:
                id_list = self.ID["PLAYER_WITH_FLAG"] if p["hasFlag"] else self.ID["PLAYER"]
                matrix[p["posY"]][p["posX"]] = id_list[i]
                
        return matrix

    def print_matrix(self, matrix):
        """漂亮的打印输出"""
        if matrix is None: return
        print("\n" + "="*40)
        print(f"Current Game Matrix ({self.width}x{self.height}):")
        for row in matrix:
            # 使用 zfill 保证对齐
            print(" ".join(str(cell).zfill(2) for cell in row))
        print("="*40)

# 测试驱动 (可以直接运行 python matrix_utils.py 测试)
if __name__ == "__main__":
    import json
    import os
    
    # 模拟读取样例文件
    backend_path = "c:/Users/Earmer/flag_game/CTF/backend/"
    with open(os.path.join(backend_path, "example_init.json")) as f:
        init_data = json.load(f)
    with open(os.path.join(backend_path, "example_plan_next_actions.json")) as f:
        status_data = json.load(f)
        
    converter = CTFMatrixConverter(width=20, height=20)
    converter.initialize_static_map(init_data)
    current_matrix = converter.convert_to_matrix(status_data)
    converter.print_matrix(current_matrix)