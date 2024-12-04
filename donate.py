# -*- coding: utf-8 -*-
import json
from typing import Dict, List
from ollama import Client
import sys
from datetime import datetime
import os

class Logger:
    def __init__(self):
        self.log_dir = "log"
        self.ensure_log_directory()
        self.log_file = self.get_log_file()

    def ensure_log_directory(self):
        """確保log目錄存在"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def get_log_file(self) -> str:
        """獲取今天的log檔案路徑"""
        today = datetime.now().strftime('%Y-%m-%d')
        return os.path.join(self.log_dir, f'{today}.log')

    def log(self, message: str):
        """記錄日誌"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}\n"

        # 輸出到控制台
        print(log_message.strip())

        # 寫入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message)

class DonationAssistant:
    def __init__(self, model_name: str = "llama2"):
        self.logger = Logger()
        self.logger.log("初始化 DonationAssistant...")
        self.client = Client()
        self.model_name = model_name
        self.knowledge_base = self._load_knowledge_base()
        self.logger.log("初始化完成")

    def _load_knowledge_base(self) -> Dict:
        """從kb目錄載入知識庫"""
        self.logger.log("載入知識庫...")
        kb_dir = "kb"
        kb_file = os.path.join(kb_dir, "database.json")

        try:
            # 確保目錄存在
            if not os.path.exists(kb_dir):
                os.makedirs(kb_dir)
                self.logger.log(f"創建知識庫目錄: {kb_dir}")

            # 讀取知識庫文件
            if os.path.exists(kb_file):
                with open(kb_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.logger.log("知識庫載入成功")
                return data
            else:
                self.logger.log("找不到知識庫檔案，建立新的知識庫")
                initial_data = {"locations": [], "rules": [], "items": []}
                with open(kb_file, 'w', encoding='utf-8') as f:
                    json.dump(initial_data, f, ensure_ascii=False, indent=2)
                return initial_data

        except Exception as e:
            self.logger.log(f"載入知識庫時發生錯誤: {str(e)}")
            return {"locations": [], "rules": [], "items": []}

    def ensure_model(self):
        """確保模型已下載"""
        self.logger.log(f"檢查模型 {self.model_name}")
        try:
            self.logger.log("開始下載模型...")
            self.client.pull(model=self.model_name)
            self.logger.log("模型下載完成")
            return {"status": "success", "message": "模型準備完成"}
        except Exception as e:
            error_msg = f"模型下載失敗: {str(e)}"
            self.logger.log(error_msg)
            return {"status": "error", "message": error_msg}

    def query(self, user_input: str) -> Dict:
        """處理用戶查詢"""
        self.logger.log(f"處理查詢: {user_input}")
        try:
            system_prompt = """你是一個繁體中文的捐贈諮詢助手。
請嚴格遵守以下規則：
1. 只使用繁體中文回答
2. 不要使用英文
3. 保持簡潔明瞭的回答方式
4. 確保回答包含具體的地址和聯絡方式
5. 回答格式應該是：
   - 捐贈地點：[名稱]
   - 地址：[地址]
   - 可接受物品：[列表]
   - 營業時間：[時間]
   - 聯絡方式：[電話]
   - 注意事項：[說明]"""

            user_prompt = f"""
知識庫資訊：
{json.dumps(self.knowledge_base, ensure_ascii=False, indent=2)}

用戶問題：{user_input}

請用繁體中文回答上述問題。
"""
            self.logger.log("發送請求到 Ollama...")
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            self.logger.log("收到 Ollama 回應")

            response_content = response['message']['content']
            self.logger.log(f"回應內容: {response_content}")

            return {
                "status": "success",
                "response": response_content
            }

        except Exception as e:
            error_msg = f"查詢發生錯誤: {str(e)}"
            self.logger.log(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }

def main():
    logger = Logger()
    logger.log("程式開始執行")

    try:
        # 初始化助手
        assistant = DonationAssistant()

        # 確保模型已下載
        logger.log("檢查模型狀態...")
        result = assistant.ensure_model()
        logger.log(f"模型狀態: {result}")

        if result["status"] == "success":
            # 測試查詢
            query = "我想捐贈罐頭，請問哪裡可以捐？"
            logger.log(f"提出問題: {query}")
            result = assistant.query(query)
            logger.log(f"回答: {result.get('response', '沒有回應')}")
        else:
            logger.log("模型準備失敗，無法進行查詢")

    except Exception as e:
        logger.log(f"程式執行發生錯誤: {str(e)}")
    finally:
        logger.log("程式執行結束")
        input("按 Enter 鍵結束程式...")

if __name__ == "__main__":
    main()