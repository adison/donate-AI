# -*- coding: utf-8 -*-
import json
from typing import Dict, List
from ollama import Client
import sys

# 設定系統編碼
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class DonationAssistant:
    def __init__(self, model_name: str = "llama2"):
        self.client = Client()
        self.model_name = model_name
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> Dict:
        """載入知識庫"""
        try:
            with open('knowledge_base.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"locations": [], "rules": [], "items": []}

    def ensure_model(self):
        """確保模型已下載"""
        try:
            self.client.pull(model=self.model_name)
            return {"status": "success", "message": "模型準備完成"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def query(self, user_input: str) -> Dict:
        """處理用戶查詢"""
        try:
            # 準備提示詞
            prompt = f"""
你是一個捐贈諮詢助手。請用繁體中文回答。

知識庫資訊：
{json.dumps(self.knowledge_base, ensure_ascii=False, indent=2)}

用戶問題：{user_input}
"""
            # 取得回應
            response = self.client.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            return {
                "status": "success",
                "response": response['message']['content']
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"查詢失敗: {str(e)}"
            }

def main():
    # 初始化助手
    assistant = DonationAssistant()

    # 更新知識庫示例
    sample_data = {
        "locations": [
            {
                "id": "loc_001",
                "name": "台北食物銀行",
                "address": "台北市信義區信義路5段150號",
                "accepted_items": ["乾糧", "罐頭"],
                "hours": "週一至週五 9:00-18:00",
                "contact": "02-1234-5678"
            }
        ]
    }

    # 確保模型已下載
    print("正在確認模型...")
    result = assistant.ensure_model()
    print(result)

    if result["status"] == "success":
        # 測試查詢
        query = "我想捐贈罐頭，請問哪裡可以捐？"
        print(f"\n問題: {query}")
        result = assistant.query(query)
        print("回答:", result["response"])

if __name__ == "__main__":
    main()