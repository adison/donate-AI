# -*- coding: utf-8 -*-
import json
from typing import Dict, List
from ollama import Client
import sys
import time

class DonationAssistant:
    def __init__(self, model_name: str = "llama2"):
        print("初始化 DonationAssistant...")
        self.client = Client()
        self.model_name = model_name
        self.knowledge_base = self._load_knowledge_base()
        print("初始化完成")

    def _load_knowledge_base(self) -> Dict:
        """載入知識庫"""
        print("載入知識庫...")
        try:
            with open('knowledge_base.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                print("知識庫載入成功")
                return data
        except FileNotFoundError:
            print("找不到知識庫檔案，建立新的知識庫")
            return {"locations": [], "rules": [], "items": []}

    def ensure_model(self):
        """確保模型已下載"""
        print(f"檢查模型 {self.model_name} ...")
        try:
            print("開始下載模型...")
            self.client.pull(model=self.model_name)
            print("模型下載完成")
            return {"status": "success", "message": "模型準備完成"}
        except Exception as e:
            print(f"模型下載失敗: {str(e)}")
            return {"status": "error", "message": str(e)}

    def query(self, user_input: str) -> Dict:
        """處理用戶查詢"""
        print(f"處理查詢: {user_input}")
        try:
            prompt = f"""
你是一個捐贈諮詢助手。請用繁體中文回答。

知識庫資訊：
{json.dumps(self.knowledge_base, ensure_ascii=False, indent=2)}

用戶問題：{user_input}
"""
            print("發送請求到 Ollama...")
            response = self.client.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            print("收到 Ollama 回應")

            return {
                "status": "success",
                "response": response['message']['content']
            }

        except Exception as e:
            print(f"查詢發生錯誤: {str(e)}")
            return {
                "status": "error",
                "message": f"查詢失敗: {str(e)}"
            }

def save_knowledge_base(data: Dict):
    """儲存知識庫"""
    print("儲存知識庫...")
    with open('knowledge_base.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("知識庫儲存完成")

def main():
    print("程式開始執行...")

    # 初始化助手
    assistant = DonationAssistant()

    # 準備範例資料
    print("準備範例資料...")
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

    # 儲存範例資料
    save_knowledge_base(sample_data)

    # 確保模型已下載
    print("檢查模型狀態...")
    result = assistant.ensure_model()
    print("模型狀態:", result)

    if result["status"] == "success":
        # 測試查詢
        query = "我想捐贈罐頭，請問哪裡可以捐？"
        print(f"\n提出問題: {query}")
        result = assistant.query(query)
        print("回答:", result.get("response", "沒有回應"))
    else:
        print("模型準備失敗，無法進行查詢")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程式執行發生錯誤: {str(e)}")
    finally:
        print("程式執行結束")
        # 保持視窗開啟（如果在 Windows 上直接執行）
        input("按 Enter 鍵結束程式...")