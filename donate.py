import json
from typing import Dict, List
from ollama import Client
import asyncio

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
            # pull 不需要 await
            self.client.pull(model=self.model_name)
            return {"status": "success", "message": "Model ready"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def query(self, user_input: str) -> Dict:
        """處理用戶查詢"""
        try:
            # chat 也不需要 await
            messages = [
                {
                    "role": "system",
                    "content": "你是一個捐贈諮詢助手，請根據知識庫回答問題。"
                },
                {
                    "role": "user",
                    "content": f"""
基於以下知識：
{json.dumps(self.knowledge_base, ensure_ascii=False)}

問題：{user_input}
                    """
                }
            ]

            response = self.client.chat(
                model=self.model_name,
                messages=messages
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

    # 確保模型已下載
    print("正在確認模型...")
    result = assistant.ensure_model()
    print(result)

    if result["status"] == "success":
        # 測試查詢
        query = "我想捐贈罐頭，請問哪裡可以捐？"
        print(f"\n問題: {query}")
        result = assistant.query(query)
        print("回答:", result)

if __name__ == "__main__":
    main()