import json
from typing import Dict, List, Optional
from ollama import Client
import os

class DonationAISystem:
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

    def save_knowledge_base(self):
        """儲存知識庫"""
        with open('knowledge_base.json', 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)

    async def train_mode(self, training_data: List[Dict]):
        """訓練模式（使用 fine-tuning 方式）"""
        try:
            # 準備訓練數據
            formatted_data = self._format_training_data(training_data)

            # 將訓練數據轉換為 Ollama 可用的格式
            training_examples = self._prepare_training_examples(formatted_data)

            # 使用現有模型進行預測並記錄結果
            for example in training_examples:
                response = await self.client.chat(
                    model=self.model_name,
                    messages=[{
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": example["input"]
                    }]
                )

            # 更新知識庫
            self.knowledge_base.update(formatted_data)
            self.save_knowledge_base()

            return {"status": "success", "message": "Knowledge base updated successfully"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _format_training_data(self, data: List[Dict]) -> Dict:
        """格式化訓練數據"""
        formatted = {
            "locations": [],
            "rules": [],
            "items": []
        }

        for item in data:
            category = item.get("category")
            if category in formatted:
                formatted[category].append(item)

        return formatted

    def _prepare_training_examples(self, formatted_data: Dict) -> List[Dict]:
        """準備訓練範例"""
        examples = []

        # 為每個位置創建訓練範例
        for location in formatted_data.get("locations", []):
            example = {
                "input": f"這個地方接受什麼捐贈？地點：{location['name']}",
                "expected": f"在{location['name']}（地址：{location['address']}），可以接受以下捐贈物品：{', '.join(location['accepted_items'])}"
            }
            examples.append(example)

        return examples

    def _get_system_prompt(self) -> str:
        """獲取系統 prompt"""
        return """你是一個捐贈助手AI，負責協助人們找到合適的捐贈地點和方式。
        請根據提供的知識庫資訊，準確回答用戶的問題。
        如果不確定，請誠實說明。"""

    async def use_mode(self, query: str) -> Dict:
        """使用模式"""
        try:
            # 準備完整的對話內容
            messages = [
                {
                    "role": "system",
                    "content": self._get_system_prompt()
                },
                {
                    "role": "user",
                    "content": f"""
                    基於以下資訊回答問題：
                    {json.dumps(self.knowledge_base, ensure_ascii=False)}

                    用戶問題：{query}
                    """
                }
            ]

            # 獲取回應
            response = await self.client.chat(
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
                "message": str(e)
            }

# 使用範例
async def main():
    # 初始化系統
    ai_system = DonationAISystem()

    # 訓練模式示例
    training_data = [
        {
            "category": "locations",
            "name": "台北食物銀行",
            "address": "台北市信義區信義路5段150號",
            "accepted_items": ["乾糧", "罐頭"]
        }
    ]

    # 確保模型已下載
    try:
        print("Ensuring model is available...")
        await ai_system.client.pull(model=ai_system.model_name)
    except Exception as e:
        print(f"Error pulling model: {e}")
        return

    print("Training system...")
    train_result = await ai_system.train_mode(training_data)
    print("Training result:", train_result)

    print("\nTesting system...")
    query = "我想捐贈罐頭，請問哪裡可以捐？"
    use_result = await ai_system.use_mode(query)
    print("Query result:", use_result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())