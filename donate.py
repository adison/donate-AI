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
        """訓練模式"""
        try:
            # 準備訓練數據
            formatted_data = self._format_training_data(training_data)

            # 建立或更新模型
            await self.client.create(
                model=self.model_name,
                path='./training_data',
                template=self._get_prompt_template()
            )

            # 更新知識庫
            self.knowledge_base.update(formatted_data)
            self.save_knowledge_base()

            return {"status": "success", "message": "Model trained successfully"}

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

    def _get_prompt_template(self) -> str:
        """獲取 prompt 模板"""
        return """
        System: You are a donation assistant. Use the following knowledge to answer questions:
        {context}

        Human: {query}

        Assistant: Let me help you with that.
        """

    async def use_mode(self, query: str) -> Dict:
        """使用模式"""
        try:
            # 準備 context
            context = json.dumps(self.knowledge_base, ensure_ascii=False)

            # 生成 prompt
            prompt = self._get_prompt_template().format(
                context=context,
                query=query
            )

            # 獲取回應
            response = await self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )

            return {
                "status": "success",
                "response": response.message.content
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    async def update_knowledge(self, new_data: Dict):
        """更新知識庫"""
        try:
            # 更新知識庫
            self.knowledge_base.update(new_data)
            self.save_knowledge_base()

            # 可選：重新訓練模型
            await self.train_mode([new_data])

            return {"status": "success", "message": "Knowledge base updated"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

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

    train_result = await ai_system.train_mode(training_data)
    print("Training result:", train_result)

    # 使用模式示例
    query = "我想捐贈罐頭，請問哪裡可以捐？"
    use_result = await ai_system.use_mode(query)
    print("Query result:", use_result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())