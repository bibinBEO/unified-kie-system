import torch
import asyncio
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import json
from typing import Dict, Any
from datetime import datetime

class LayoutLMExtractor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self):
        """Initialize LayoutLM model"""
        def load_model():
            model_name = self.config.LAYOUTLM_MODEL
            
            processor = LayoutLMv3Processor.from_pretrained(model_name)
            model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
            
            return model, processor
        
        loop = asyncio.get_event_loop()
        self.model, self.processor = await loop.run_in_executor(None, load_model)
        print(f"✅ LayoutLM v3 loaded on {self.device}")

    async def extract(self, image: Image.Image, language: str = "auto") -> Dict[str, Any]:
        """Extract information using LayoutLM"""
        def inference():
            # Process image
            encoding = self.processor(image, return_tensors="pt")
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoding)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Convert predictions to readable format
            tokens = self.processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
            predictions = predictions[0].cpu().numpy()
            
            # Extract entities (simplified)
            entities = []
            for i, (token, pred) in enumerate(zip(tokens, predictions)):
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    max_label_idx = int(pred.argmax())  # Convert numpy int to Python int
                    confidence = float(pred[max_label_idx])  # Convert numpy float to Python float
                    if confidence > 0.5:  # Confidence threshold
                        entities.append({
                            "token": token,
                            "label": f"LABEL_{max_label_idx}",
                            "confidence": confidence
                        })
            
            return entities
        
        loop = asyncio.get_event_loop()
        entities = await loop.run_in_executor(None, inference)
        
        # Process entities into key-value pairs
        key_values = self._entities_to_key_values(entities)
        
        return {
            "raw_entities": entities,
            "key_values": key_values,
            "extraction_method": "layoutlm_v3",
            "timestamp": datetime.now().isoformat()
        }

    def _entities_to_key_values(self, entities) -> Dict[str, Any]:
        """Convert entities to key-value pairs"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated entity linking
        key_values = {}
        
        current_key = None
        current_value = []
        
        for entity in entities:
            token = entity["token"].replace("##", "")  # Handle subword tokens
            label = entity["label"]
            
            if "KEY" in label:
                if current_key and current_value:
                    key_values[current_key] = " ".join(current_value)
                current_key = token.lower().replace(" ", "_")
                current_value = []
            elif "VALUE" in label and current_key:
                current_value.append(token)
        
        # Add last key-value pair
        if current_key and current_value:
            key_values[current_key] = " ".join(current_value)
        
        return key_values