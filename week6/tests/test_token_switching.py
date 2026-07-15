import pytest
from PIL import Image
from week6.services.generation_service import GenerationService
from week6.services.lora_service import LoRAService

def test_generation_service_token_switching():
    service = GenerationService(mock_mode=True)
    
    # 1. Base prompt without token
    res_base = service.generate("A nice running hoodie")
    assert res_base.is_ok
    assert "LoRA" not in res_base.meta.get("style_preset", "")
    
    # 2. Prompt with brand token <nike>
    res_nike = service.generate("<nike> A nice running hoodie")
    assert res_nike.is_ok
    # Cleaned prompt should not have the token
    assert "<nike>" not in res_nike.data["prompt"]
    # Should automatically detect Nike style label
    assert "LoRA [NIKE] (Mock)" in res_nike.meta.get("style_preset", "")

def test_lora_service_token_switching():
    service = LoRAService(mock_mode=True)
    
    # 1. Normal prompt with brand
    res_base = service.generate_with_brand("A running shoe", "gucci")
    assert res_base.is_ok
    assert res_base.meta["brand"] == "Gucci"
    
    # 2. Override brand dropdown with <nike> token in prompt
    res_nike = service.generate_with_brand("<nike> A running shoe", "gucci")
    assert res_nike.is_ok
    # Token should be stripped
    assert "<nike>" not in res_nike.data["prompt"]
    # Brand should be overridden to nike (even though "gucci" was passed to the dropdown parameter)
    assert res_nike.meta["brand"] == "Nike"
    assert "nike" in res_nike.data["brand"]
