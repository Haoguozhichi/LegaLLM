import os
import json
import argparse
from vllm import LLM, SamplingParams
from autoawq import AutoAWQForCausalLM

# ========================
# Step 1: 模型加载与量化
# ========================

def load_quantized_model(model_path, quantized_path=None):
    """
    加载模型并应用 AWQ 8-bit 量化。
    如果已经量化过，直接加载量化后的模型。
    """
    if quantized_path and os.path.exists(quantized_path):
        print("Loading pre-quantized model...")
        model = AutoAWQForCausalLM.from_pretrained(quantized_path)
    else:
        print("Quantizing the model using AWQ...")
        model = AutoAWQForCausalLM.from_pretrained(model_path)
        model.quantize()
        model.save_pretrained(quantized_path)  # 保存量化后的模型
    return model

# ========================
# Step 2: 使用 vLLM 引擎进行推理
# ========================

class VLLMInferenceEngine:
    def __init__(self, model_path, quantized_path=None, gpu_memory_utilization=0.9):
        """
        初始化推理引擎。
        :param model_path: 原始模型路径
        :param quantized_path: 量化后模型路径（可选）
        :param gpu_memory_utilization: 显存利用率（0.0 到 1.0）
        """
        # 加载量化模型
        self.model = load_quantized_model(model_path, quantized_path)
        
        # 初始化 vLLM 引擎
        self.llm = LLM(
            model=self.model,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True  # 启用动态显存分配
        )
    
    def generate(self, prompts, max_tokens=100, temperature=0.7, top_p=0.9):
        """
        批量生成推理结果。
        :param prompts: 输入的 Prompt 列表
        :param max_tokens: 最大生成长度
        :param temperature: 温度参数，控制生成随机性
        :param top_p: 核采样参数
        :return: 生成的文本列表
        """
        # 设置采样参数
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # 使用 vLLM 进行推理
        outputs = self.llm.generate(prompts, sampling_params)
        
        # 提取生成的文本
        generated_texts = [output.outputs[0].text for output in outputs]
        return generated_texts

# ========================
# Step 3: 命令行接口
# ========================

def main():
    # 加载配置文件
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="基于 vLLM 的法律大模型推理工具")
    
    # 添加命令行参数，并使用配置文件中的值作为默认值
    parser.add_argument("--model_path", type=str, default=config["model_path"], help="原始模型路径")
    parser.add_argument("--quantized_path", type=str, default=config["quantized_path"], help="量化后模型路径（可选）")
    parser.add_argument("--gpu_memory_utilization", type=float, default=config["gpu_memory_utilization"], help="显存利用率（0.0 到 1.0）")
    parser.add_argument("--max_tokens", type=int, default=config["max_tokens"], help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=config["temperature"], help="温度参数，控制生成随机性")
    parser.add_argument("--top_p", type=float, default=config["top_p"], help="核采样参数")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 初始化推理引擎
    inference_engine = VLLMInferenceEngine(
        model_path=args.model_path,
        quantized_path=args.quantized_path,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    print("模型已加载，输入 'exit' 或 'quit' 退出对话。\n")
    
    # 开启交互式对话
    while True:
        try:
            # 用户输入
            prompt = input("请输入你的问题：")
            
            # 检查退出条件
            if prompt.lower() in ["exit", "quit"]:
                print("退出对话，再见！")
                break
            
            # 执行推理
            print("正在生成回答...")
            results = inference_engine.generate(
                prompts=[prompt],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            # 输出结果
            print(f"\n回答: {results[0]}\n")
        
        except KeyboardInterrupt:
            print("\n检测到中断信号，退出对话。")
            break

if __name__ == "__main__":
    main()