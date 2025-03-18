---

# **LegaLLM: 高效法律大模型微调与推理加速**

![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue) ![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)

---

## **项目简介**

本项目基于 LLaMA-Factory 开源框架开发，旨在通过高效微调和推理优化技术，使 DeepSeek 32B R1 模型具备强大的法律问答、法律解释和案例分析能力。我们利用 **881 部全国性法律**（包括宪法、民法、刑法等）中的 **20 万条法律条款**，对模型进行微调，使其在端侧部署中表现出色。

项目亮点：
- **高效微调**：采用 **4-bit QLoRA + DeepSpeed ZeRO-3** 方案，在单卡 24GB 显存下完成大规模模型微调。
- **推理优化**：结合 **AWQ 8-bit 量化压缩** 和 **vLLM 引擎**，显著提升推理速度并降低显存占用。
- **三段式 Prompt 设计**：优化指令结构，增强模型生成的专业性和准确性。

---

## **安装与运行**

### **环境准备**
1. 克隆本项目代码：
   ```bash
   git clone https://github.com/your-repo/legal-lm.git
   cd legal-lm
   ```
2. 创建虚拟环境并安装依赖：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows 用户使用 `venv\Scripts\activate`
   pip install -e ".[torch,metrics,deepspeed,bitsandbytes,awq,vllm]"
   ```

### **数据预处理**
- 数据文件及预处理代码存储在 `my_job/data` 文件夹中。
- 数据已处理为 Alpaca 格式，并设计了三段式 Prompt 模板，确保输入输出结构化。

### **模型微调**
1. 运行以下命令开始微调：
   ```bash
   lmf train my_job/train/deepseekr1_32b_lora_sft_chlaw.yaml
   ```
2. 微调完成后，模型参数将保存在 `saves/deepseekr1-32b/qlora/sft` 文件夹中。

### **推理部署**
1. 在 `vllm` 文件夹中运行以下命令生成 AWQ 8-bit 量化后的模型：
   ```bash
   python run_vllm.py
   ```
2. 量化后的模型将保存在 `saves/awq_quantized_model` 文件夹中。
3. 使用以下命令加载并运行模型：
   ```bash
   python inference.py --model_path saves/awq_quantized_model
   ```

---

## **技术栈**

- **数据处理**：Alpaca 格式、三段式 Prompt 设计
- **微调方案**：
  - **4-bit QLoRA**：减少显存占用，支持单卡微调
  - **DeepSpeed ZeRO-3**：分布式训练优化
- **推理优化**：
  - **AWQ 8-bit 量化**：压缩模型体积，降低显存需求
  - **vLLM 引擎**：实现动态显存分配和高效推理
- **硬件要求**：单卡 RTX 4090 或类似配置

---

## **功能亮点**

1. **法律问答**：
   - 输入问题后，模型能够准确匹配相关法律条文并生成专业回答。
   - 示例：
     ```plaintext
     输入：什么是“不可抗力”？
     输出：根据《民法典》第 180 条，“不可抗力”是指不能预见、不能避免且不能克服的客观情况。
     ```

2. **法律解释**：
   - 对复杂法律条文进行通俗易懂的解析。
   - 示例：
     ```plaintext
     输入：请解释《刑法》第 232 条。
     输出：该条款规定了故意杀人罪的定义及量刑标准。
     ```

3. **案例分析**：
   - 结合法律条文和案例背景，提供专业的法律建议。
   - 示例：
     ```plaintext
     输入：某人因合同纠纷被起诉，如何应对？
     输出：建议先确认合同条款是否明确，再根据《合同法》第 XXX 条提出答辩。
     ```

---

## **贡献指南**

欢迎任何形式的贡献！如果你有任何改进建议或发现了问题，请提交 Issue 或 Pull Request。

1. Fork 本项目。
2. 创建你的分支 (`git checkout -b feature/your-feature`)。
3. 提交更改 (`git commit -m "Add your feature"`)。
4. 推送到分支 (`git push origin feature/your-feature`)。
5. 提交 Pull Request。

---

## 致谢

本项目基于以下开源框架开发，特别感谢其开发团队的贡献：

- **LLaMA-Factory**: [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
  - LLaMA-Factory 是一个用于高效微调和推理大语言模型的开源框架。
  - 本项目使用了 LLaMA-Factory 提供的工具和功能。

- **vLLM**: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
  - vLLM 是一个高性能的推理引擎，支持动态显存分配和批处理调度。

本项目遵守上述开源框架的许可证要求。

---

## **许可证**

本项目采用 [MIT License](LICENSE) 开源协议。

---

希望这篇 README 符合你的需求！如果有需要调整的地方，请随时告诉我！
