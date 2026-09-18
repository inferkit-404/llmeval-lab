# LLMEval-Lab

**轻量级多模态评测实验室**

一个全面的、生产级的大模型评测框架，支持语言模型和多模态模型的统一评测接口、Benchmark接入、Checkpoint追踪和自动化评测工作流。

## 特性

- **统一评测接口** - 通过单一API评测语言模型和多模态模型
- **多后端支持** - HuggingFace Transformers、vLLM、OpenAI API、Anthropic Claude
- **Benchmark库** - 预置支持MMLU、HellaSwag、TruthfulQA、GSM8K、MMMU等
- **Checkpoint追踪** - 监控模型训练过程中的能力变化
- **自动化工作流** - 定时调度评测任务，支持失败重试
- **交互式仪表板** - 通过雷达图、能力对比、趋势分析可视化结果
- **可扩展设计** - 轻松添加新的Benchmark和模型适配器

## 安装

```bash
# 基本安装
pip install llmeval-lab

# 安装所有依赖
pip install llmeval-lab[all]

# 支持vLLM
pip install llmeval-lab[vllm]

# 支持API
pip install llmeval-lab[api]
```

## 快速开始

```python
from llmeval_lab import LLMEvaluator

# 初始化评测器
evaluator = LLMEvaluator(
    model="gpt-4",
    backend="api"
)

# 添加Benchmark
evaluator.add_benchmark("mmlu")
evaluator.add_benchmark("hellaswag")

# 运行评测
results = evaluator.run(
    benchmarks=["mmlu", "hellaswag"],
    num_samples=10
)

# 打印摘要
evaluator.print_summary(results)
```

## 命令行接口

```bash
# 运行评测
llmeval eval --model gpt-4 --backend api --benchmarks mmlu hellaswag

# 追踪Checkpoint
llmeval track --model llama-2 --checkpoint ./checkpoints/step1000 --metrics mmlu=0.65

# 列出Benchmark
llmeval list

# 分析结果
llmeval analyze --input results.json --model gpt-4 --output report.html
```

## 支持的模型

| 后端 | 模型 | 安装方式 |
|------|------|----------|
| HuggingFace | GPT-2, LLaMA, Mistral, Qwen等 | `pip install transformers torch` |
| vLLM | 所有HuggingFace模型（vLLM优化） | `pip install vllm` |
| OpenAI | GPT-4, GPT-3.5-turbo | `pip install openai` |
| Anthropic | Claude 3, Claude 2 | `pip install anthropic` |

## 支持的Benchmark

### 语言模型Benchmark
- **MMLU** - 大规模多任务语言理解
- **HellaSwag** - 常识推理
- **TruthfulQA** - 真实性评估
- **GSM8K** - 数学推理

### 多模态Benchmark
- **MMMU** - 大规模多模态理解
- **SEED-Bench** - 多模态理解

## 架构

```
LLMEval-Lab/
├── core/              # 核心评测引擎
├── adapters/         # 模型适配器
├── benchmarks/       # Benchmark实现
├── data/            # 数据处理
├── workflow/        # 任务调度
├── visualization/   # 图表和仪表板
└── cli.py           # 命令行接口
```

## Checkpoint追踪

监控模型在训练过程中的能力变化：

```python
from llmeval_lab import CheckpointTracker

tracker = CheckpointTracker()

# 记录Checkpoint评测结果
tracker.record(
    model_name="llama-2-7b",
    checkpoint_path="./checkpoints/step_5000",
    metrics={"mmlu": 0.72, "hellaswag": 0.78},
    step=5000
)

# 生成能力报告
tracker.print_capability_report("llama-2-7b")
```

## 自动化工作流

使用cron-like表达式调度评测任务：

```python
from llmeval_lab import TaskScheduler

scheduler = TaskScheduler(max_concurrent=4)

scheduler.add_task(
    task_id="nightly_eval",
    name="每日评测",
    func=run_evaluation,
)

scheduler.schedule_task("nightly_eval", "0 9 * * *")  # 每天上午9点
```

## 可视化

生成交互式HTML报告：

```python
from llmeval_lab import Dashboard

dashboard = Dashboard()
dashboard.create_model_comparison_dashboard(
    results={"model_a": {...}, "model_b": {...}},
    output_path="comparison.html"
)
```

## 开发

```bash
# 克隆仓库
git clone https://github.com/ai-never/LLMEval-Lab.git
cd LLMEval-Lab

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 格式化代码
black .
```

## 贡献

欢迎贡献！请提交Pull Request。

## 许可证

Apache License 2.0 - 详见 [LICENSE](LICENSE)。

## 引用

```bibtex
@software{llmeval_lab,
  title = {LLMEval-Lab: 轻量级多模态评测实验室},
  author = {LLMEval Team},
  year = {2024},
  url = {https://github.com/ai-never/LLMEval-Lab}
}
```
