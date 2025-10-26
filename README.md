# ⚙️ CUDA Kernel Inference Profiler

**CUDA Kernel Inference Profiler** is a lightweight, Colab-ready benchmarking tool that profiles transformer inference at the **kernel, operator, and memory** level.  
It helps you **analyze bottlenecks**, visualize **kernel-level latency**, and generate **actionable optimization hints** — inspired by NVIDIA’s Algorithmic Model Optimization workflows.

---

## 🚀 Overview

When deploying LLMs or diffusion models, most inefficiencies come from:
- Unoptimized CUDA kernels  
- Memory-bound attention operations  
- Redundant operator launches  
- Lack of kernel fusion or precision tuning  

This notebook automatically:
1. Profiles transformer inference using `torch.profiler`
2. Aggregates CUDA + CPU time per operator
3. Detects performance hotspots
4. Suggests optimizations (FlashAttention, fused LayerNorm, mixed precision, TensorRT)
5. Optionally tests `torch.compile` to compare kernel fusion gains

---

## 🧩 Features

- ✅ **Operator-level profiling** via PyTorch Profiler  
- ✅ **Memory and latency measurement** (GPU peak tracking)  
- ✅ **Automatic bottleneck classification** (compute vs memory bound)  
- ✅ **Optimization hints** for real-world models  
- ✅ **Chrome trace export** for deep analysis (`chrome://tracing`)  
- ✅ **Torch.compile comparison** for fused kernel testing  

---

## 📘 Usage

1. Clone or open the notebook on Colab:
   ```bash
   git clone https://github.com/yourusername/cuda_kernel_inference_profiler.git
   cd cuda_kernel_inference_profiler
   ```
   or  
   [💻 Open in Google Colab](https://colab.research.google.com/github/yourusername/cuda_kernel_inference_profiler/blob/main/cuda_kernel_inference_profiler.ipynb)

2. Install dependencies (auto-installed in the first cell):
   ```bash
   pip install transformers datasets accelerate
   ```

3. Run the notebook sequentially — each cell handles one step:
   - Model + tokenizer loading  
   - Real dataset sampling (WikiText-2)  
   - Profiling with `torch.profiler`  
   - Visualization of top bottlenecks  
   - Optimization hint generation  
   - (Optional) `torch.compile` latency test  

---

## 📊 Example Output

| Metric | FP16 (Baseline) | After Optimization | Δ |
|:--|:--:|:--:|:--:|
| Latency / batch | 1.9 s | 1.2 s | 🔼 1.6× faster |
| Peak GPU memory | 9.2 GB | 5.8 GB | 🔽 1.6× less |
| Accuracy Δ (PPL) | +0.7 % | — | negligible |

Chrome trace (`trace.json`) can be loaded in Chrome → `chrome://tracing`.

---

## 🧠 Insights

- **Memory-bound ops** (e.g., attention, softmax, layernorm) often dominate latency.  
  → FlashAttention and fused norms can drastically help.  
- **Compute-bound ops** (e.g., matmul, linear) benefit from Tensor Cores via FP16/BF16.  
- **Torch.compile** can fuse ops and reduce launch overheads without manual kernel editing.

---

## 📂 Repo Structure

```
cuda_kernel_inference_profiler/
│
├── cuda_kernel_inference_profiler.ipynb     # Main notebook
├── results/
│   ├── trace.json                           # Chrome trace output
│   └── reports/                             # (Optional) markdown summaries
├── README.md
└── requirements.txt                         # Dependencies (auto-install optional)
```

---

## 🧑‍💻 Tech Stack

- **Framework:** PyTorch 2.x (`torch.profiler`, `torch.compile`)  
- **Inference:** Hugging Face Transformers (OPT models)  
- **Visualization:** Matplotlib, JSON trace export  
- **Environment:** Google Colab / CUDA GPU (T4, A100, RTX 40xx)  

---

## 📈 Future Work

- Integrate **FlashAttention2** and **Triton kernels**  
- Add **module-level hooks** for per-layer analysis  
- Export optimized graph to **TensorRT-LLM** for real deployment tests  

---

## 📜 License

MIT License © 2025 Your Name  
Contributions welcome — feel free to open issues or pull requests.

---

### 🌐 Connect
- **Author:** [Syed Mohammed Faham](https://github.com/iamfaham)  
- **Project Name:** *CUDA Kernel Inference Profiler*  
- **Version:** v1.0.0
