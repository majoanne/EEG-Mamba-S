import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gradio as gr
import mne
import plotly.graph_objects as go
from pathlib import Path
from einops import rearrange, repeat, einsum

BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "models"
PICTURE_DIR = BASE_DIR / "picture"
TEST_DATA_DIR = BASE_DIR / "test_data"

FIG1_PATH = PICTURE_DIR / "fig1.png"
FIG2_PATH = PICTURE_DIR / "fig2.png"
AUTHOR_PHOTO_PATH = PICTURE_DIR / "mjl.jpg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 dt_rank=13, conv_bias=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = self.act(x)
        y = self._ssm(x)
        y = y * self.act(res)
        return self.out_proj(y)

    def _ssm(self, x):
        (b, l, d_in) = x.shape
        n = self.A_log.shape[1]
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split([self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, x, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x_state = torch.zeros((b, d_in, n), device=x.device)
        ys = []
        for i in range(l):
            x_state = deltaA[:, i] * x_state + deltaB_u[:, i]
            y = einsum(x_state, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        return y + x * D

class EEGMambaS(nn.Module):
    def __init__(self, d_model=200, d_state=16, d_conv=4, expand=2,
                 dt_rank=13, num_layers=4, num_classes=1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.backbone = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.backbone.append(
                MambaBlock(d_model, d_state, d_conv, expand, dt_rank, conv_bias=True, bias=False)
            )
            self.norms.append(nn.LayerNorm(d_model))
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, C * T, F)
        for layer, norm in zip(self.backbone, self.norms):
            residual = x
            x = norm(layer(x)) + residual
        x = x.mean(dim=1)
        return torch.sigmoid(self.classifier(x))

def load_models():
    model_s = EEGMambaS()
    model_c = EEGMambaS()
    s_path = MODEL_DIR / "seizeit2model.pth"
    c_path = MODEL_DIR / "chbmitmodel.pth"

    if s_path.exists():
        state = torch.load(s_path, map_location=DEVICE, weights_only=True)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model_s.load_state_dict(state)
        print("✅ Loaded SeizeIT2 model")
    else:
        print(f"❌ Not found: {s_path}")

    if c_path.exists():
        state = torch.load(c_path, map_location=DEVICE, weights_only=True)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model_c.load_state_dict(state)
        print("✅ Loaded CHB-MIT model")
    else:
        print(f"❌ Not found: {c_path}")

    model_s.eval()
    model_c.eval()
    return model_s.to(DEVICE), model_c.to(DEVICE)

model_seizeit, model_chbmit = load_models()

def plot_eeg_interactive(data, sfreq, title, ch_names):
    time_axis = np.arange(data.shape[1]) / sfreq
    fig = go.Figure()
    offset = np.max(np.abs(data)) * 2.5
    for i, ch in enumerate(data):
        fig.add_trace(go.Scatter(
            x=time_axis, y=ch - i*offset,
            mode='lines', name=ch_names[i], line=dict(width=1.2)
        ))
    fig.update_layout(
        title=title, xaxis_title='Time (s)', yaxis_title='',
        showlegend=True, height=320, template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_yaxes(showticklabels=False)
    return fig

def process_and_predict(file_obj, dataset_choice):
    if file_obj is None:
        raise gr.Error("Please upload an EEG file (.edf, .fif, .csv)")

    if "SeizeIT2" in dataset_choice:
        model = model_seizeit
        l_freq, h_freq = 0.5, 40.0
        ch_names = ['Left BTE', 'Right BTE']
    else:
        model = model_chbmit
        l_freq, h_freq = 0.5, 75.0
        ch_names = ['T7-P7', 'T8-P8']

    try:
        fname = file_obj.name
        if fname.endswith('.edf'):
            raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
        elif fname.endswith('.fif'):
            raw = mne.io.read_raw_fif(fname, preload=True, verbose=False)
        else:
            data = np.loadtxt(fname, delimiter=',').T
            raw = mne.io.RawArray(data, mne.create_info(ch_names, 200, ch_types='eeg'), verbose=False)

        raw = raw.pick_channels(raw.ch_names[:2]) if len(raw.ch_names) >= 2 else raw
        sfreq_orig = raw.info['sfreq']
        n_samples = int(sfreq_orig * 30)
        data_raw, _ = raw[:2, :n_samples] if raw.n_times >= n_samples else raw[:2, :]
        if data_raw.shape[1] < n_samples:
            data_raw = np.pad(data_raw, ((0,0),(0, n_samples - data_raw.shape[1])), mode='constant')

        fig_raw = plot_eeg_interactive(data_raw, sfreq_orig, "Raw EEG Signal", ch_names)

        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        raw.notch_filter(50.0, verbose=False)
        raw.resample(200)
        data_proc = raw.get_data()[:2, :n_samples] if raw.n_times >= n_samples else raw.get_data()[:2, :]

        for ch in range(2):
            mu, sig = np.mean(data_proc[ch]), np.std(data_proc[ch])
            data_proc[ch] = (data_proc[ch] - mu) / (sig + 1e-6)

        fig_proc = plot_eeg_interactive(data_proc, 200, "Preprocessed Signal (Filtered + Z‑score)", ch_names)

        tensor_data = torch.tensor(data_proc, dtype=torch.float32).view(1, 2, 30, 200).to(DEVICE)
        with torch.no_grad():
            prob = model(tensor_data).item()

        if prob > 0.4:
            result_html = """
            <div style="border:3px solid #d32f2f; padding:20px; border-radius:12px; background:#fff5f5; min-height:150px; display:flex; align-items:center; justify-content:center; text-align:center;">
                <span style="color:#b71c1c; font-size:2em; font-weight:bold;">⚠️ High likelihood of preictal activity</span>
            </div>
            """
        else:
            result_html = """
            <div style="border:3px solid #2e7d32; padding:20px; border-radius:12px; background:#f0fff4; min-height:150px; display:flex; align-items:center; justify-content:center; text-align:center;">
                <span style="color:#1b5e20; font-size:2em; font-weight:bold;">✅ Low likelihood (likely interictal)</span>
            </div>
            """
        return fig_raw, fig_proc, result_html
    except Exception as e:
        raise gr.Error(f"Processing failed: {str(e)}")

def load_seizeit2_seizure():
    return str(TEST_DATA_DIR / "seizeit2_seizure.edf"), "SeizeIT2 (BTE - Behind The Ear)", "**Current Mode: SeizeIT2 (BTE - Behind The Ear)**"

def load_seizeit2_normal():
    return str(TEST_DATA_DIR / "seizeit2_normal.edf"), "SeizeIT2 (BTE - Behind The Ear)", "**Current Mode: SeizeIT2 (BTE - Behind The Ear)**"

def load_chbmit_seizure():
    return str(TEST_DATA_DIR / "chbmit_seizure.edf"), "CHB-MIT (Temporal T7-P7 / T8-P8)", "**Current Mode: CHB-MIT (Scalp T7-P7 / T8-P8)**"

def load_chbmit_normal():
    return str(TEST_DATA_DIR / "chbmit_normal.edf"), "CHB-MIT (Temporal T7-P7 / T8-P8)", "**Current Mode: CHB-MIT (Scalp T7-P7 / T8-P8)**"

def activate_seizeit2():
    return "SeizeIT2 (BTE - Behind The Ear)", "**Current Mode: SeizeIT2 (BTE - Behind The Ear)**"

def activate_chbmit():
    return "CHB-MIT (Temporal T7-P7 / T8-P8)", "**Current Mode: CHB-MIT (Scalp T7-P7 / T8-P8)**"

css = """
.tab-nav button { font-size: 27px !important; font-weight: 600 !important; }
.framework-img { max-width: 50% !important; height: auto !important; display: block; margin-left: 0; }
.electrode-img { max-width: 75% !important; height: auto !important; display: block; margin-left: 0; }
.control-block { max-width: 75% !important; margin-left: 0 !important; }
.author-row { justify-content: center !important; gap: 0 !important; align-items: flex-start !important; }
.author-photo-col { width: auto !important; flex: none !important; }
.author-photo-col .gr-image { display: block; margin: 0; margin-left: auto; }
.author-info { text-align: left !important; padding-left: 0 !important; margin-left: 0 !important; white-space: nowrap !important; }
.example-row { margin-top: 8px; margin-bottom: 8px; }
.upload-hint { font-size: 0.9em; color: #666; }
"""

with gr.Blocks(title="EEG-Mamba-S Seizure Predictor", css=css, theme=gr.themes.Soft()) as demo:

    with gr.Tab("📚 Algorithm & Background"):
        gr.Markdown("""
        # 🔬 EEG‑Mamba‑S: A Lightweight State‑Space Model for Two‑Channel Wearable Seizure Prediction

        ## 📋 Overview
        - **Architecture**: 4‑layer S6 backbone, linear complexity O(L)  
        - **Parameters**: 1.1 M  
        - **Inference speed**: 1.4 ms (NVIDIA RTX 5060 Laptop GPU, FP32)  
        - **Training strategy**: Random undersampling + Focal Loss (γ=3.0, α=0.5)  
        - **Training mode**: All models trained **from scratch** (no pretrained weights)

        ## 🧪 Core Performance
        | Dataset | Channels | ROC‑AUC (3‑seed avg) | Notes |
        |--------|--------|----------------------|------|
        | **SeizeIT2** (wearable BTE) | 2 | **0.9020 ± 0.0067** | 20 subjects, 80/20 split |
        | **CHB‑MIT** (scalp EEG) | 2 | **0.9048 ± 0.0036** | Independently reproduced from scratch |

        At threshold 0.4: sensitivity 89.3%, specificity 74.3%.

        ## 🏗️ Architecture Highlights
        Input (2, 30, 200) → interleave to (60, 200) → 4×S6 blocks (d_model=200, d_state=16, expand=2, residual + LayerNorm) → global average pooling → 3‑layer MLP → sigmoid.

        ## 🔬 Ablation Study Key Findings
        - 4 layers >> 1/2 layers; Focal Loss (γ=3) > BCE; 30s window > 15s; 0.5‑40 Hz filter robust.
        """)
        if FIG1_PATH.exists():
            gr.Markdown("### Overall Framework")
            gr.Image(str(FIG1_PATH), label="EEG‑Mamba‑S Pipeline", show_label=False, elem_classes="framework-img")

        gr.Markdown("---")
        gr.Markdown("## 📞 Contact us while you have any problems with EEG‑Mamba‑S")
        with gr.Row(elem_classes="author-row", equal_height=False):
            with gr.Column(scale=1, min_width=100, elem_classes="author-photo-col"):
                gr.Image(value=str(AUTHOR_PHOTO_PATH), label="", show_label=False,
                         height=160, width=120, interactive=False, container=False)
            with gr.Column(scale=5, elem_classes="author-info"):
                gr.HTML("""
                <ul style="list-style: none; padding: 0; margin: 0;">
                    <li style="margin-bottom: 8px;">
                        <span style="font-size: 1.6em; font-weight: bold; color: #1A7A4E; text-decoration: underline;">Junli Ma</span>
                    </li>
                    <li style="margin-bottom: 8px; display: flex; align-items: center; gap: 8px; font-size: 1em;">
                        <span style="font-size: 1.2em;">📧</span> 
                        <b>Email:</b> <a href="mailto:majoanne@163.com" style="color: #2563EB; text-decoration: underline;">majoanne@163.com</a>
                    </li>
                    <li style="display: flex; align-items: flex-start; gap: 8px; font-size: 1em; line-height: 1.3;">
                        <span style="font-size: 1.2em;">🏢</span> 
                        <b>Address:</b> <span>Fujian Medical University, Xue Yuan Road, University Town, FuZhou, Fujian, China</span>
                    </li>
                </ul>
                """)

    with gr.Tab("⚡ Interactive Prediction"):
        gr.Markdown("## 🧠 Real‑Time Seizure Prediction Based on Two‑Channel Wearable EEG")

        if FIG2_PATH.exists():
            gr.Image(str(FIG2_PATH), label="Electrode setups: SeizeIT2 (left) vs. CHB‑MIT (right)", show_label=False, elem_classes="electrode-img")
        else:
            gr.Markdown("*(Electrode setup image not found)*")

        with gr.Column(elem_classes="control-block"):
            dataset_state = gr.State("SeizeIT2 (BTE - Behind The Ear)")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### SeizeIT2 (BTE - Behind The Ear)")
                    gr.Markdown("*Wearable behind‑the‑ear bipolar channel configuration*")
                    btn_activate_sz = gr.Button("🔵 Activate SeizeIT2 Mode", variant="secondary")
                    gr.Markdown("#### Test examples")
                    with gr.Row(elem_classes="example-row"):
                        btn_sz_seiz = gr.Button("🔴 Seizure", size="sm")
                        btn_sz_norm = gr.Button("🟢 Normal", size="sm")

                with gr.Column(scale=1):
                    gr.Markdown("### CHB‑MIT (Scalp T7‑P7 / T8‑P8)")
                    gr.Markdown("*Clinical scalp EEG approximated bipolar channels*")
                    btn_activate_chb = gr.Button("🔴 Activate CHB‑MIT Mode", variant="secondary")
                    gr.Markdown("#### Test examples")
                    with gr.Row(elem_classes="example-row"):
                        btn_chb_seiz = gr.Button("🔴 Seizure", size="sm")
                        btn_chb_norm = gr.Button("🟢 Normal", size="sm")

            mode_display = gr.Markdown("**Current Mode: SeizeIT2 (BTE - Behind The Ear)**")
            eeg_file = gr.File(label="📁 Upload 30‑second EEG segment", file_types=[".edf", ".fif", ".csv"])
            gr.Markdown("💡 **Upload a 30‑second two‑channel EEG file (.edf, .fif, or .csv)** and click **Run Prediction**. You can also try the built‑in test examples above to see the model's performance on known recordings.", elem_classes="upload-hint")
            run_btn = gr.Button("🚀 Run Prediction", variant="primary")

            with gr.Row():
                with gr.Column(scale=2):
                    plot_raw = gr.Plot(label="Raw Signal")
                    plot_proc = gr.Plot(label="Preprocessed Signal")
                with gr.Column(scale=1):
                    result_html = gr.HTML()

        btn_activate_sz.click(fn=activate_seizeit2, outputs=[dataset_state, mode_display])
        btn_activate_chb.click(fn=activate_chbmit, outputs=[dataset_state, mode_display])
        btn_sz_seiz.click(fn=load_seizeit2_seizure, outputs=[eeg_file, dataset_state, mode_display])
        btn_sz_norm.click(fn=load_seizeit2_normal, outputs=[eeg_file, dataset_state, mode_display])
        btn_chb_seiz.click(fn=load_chbmit_seizure, outputs=[eeg_file, dataset_state, mode_display])
        btn_chb_norm.click(fn=load_chbmit_normal, outputs=[eeg_file, dataset_state, mode_display])
        run_btn.click(fn=process_and_predict, inputs=[eeg_file, dataset_state], outputs=[plot_raw, plot_proc, result_html])

    with gr.Tab("🛠️ Explore More"):
        gr.Markdown("""# 🚀 Explore More: Resources for You""")
        gr.Markdown("**Interested in reproducing our results or exploring seizure prediction further? All key resources are organized below.**")

        gr.HTML("""
        <div style="background: #f0fdf4; border-left: 6px solid #16a34a; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin-top:0; color:#15803d;">📥 Datasets</h2>
            <p style="font-size:1.05em; line-height:1.5;">The two EEG cohorts used in our study are publicly available.</p>
            <ul style="line-height:1.8; font-size:1.05em;">
                <li><b>SeizeIT2</b> (wearable behind‑the‑ear, 125 subjects) → <a href="https://doi.org/10.18112/openneuro.ds005873.v1.1.0" target="_blank" style="color: #16a34a; text-decoration: underline;">OpenNeuro DOI</a></li>
                <li><b>CHB‑MIT</b> (scalp EEG, pediatric, 24 cases) → <a href="https://physionet.org/content/chbmit/1.0.0/" target="_blank" style="color: #16a34a; text-decoration: underline;">PhysioNet</a></li>
                <li><b>Siena Scalp EEG</b> (focal epilepsy, 14 adults) → <a href="https://physionet.org/content/siena-scalp-eeg/1.0.0/" target="_blank" style="color: #16a34a; text-decoration: underline;">PhysioNet</a></li>
            </ul>
        </div>
        """)

        gr.HTML("""
        <div style="background: #eff6ff; border-left: 6px solid #2563eb; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin-top:0; color:#1e40af;">💻 Open‑Source Code</h2>
            <p style="font-size:1.05em; line-height:1.5;">Our implementation and the original Mamba SSM.</p>
            <ul style="line-height:1.8; font-size:1.05em;">
                <li><a href="https://github.com/majoanne/EEG-Mamba-S" target="_blank" style="color: #2563eb; text-decoration: underline;"><b>EEG‑Mamba‑S</b></a> — Official training code & pretrained models</li>
                <li><a href="https://github.com/state-spaces/mamba" target="_blank" style="color: #2563eb; text-decoration: underline;"><b>Mamba</b></a> — Original selective SSM (PyTorch)</li>
            </ul>
        </div>
        """)

        gr.HTML("""
        <div style="background: #fff8f1; border-left: 6px solid #ea580c; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin-top:0; color:#9a3412;">📚 Papers & Tutorials</h2>
            <p style="font-size:1.05em; line-height:1.5;">Key references for Mamba, Focal Loss, and community examples.</p>
            <ul style="line-height:1.8; font-size:1.05em;">
                <li><b>Mamba paper</b>: <a href="https://arxiv.org/abs/2312.00752" target="_blank" style="color: #ea580c; text-decoration: underline;">arXiv 2312.00752</a></li>
                <li><b>Focal Loss paper</b>: <a href="https://arxiv.org/abs/1708.02002" target="_blank" style="color: #ea580c; text-decoration: underline;">arXiv 1708.02002</a></li>
                <li><b>Kaggle notebooks</b>: <a href="https://www.kaggle.com/search?q=seizure+prediction+eeg" target="_blank" style="color: #ea580c; text-decoration: underline;">Seizure prediction examples</a></li>
            </ul>
        </div>
        """)

        gr.Markdown("**✨ After exploring these resources, feel free to come back and use our app for real‑time seizure prediction!**")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)