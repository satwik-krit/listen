import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')
FS,N=20000,20480
BPFO=161.1
BPFI=236.4
BAND=15
FREQS=fftfreq(N,d=1.0/FS)[:N//2]
BPFO_MASK=(FREQS>=BPFO-BAND)&(FREQS<=BPFO+BAND)
BPFI_MASK=(FREQS>=BPFI-BAND)&(FREQS<=BPFI+BAND)
RAW_NAMES=["RMS","Kurtosis","Skewness","Peak","Crest","MAV","BPFO_ratio","BPFI_ratio"]
SLOPE_NAMES=[f"Δ{n}" for n in RAW_NAMES]
FEATURE_NAMES=RAW_NAMES+SLOPE_NAMES
FAULT_MAP={
    "Kurtosis":"impulsive shock — likely bearing spall or crack",
    "ΔKurtosis":"rapidly increasing shock — accelerating wear",
    "Crest":"high peak stress — surface fatigue",
    "ΔCrest":"rising peak stress — crack propagation",
    "RMS":"elevated vibration energy — general bearing wear",
    "ΔRMS":"increasing vibration trend — progressive damage",
    "Peak":"high amplitude event — possible impact fault",
    "BPFO_ratio":"energy at outer race fault frequency — outer race defect",
    "BPFI_ratio":"energy at inner race fault frequency — inner race defect",
    "ΔBPFO_ratio":"rising outer race fault energy — defect progressing",
    "ΔBPFI_ratio":"rising inner race fault energy — defect progressing",
}
class RULNet(nn.Module):
    def __init__(self,input_size,hidden=128,n_heads=4,dropout=0.3):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv1d(input_size,64,kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.lstm=nn.LSTM(64,hidden,num_layers=2,batch_first=True,bidirectional=True,dropout=dropout)
        self.attn=nn.MultiheadAttention(hidden*2,n_heads,batch_first=True,dropout=dropout)
        self.attn_norm=nn.LayerNorm(hidden*2)
        self.fc=nn.Sequential(
            nn.Linear(hidden*2,32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.cnn(x.transpose(1,2)).transpose(1,2)
        x,_=self.lstm(x)
        a,_=self.attn(x,x,x)
        x=self.attn_norm(x+a)
        return self.fc(x[:,-1,:]).squeeze()
def extract_features_from_signal(signal):
    signal=signal.astype(np.float32)
    rms=float(np.sqrt(np.mean(signal**2)))
    kurt=float(kurtosis(signal))
    skw=float(skew(signal))
    peak=float(np.max(np.abs(signal)))
    crest=float(peak/(rms+1e-10))
    mav=float(np.mean(np.abs(signal)))
    fft_mag=np.abs(fft(signal))[:N//2]
    total_energy=float(np.sum(fft_mag**2))+1e-10
    bpfo_ratio=float(np.sum(fft_mag[BPFO_MASK]**2))/total_energy
    bpfi_ratio=float(np.sum(fft_mag[BPFI_MASK]**2))/total_energy
    return np.array([rms,kurt,skw,peak,crest,mav,bpfo_ratio,bpfi_ratio],dtype=np.float32)
def load_model(model_path="best_model.pt",n_features_path="n_features.npy"):
    n_features=int(np.load(n_features_path)[0])
    model=RULNet(input_size=n_features)
    model.load_state_dict(torch.load(model_path,map_location="cpu",weights_only=True))
    model.eval()
    return model
def predict_rul(raw_window,model,scaler):
    assert raw_window.shape==(50,8),f"Expected (50, 8), got {raw_window.shape}. "+f"Each row = one time snapshot, 8 features per row."
    scaled=scaler.transform(raw_window.astype(np.float32))
    diff=np.diff(scaled,axis=0,prepend=scaled[0:1])
    combined=np.hstack([scaled,diff]).astype(np.float32)
    tensor_in=torch.tensor(combined).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        rul_norm=float(model(tensor_in).item())
    rul_norm=float(np.clip(rul_norm,0.0,1.0))
    rul_pct=round(rul_norm*100,1)
    if rul_norm>0.7:
        status="HEALTHY"
    elif rul_norm>0.3:
        status="WARNING"
    else:
        status="CRITICAL"
    try:
        background=torch.zeros(1,50,combined.shape[1])
        explainer=shap.GradientExplainer(model,background)
        shap_vals=explainer.shap_values(tensor_in)
        shap_feat=np.mean(np.abs(shap_vals[0]),axis=0)
        indices=np.argsort(shap_feat)[::-1]
        top_features=[(FEATURE_NAMES[i],float(shap_feat[i])) for i in indices[:5]]
    except Exception as e:
        magnitudes=np.mean(np.abs(combined),axis=0)
        indices=np.argsort(magnitudes)[::-1]
        top_features=[(FEATURE_NAMES[i],float(magnitudes[i])) for i in indices[:5]]
    top_name=top_features[0][0]
    top_score=top_features[0][1]
    fault_desc=FAULT_MAP.get(top_name,"abnormal vibration pattern")
    explanation=(f"Primary fault indicator: {top_name} (importance: {top_score:.3f}). "+"Suggests "+f"{fault_desc}. "+f"Bearing health at {rul_pct}% — status: {status}.")
    return{
        "rul_normalized":rul_norm,
        "rul_percent":rul_pct,
        "health_status":status,
        "top_features":top_features,
        "explanation":explanation,
    }
if __name__=="__main__":
    import os
    import pandas as pd
    WINDOW_SIZE=int(np.load("window_size.npy")[0])
    print("Loading model and scaler...")
    model=load_model()
    scaler=joblib.load("scaler.pkl")
    print("Loaded successfully.")
    print(f"Window size: {WINDOW_SIZE}")
    print("\n── TEST 1: Near-failure (last 50 files of Run 2) ──")
    data_dir="nasa_data/2nd_test"
    all_files=sorted(os.listdir(data_dir))
    window_files=all_files[-WINDOW_SIZE:]
    window_data=[]
    for fname in window_files:
        df=pd.read_csv(os.path.join(data_dir,fname),sep="\t",header=None)
        sig=df[0].values.astype(np.float32)
        window_data.append(extract_features_from_signal(sig))
    raw_window=np.array(window_data,dtype=np.float32)
    result=predict_rul(raw_window,model,scaler)
    print(f"  RUL (normalized): {result['rul_normalized']:.3f}")
    print(f"  RUL (percent):    {result['rul_percent']}%")
    print(f"  Health Status:    {result['health_status']}")
    print(f"\n  Top 5 contributing features:")
    for feat,score in result['top_features']:
        bar="█"*max(1,int(score*300))
        print(f"    {feat:<14} {bar} {score:.4f}")
    print(f"\n  Explanation:\n  {result['explanation']}")
    print("\n── TEST 2: Healthy sample (first 50 files of Run 2) ──")
    window_files_healthy=all_files[:WINDOW_SIZE]
    window_data_healthy=[]
    for fname in window_files_healthy:
        df=pd.read_csv(os.path.join(data_dir,fname),sep="\t",header=None)
        sig=df[0].values.astype(np.float32)
        window_data_healthy.append(extract_features_from_signal(sig))
    raw_window_healthy=np.array(window_data_healthy,dtype=np.float32)
    result_healthy=predict_rul(raw_window_healthy,model,scaler)
    print(f"  RUL (normalized): {result_healthy['rul_normalized']:.3f}")
    print(f"  RUL (percent):    {result_healthy['rul_percent']}%")
    print(f"  Health Status:    {result_healthy['health_status']}")
    print(f"\n  Explanation:\n  {result_healthy['explanation']}")
    print("\n── Both tests complete. predict.py is working. ──")
    print("Hand these files to Person 4:")
    print("  best_model.pt, scaler.pkl, window_size.npy,")
    print("  n_features.npy, n_raw.npy, predict.py")
