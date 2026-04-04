import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq
import multiprocessing as mp
RUNS=[{"path":"nasa_data/1st_test","cols":8},{"path":"nasa_data/2nd_test","cols":4},{"path":"nasa_data/3rd_test/4th_test/txt","cols":4}]
FS,N=20000,20480
BPFO=161.1
BPFI=236.4
BAND=15
FREQS=fftfreq(N,d=1.0/FS)[:N//2]
BPFO_MASK=(FREQS>=BPFO-BAND)&(FREQS<=BPFO+BAND)
BPFI_MASK=(FREQS>=BPFI-BAND)&(FREQS<=BPFI+BAND)
def get_features(signal):
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
    return[rms,kurt,skw,peak,crest,mav,bpfo_ratio,bpfi_ratio]
def process_file(args):
    fpath,run_cols=args
    try:
        df=pd.read_csv(fpath,sep='\t',header=None)
        target_cols=[0,2,4,6]if run_cols==8 else[0,1,2,3]
        feat_row=[]
        for c in target_cols:
            feat_row.extend(get_features(df[c].values))
        return feat_row
    except:
        return None
if __name__=="__main__":
    mp.freeze_support()
    all_X=[]
    for run in RUNS:
        folder=run['path']
        files=sorted(os.listdir(folder))
        print(f"Extracting {folder} ({len(files)} files)...")
        with mp.Pool(mp.cpu_count()) as pool:
            results=pool.map(process_file,[(os.path.join(folder,f),run['cols']) for f in files])
        run_data=np.array([r for r in results if r is not None],dtype=np.float32)
        all_X.append(run_data)
        print(f"  Done: {run_data.shape} — 8 features x 4 bearings = 32 per file")
    np.save("X_raw_features.npy",np.array(all_X,dtype=object))
    print("\nSaved X_raw_features.npy")
    print("Features per file: 32 (8 per bearing x 4 bearings)")
