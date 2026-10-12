# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import time
# pyrefly: ignore [missing-import]
from scipy import signal
from mediapipe_face import MediaPipeFace
from signal_processing import Signal_processing

class Process(object):
    def __init__(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.buffer_size = 100
        self.times = []
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
        self.peaks = []
        self.fu = MediaPipeFace()
        self.sp = Signal_processing()

        self._hamming = np.hamming(self.buffer_size)
        # Zero-pad the FFT for finer frequency resolution. With buffer_size=100
        # and fps~30, native bin width is ~18 BPM — coarse enough that argmax
        # can miss the true HR by a bin (e.g. read 63 when truth is 75). Zero
        # padding to 1024 points interpolates the spectrum so argmax lands on
        # a ~3.5 BPM grid. No accuracy compromise: we're not inventing signal,
        # just reading the existing DFT on a denser grid.
        self._n_fft = 1024
        self._freq_axis = np.arange(self._n_fft // 2 + 1)
        self._bp_low = 0.8
        self._bp_high = 3.0
        self._bp_order = 3
        self._last_fps_for_bp = None
        self._bp_b = None
        self._bp_a = None

        # Signal-quality gate. The FFT argmax in a noise-only buffer is still
        # "a peak" — without an SNR check the displayed BPM is a random number
        # in [50, 180]. Require the peak to stand clearly above the in-band
        # median before trusting it, and discard the first few readings after
        # the buffer first fills (the tail of signal warm-up).
        self._snr_threshold = 4.0
        self._bpm_valid = False
        self.bpm_snr = 0.0
        self._warmup_frames = 10
        self._frames_since_full = 0

        #self.red = np.zeros((256,256,3),np.uint8)
        
    def extractColor(self, frame):
        
        #r = np.mean(frame[:,:,0])
        g = np.mean(frame[:,:,1])
        #b = np.mean(frame[:,:,2])
        #return r, g, b
        return g
        
    def run(self):
        # frame, face_frame, ROI1, ROI2, status, mask = self.fd.face_detect(self.frame_in)
        
        frame = self.frame_in
        ret_process = self.fu.no_age_gender_face_process(frame, "5")
        if ret_process is None:
            return False
        rects, face, shape, aligned_face, aligned_shape = ret_process

        r = rects[0]
        x, y, w, h = r.left(), r.top(), r.width(), r.height()
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        if(len(aligned_shape)==68):
            cv2.rectangle(aligned_face,(aligned_shape[54][0], aligned_shape[29][1]), #draw rectangle on right and left cheeks
                    (aligned_shape[12][0],aligned_shape[33][1]), (0,255,0), 0)
            cv2.rectangle(aligned_face, (aligned_shape[4][0], aligned_shape[29][1]), 
                    (aligned_shape[48][0],aligned_shape[33][1]), (0,255,0), 0)
        else:
            cv2.rectangle(aligned_face, (aligned_shape[0][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                        (aligned_shape[1][0],aligned_shape[4][1]), (0,255,0), 0)
            
            cv2.rectangle(aligned_face, (aligned_shape[2][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                        (aligned_shape[3][0],aligned_shape[4][1]), (0,255,0), 0)
        
        for (x, y) in aligned_shape: 
            cv2.circle(aligned_face, (x, y), 1, (0, 0, 255), -1)


        ROIs = self.fu.ROI_extraction(aligned_face, aligned_shape)
        # Landmark jitter near the buffer edge can produce a degenerate
        # (zero-area) cheek slice whose mean is NaN; skip this frame rather
        # than poison the data buffer, which makes detrend raise.
        if any(roi.size == 0 for roi in ROIs):
            return False
        green_val = self.sp.extract_color(ROIs)
        if not np.isfinite(green_val):
            return False

        self.frame_out = frame
        self.frame_ROI = aligned_face
        
        # g1 = self.extractColor(ROI1)
        # g2 = self.extractColor(ROI2)
        #g3 = self.extractColor(ROI3)
        
        L = len(self.data_buffer)
        
        #calculate average green value of 2 ROIs
        #r = (r1+r2)/2
        #g = (g1+g2)/2
        #b = (b1+b2)/2

        g = green_val
        
        if(abs(g-np.mean(self.data_buffer))>10 and L>99): #remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
            g = self.data_buffer[-1]
        
        self.times.append(time.time() - self.t0)
        self.data_buffer.append(g)

        #only process in a fixed-size buffer
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size
            
        processed = np.array(self.data_buffer)
        
        # start calculating after the first 10 frames
        if L == self.buffer_size:
            
            # N samples span N-1 sample intervals, so the true sample rate
            # is (L-1)/duration, not L/duration. Previous off-by-one inflated
            # fps by ~1% and shifted every computed BPM by the same factor.
            dt = self.times[-1] - self.times[0]
            if dt <= 0:
                dt = float(L - 1) / 30.0
            self.fps = float(L - 1) / dt
            even_times = np.linspace(self.times[0], self.times[-1], L)

            processed = signal.detrend(processed)#detrend the signal to avoid interference of light change
            interpolated = np.interp(even_times, self.times, processed) #interpolation by 1
            interpolated = self._hamming * interpolated#make the signal become more periodic (advoid spectral leakage)
            #norm = (interpolated - np.mean(interpolated))/np.std(interpolated)#normalization
            norm = interpolated/np.linalg.norm(interpolated)
            # Zero-pad to self._n_fft for finer spectral resolution.
            raw = np.fft.rfft(norm * 30, n=self._n_fft)

            self.freqs = float(self.fps) / self._n_fft * self._freq_axis
            freqs = 60. * self.freqs
            
            # idx_remove = np.where((freqs < 50) & (freqs > 180))
            # raw[idx_remove] = 0
            
            self.fft = np.abs(raw)**2#get amplitude spectrum
        
            idx = np.where((freqs > 50) & (freqs < 180))[0]#the range of frequency that HR is supposed to be within
            if len(idx) > 0 and len(self.fft[idx]) > 0:
                pruned = self.fft[idx]
                pfreq = freqs[idx]

                self.freqs = pfreq
                self.fft = pruned

                idx2 = np.argmax(pruned)#max in the range can be HR

                # SNR = peak / median-in-band. Median is robust to a second peak
                # (e.g. harmonic). A clean cardiac signal typically scores >5;
                # pure noise sits around 1-2.
                peak_val = pruned[idx2]
                band_median = np.median(pruned)
                self.bpm_snr = float(peak_val / band_median) if band_median > 0 else 0.0

                self._frames_since_full += 1
                warm = self._frames_since_full >= self._warmup_frames
                self._bpm_valid = warm and (self.bpm_snr >= self._snr_threshold)

                self.bpm = self.freqs[idx2]
                if self._bpm_valid:
                    self.bpms.append(self.bpm)

            processed = self.butter_bandpass_filter(processed,0.8,3,self.fps,order = 3)
            #ifft = np.fft.irfft(raw)
        self.samples = processed # multiply the signal with 5 for easier to see in the plot
        #TODO: find peaks to draw HR-like signal.
        
        # if(mask.shape[0]!=10): 
        #     out = np.zeros_like(aligned_face)
        #     mask = mask.astype(np.bool)
        #     out[mask] = aligned_face[mask]
        #     if(processed[-1]>np.mean(processed)):
        #         out[mask,2] = 180 + processed[-1]*10
        #     aligned_face[mask] = out[mask]
            
            
        #cv2.imshow("face", face_frame)
        #out = cv2.add(face_frame,out)
        # else:
            # cv2.imshow("face", face_frame)
        return True
    
    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = []
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
        self._last_fps_for_bp = None
        self._bp_b = None
        self._bp_a = None
        self._bpm_valid = False
        self.bpm_snr = 0.0
        self._frames_since_full = 0
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        if fs <= 0:
            fs = 30.0
        nyq = 0.5 * fs
        low = max(0.01, min(0.99, lowcut / nyq))
        high = max(0.01, min(0.99, highcut / nyq))
        if low >= high:
            low, high = 0.1, 0.9
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a


    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        if (self._bp_b is None
                or self._last_fps_for_bp is None
                or abs(fs - self._last_fps_for_bp) > 0.5
                or lowcut != self._bp_low
                or highcut != self._bp_high
                or order != self._bp_order):
            self._bp_b, self._bp_a = self.butter_bandpass(lowcut, highcut, fs, order=order)
            self._last_fps_for_bp = fs
            self._bp_low = lowcut
            self._bp_high = highcut
            self._bp_order = order
        return signal.lfilter(self._bp_b, self._bp_a, data)


if __name__ == "__main__":
    print("Testing Process...")
    proc = Process()
    print("[OK] Process instance created.")
    proc.frame_in = np.zeros((480, 640, 3), dtype=np.uint8)
    ok = proc.run()
    print(f"[OK] Run on blank frame returned {ok} without errors.")
    if hasattr(proc.fu, "close"):
        proc.fu.close()
    print("[OK] Process self-test passed.")
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
