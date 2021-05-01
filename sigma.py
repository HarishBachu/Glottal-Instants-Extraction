#a python file to import the SIGMA algorithm for glottal instants extraction
#pass as arguments (egg_signal,sample_rate_of_egg) to get_glottal,
#output is locations of gci and goi within given egg signal

import numpy as np
import librosa
import pywt
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt 

def energy_weighted_group_delay(signal,sr_egg):
    
    R = int(0.0025*sr_egg)
    group_delay = np.zeros((len(signal) - R + 1, R))
    mult = np.arange(R)
    
    for i in range(len(signal) - R + 1):
        X = np.fft.fft(signal[i:i+R])
        X_r = np.fft.fft(signal[i:i+R]*mult)
        group_delay[i] = np.real(X_r/X)

    ewgd = np.zeros(len(signal) - R + 1)
    
    for i in range(len(signal) - R + 1):
        X_sq = np.square(np.abs(np.fft.fft(signal[i:i+R])))
        ewgd[i] = np.sum(X_sq*group_delay[i])/np.sum(X_sq)
        
    ewgd -= (R - 1)/2.
    
    return ewgd

def zero_crossing_pos2neg(signal):
    rectified = signal > 0
    return np.where(np.logical_and(np.logical_xor(rectified[:-1],rectified[1:]),rectified[:-1]))[0]

def get_cluster(zc_pos2neg,ewgd,p_positive,sr_egg):
    feature_mat = np.zeros((len(zc_pos2neg),3))
    R = int(0.0025*sr_egg)
        
    for i in range(len(zc_pos2neg)):
        ewgd_window = ewgd[zc_pos2neg[i] - int((R-1)/2):zc_pos2neg[i] + int((R-1)/2) + 1]
        l_ = len(ewgd_window)
        ideal = np.arange(l_//2, -(l_//2 + 1), -1)
        feature_mat[i,0] = np.sqrt(np.mean(np.square(ewgd_window - ideal[:len(ewgd_window)])))
        p_pos_window = p_positive[zc_pos2neg[i] - int((R-1)/2):zc_pos2neg[i] + int((R-1)/2) + 1]
        feature_mat[i,1] = np.amax(p_pos_window**(1/3.))
        feature_mat[i,2] = np.sum(p_pos_window**(1/3.))
        
    gmm = GaussianMixture(n_components=2)
    gmm.fit(feature_mat)
    label = gmm.predict(feature_mat)
    
    if np.mean(feature_mat[label == 1,2]) > np.mean(feature_mat[label == 0,2]):
        return zc_pos2neg[label == 1]
    return zc_pos2neg[label == 0]

def swallowing(gci,sr_egg):
    N_max = 0.02 * sr_egg
    diff = gci[1:] - gci[:-1]
    keep = np.zeros(len(gci))
    remove = diff>N_max
    keep[:-1] += remove
    keep[1:] += remove
    return gci[keep == 0]

def goi_post_processing(goi_candidates,gci,sr_egg):
    goi = []
    N_max = 0.02 * sr_egg
    for i in range(len(gci)-1):
        if gci[i+1] - gci[i] < N_max:
            check = 0
            for j in range(gci[i] + int((gci[i+1]-gci[i])*0.1),gci[i] + int((gci[i+1]-gci[i])*0.9)):
                if j in goi_candidates:
                    goi.append(j)
                    check = 1
                    break
            if check == 0:
                goi.append(gci[i] + int((gci[i+1]-gci[i])*0.5))
    return goi 

def get_glottal(egg, sr_egg):
    
    # if len(egg)%8 != 0:
    #     egg = egg[:-(len(egg)%8)]
    
    swt = pywt.swt(egg, wavelet = "bior1.5", level = 3)
    multiscale_product = swt[0][1]*swt[1][1]*swt[2][1]
    
    p_positive = multiscale_product*(multiscale_product>0)
    p_negative = multiscale_product*(multiscale_product<0)
    
    ewgd_gci = energy_weighted_group_delay(p_positive,sr_egg)
    ewgd_gci[np.where(np.isnan(ewgd_gci))] = 0
    
    ewgd_goi = energy_weighted_group_delay(-p_negative,sr_egg)
    ewgd_goi[np.where(np.isnan(ewgd_goi))] = 0
    
    zc_pos2neg_gci = zero_crossing_pos2neg(ewgd_gci)
    zc_pos2neg_goi = zero_crossing_pos2neg(ewgd_goi)
    
    R = int(0.0025*sr_egg)

    for i in range(len(zc_pos2neg_gci)):
        if zc_pos2neg_gci[i] > int((R-1)/2):
            zc_pos2neg_gci = zc_pos2neg_gci[i:]
            break
            
    for i in range(len(zc_pos2neg_goi)):
        if zc_pos2neg_goi[i] > int((R-1)/2):
            zc_pos2neg_goi = zc_pos2neg_goi[i:]
            break
            
    cluster_gci = get_cluster(zc_pos2neg_gci,ewgd_gci,p_positive,sr_egg)
    cluster_goi = get_cluster(zc_pos2neg_goi,ewgd_goi,-p_negative,sr_egg)
    
    gci = swallowing(cluster_gci,sr_egg)
    
    goi = goi_post_processing(cluster_goi,gci,sr_egg)
    
    return gci,goi

def plot(egg,speech,gci,goi, trim = None, title = "Plot"):
    # window_start = 0
    # window_length = 16384
    
    gci_plot = np.zeros(len(egg))
    goi_plot = np.zeros(len(egg))

    gci_plot[gci] = 0.025
    goi_plot[goi] = -0.025

    plt.figure(figsize = (20, 20))
    
    # plt.plot(speech[window_start:window_start+window_length])
    plt.subplot(411)
    plt.plot(speech[:trim])
    plt.title('Speech signal', fontsize = 20)
    # plt.show()
    # plt.plot(egg[window_start:window_start+window_length])
    plt.subplot(412)
    plt.plot(egg[:trim])
    plt.title('EGG', fontsize = 20)
    # plt.show()
    # plt.plot(egg[window_start+1:window_start+window_length+1] - egg[window_start:window_start+window_length])
    plt.subplot(413)
    plt.plot(np.diff(egg)[:trim])
    plt.plot(gci_plot[:trim])
    plt.title('dEGG GCI', fontsize = 20)
    # plt.show()
    # plt.plot(gci_plot[window_start:window_start+window_length],label='gci')
    plt.subplot(414)
    plt.plot(np.diff(egg)[:trim])
    plt.plot(goi_plot[:trim])
    # plt.plot(goi_plot[window_start:window_start+window_length],label='goi')
    # plt.plot(goi_plot[:trim], label = 'goi')
    plt.title('dEGG GOI', fontsize = 20)
    # plt.legend()

    plt.subplots_adjust(top = 0.92)
    plt.suptitle(title, fontsize = 30)
    plt.show()

def main():
    pass
    
if __name__ == "__main__":
    main()