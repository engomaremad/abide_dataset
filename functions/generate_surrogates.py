#TODO generate surrogates as described in
# "Dynamic fluctuations coincide with periods of high and low modularity in resting-state functional brain network" and
#Surrogate data for hypothesis testing of physical systems
import numpy as np
def rank_match(tc,sur):
    #just to make sure no values are repeated in the tc
    tc = tc + np.random.rand(np.size(tc))*0.01
    sorted_tc = np.sort(tc)
    sorted_sur = np.sort(sur)
    ret_sur = np.zeros(np.shape(tc))
    for i in range(np.shape(tc)[0]):
        rank = np.where(sorted_tc==tc[i])
        for j in range(len(rank[0])):
            ret_sur[i+j] = sorted_sur[rank[0][j]]
    return ret_sur
def randomize_phase(sig):
    #This function adds random phase to the input signal
    # The random phase is only added to the first half as the FFT is symmetric
    # The second half is a negative flipped version of the first half
    # In this way, the ifft is real
    # The funtion handles both even and odd signal length cases
    f = np.fft.fft(sig)
    phase = np.angle(f)
    mag = np.abs(f)
    half = np.int(np.floor(np.size(phase)/2))
    phase_rand = np.random.rand(half)*2*np.pi
    if np.size(sig)%2 != 0:
        phase[1:half+1] = np.add(phase[1:half+1],phase_rand)
        phase[half+1:]= -1 *np.transpose(np.flip(phase[1:half+1]))
    else:
        phase[1:half] = np.add(phase[1:half], phase_rand[:half-1])
        phase[half+1:] = -1 * np.transpose(np.flip(phase[1:half]))
    real = np.multiply(mag,np.cos(phase))
    imag = np.multiply(mag, np.sin(phase))*1j
    signal = np.add(real,imag)
    signal = np.fft.ifft(signal)
    signal = np.real(signal)
    return signal
    x=0
def gen_surrogate(tc,n):
    suurogates = np.zeros([np.shape(tc)[0],n])
    for i in range(n):
        tmp = np.random.rand(np.shape(tc)[0])
        tmp = rank_match(tc,tmp)
        tmp = randomize_phase(tmp)
        tmp = rank_match(tc,tmp)
        suurogates[:,i] = tmp
    return suurogates


