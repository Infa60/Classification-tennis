from scipy import signal


def LP_filter(x,f_coupure,f_echantillonnage):
    # filtre passe-bas, no phase shift
    b, a = signal.butter(2, f_coupure, btype='low', analog=False, output='ba', fs=f_echantillonnage)
    y = signal.filtfilt(b, a, x, padlen=150)
    return y

