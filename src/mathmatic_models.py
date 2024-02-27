import numpy as np
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt

def stock_movement(T):
    '''
    기초자산 변화 함수 모델링
    '''
    N = T*252
    dt = T/N
    dS = np.random.randn(N)*np.sqrt(dt)
    S = np.cumsum(dS)
    S = S - S.min()
    
    return S


def future_price(S, r, d):
    '''
    선물 가격 예측 모델링
    [inputs]
        - S : 주식 가격
    '''
    F = []
    for time, S_day in enumerate(S):
        t_ramain = S.shape[0] - (time +1)
        F_day = S_day*np.exp((r-d)*t_ramain)
        F.append(F_day)
        
    return F


def option_price(S, K, r, sigma, option_type):
    '''
    블랙-숄즈 방정식
    구매가, 현재가, 기간, 이자율, 유동성으로 예상 가격 계산
    '''
    OV = []
    for time, S_day in enumerate(S):
        t_remain = S.shape[0] - (time+1)
        d1 = (np.log(S_day/K) + (r+0.5*sigma**2) + t_remain)/(sigma*np.sqrt(t_remain))
        d2 = d1 - sigma + np.sqrt(t_remain)

        if option_type == 'call':
            OV_day = S_day*stat.norm.cdf(d1) - K*np.exp(-r*t_remain)*stat.norm.cdf(d2)
        else:
            OV_day = -S_day*stat.norm.cdf(-d1) + K*np.exp(-r*t_remain)*stat.norm.cdf(-d2)

        OV.append(OV_day)
        
    return OV