import numpy as np
import yfinance as yf
import scipy.optimize as sco

class EfficientFontier:
    def __init__(self, tickers, start_date, end_date, actions):

        self.tickers    = tickers
        self.start_date = start_date
        self.end_date   = end_date
        self.actions    = actions

        self.etf        = yf.Tickers(tickers)
        self.etf_data   = None

        self.returns    = None           # 수익률
        self.expected_returns   = None   # 기대 수익률
        self.covariance_matrix  = None   # 공분산 행렬

        self.RETURN_INDEX           = 0
        self.VOLATILITYU_INDEX      = 1
        self.SHARP_INDEX            = 2
        self.WORKING_DAYS_PER_YEAR  = 252


    def get_etf_data(self, drop_col_list):
        '''
        drop_col_list can include ['Open', 'Close', 'High', 'Low', 'Volume']
        '''
        self.etf_data = self.etf.history(start = self.start_date,
                                        end = self.end_date,
                                        actions = self.actions)
        
        self.etf_data.drop(drop_col_list, inplace=True, axis=1)
        self.etf_data = self.etf_data.droplevel(0, axis=1)


    def get_optimize_model_inputs(self):
        '''
        일일 기준
        '''
        self.returns            = self.etf_data.pct_change().fillna(0)
        self.expected_returns   = self.returns.mean()*self.WORKING_DAYS_PER_YEAR
        self.covariance_matrix  = self.returns.cov()*self.WORKING_DAYS_PER_YEAR

    
    def get_statistics(self, weights):
        '''
        포트폴리오 통계치 계산 함수
        return : 
            - p_rets        : 수익률
            - p_vols        : 변동성 
            - p_rets/p_vols : 샤프 비율 
        '''

        # 투자 가중치
        weights = np.array(weights)

        # 포트폴리오 수익률
        p_rets = np.sum(self.expected_returns*weights)
        p_vols = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))

        return np.array([p_rets, p_vols, p_rets/p_vols])
    

    def get_obj_function(self, weights):
        '''
        목적함수를 Return하는 함수
        return :
            - 목적함수(포트폴리오 변동성)
        '''
        return self.get_statistics(weights)[self.VOLATILITYU_INDEX]
    

    def optimize_portpolio_about_returns(self, drop_col_list, min_return, max_return, sample_num):
        '''
        수익률 수준별 포트폴리오 최적화 수행
        '''
        self.get_etf_data(drop_col_list)
        self.get_optimize_model_inputs()

        t_rets = np.linspace(min_return, max_return, sample_num)
        t_vols = list()

        weights = np.random.random(len(self.tickers))
        weights /= np.sum(weights)

        for t_ret in t_rets:
            
            # 투자 가중치 초기값 = 동일 가중
            # 초기에는 모든 포트폴리오에 대해 동일한 가중치를 부여함 (= 1 / 포트폴리오 수)
            init_guess = np.repeat(1/len(self.tickers), len(self.tickers))

            # 제약조건 (포트폴리오 목표 수익률, 현금 보유 비중 = 0)
            constraints = ({'type':'eq', 'fun':lambda x:self.get_statistics(x)[0]-t_ret}, # 목표 수익률
                           {'type':'eq', 'fun':lambda x:np.sum(x)-1})                     # 현금 보유 비중
            
            # 자산별 경계조건 (숏 포지션 불가능)
            bounds = tuple((0.0, 1.0) for x in weights)

            # 최적화 알고리즘 수행
            results = sco.minimize(self.get_obj_function,
                                   init_guess,
                                   method='SLSQP',
                                   bounds=bounds,
                                   constraints=constraints)
            
            t_vols.append(results['fun'])

        return np.array(t_rets), np.array(t_vols)


    def get_left_bound_data(self, t_rets, t_vols):
        return t_rets[np.argmin(t_vols):], t_vols[np.argmin(t_vols):]
