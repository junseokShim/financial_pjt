from src.efficient_frontier import EfficientFontier

import matplotlib.pyplot as plt

def day3_main():
    ## defines 
    tickers = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
    start_date = '2011-01-01'
    end_date = '2024-02-28'
    actions = False

    efficient_frontier = EfficientFontier(
        tickers = tickers,
        start_date = start_date,
        end_date = end_date,
        actions = actions
    )

    drop_col_list = ['Open','High','Low','Volume']
    min_return = 0.09
    max_return = 0.20
    sample_num = 50

    target_returns, target_volatilities = efficient_frontier.optimize_portpolio_about_returns(
        drop_col_list = drop_col_list,
        min_return = min_return,
        max_return = max_return,
        sample_num = sample_num
    )

    e_returns, e_volatilities = efficient_frontier.get_left_bound_data(
        t_rets = target_returns, 
        t_vols = target_volatilities
    )

    ## draw plots
    plt.figure(figsize=(8, 8))
    plt.scatter(target_volatilities, target_returns, c=target_returns/target_volatilities, marker='x')
    plt.plot(e_volatilities, e_returns, lw=1.0)
    plt.grid(True)

    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.show()


if __name__=="__main__":
    day3_main()