import time
import yfinance as yf
import pandas as pd
from scipy.stats import ttest_1samp

class LoadData():
    """
    This class is able to pull historical market data for a given ticker between the specified start and end dates we want.
    It uses the yfinance library to download daily OHLCV data and provides a cached dataframe for analysis.
    """
    def __init__(self, start_date, end_date, ticker) -> None:
        """
        'start_date' is the start date (YYYY-MM-DD) for getting historical data.
        'end_date' is the end date (YYYY-MM-DD) for getting historical data. If None, it defaults to current time.
        'ticker' is the ticker symbol of the stock or asset to be fetched.
        """
        self._dataframe_stock = None
        self.start_date = start_date
        #If there is no end date is specifically stated, use default.
        if end_date == None:
            self.end_date = time.time()
        else:
            self.end_date = end_date
        self.ticker = ticker

    @property
    def dataframe_stock(self):
        """
        This funcrion is the property that returns the OHLCV dataframe for the specified ticker.
        If it's not yet downloaded, it fetches the data from yfinance and caches it.
        It returns a pd.DataFrame with the [Open, High, Low, Close, Adj Close, Volume].
        """
        if self._dataframe_stock == None:
            #Download daily data from yfinance.
            self._dataframe_stock = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval='1d')
            self._dataframe_stock.columns = self._dataframe_stock.columns.get_level_values(0)
        return self._dataframe_stock



class VolumeBreakoutDetect():
    """
    This class detects volume-based breakout signals from historical stock data.
    It calculates a rolling median volume and flags days when the volume exceeds a threshold multiplier.
    It then calculates returns over various holding periods post-breakout days to evaluate performance.
    """
    def __init__(self, start_date, end_date, ticker, threshold) -> None:
        """
        'start_date' is the start date (YYYY-MM-DD) for fetching historical data.
        'end_date' is the end date (YYYY-MM-DD) for fetching historical data.
        'ticker' is the ticker symbol of the asset to analyze.
        'threshold' is the multiplier for the median volume. If day's volume > threshold * median_volume, it's considered a breakout!
        """
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.volume_threshold = threshold
        self._dataframe_stock = None

    @property
    def dataframe_stock(self):
        """
        This is a Lazy-loaded property that returns the OHLCV DataFrame for the given ticker and date range.
        It uses the LoadData class to fetch and cache the data.
        Returns a pd.DataFrame with historical OHLCV data.
        """
        if self._dataframe_stock is None:
            #Instantiate LoadData and fetch the data only when needed for storage efficiancy.
            self._dataframe_stock = LoadData(self.start_date, self.end_date, self.ticker).dataframe_stock
        return self._dataframe_stock

    def _get_median_rolling_data(self):
        """
        This function is to calculate a 22-day rolling median of the trading volume and add it as a new column in the df.
        """
        dataframe_stock = self.dataframe_stock.copy()
        #Calculate the rolling median volume over 22 days.
        dataframe_stock['Median Volume'] = dataframe_stock['Volume'].rolling(window=22).median()
        return dataframe_stock

    def generate_excel(self, output_file):
        """
        This function helps us generates an Excel report containing parameters used for the analysis, 
        statistics of average returns over multiple holding periods post-breakout, and detailed information on each breakout day.

        'output_file' is the file path where the Excel output should be saved.
    
        Returns 'breakout_days pd.DataFrame that is containing rows for each identified breakout day along with 
        corresponding returns. and 'stats_df'that is containing statistical summaries of returns for various holding periods post-breakout.
        """
        dataframe_stock = self._get_median_rolling_data()

        #Mark the volume breakout days.
        dataframe_stock['Volume Breakout'] = dataframe_stock['Volume'] > self.volume_threshold * dataframe_stock['Median Volume']

        #Calculate the daily return percentage.
        dataframe_stock['Daily Return (%)'] = dataframe_stock['Close'].pct_change() * 100

        #State the holding periods to analyze returns after breakouts, this we can chnage at any time in case we want to see something else.
        return_periods = [1, 2, 5, 10, 15, 20, 25, 30]

        #Calculate the returns for various forward periods.
        for period in return_periods:
            dataframe_stock[f'{period}D Return (%)'] = dataframe_stock['Close'].pct_change(periods=period) * 100

        #Filter results to get only the breakout days.
        breakout_days = dataframe_stock[dataframe_stock['Volume Breakout']].copy()

        #Convert index to string dates and reset index.
        breakout_days['Date'] = breakout_days.index.astype(str) if breakout_days.index.name == 'Date' else breakout_days.index.map(str)
        breakout_days.reset_index(drop=True, inplace=True)

        #Store parameters used.
        parameters = {
            "Volume Threshold (%)": self.volume_threshold * 100,
            "Ticker": self.ticker,
            "Start Date": self.start_date,
            "End Date": self.end_date,
        }

        #Calculate statistics for each holding period.
        stats = []
        for period in return_periods:
            returns = breakout_days[f'{period}D Return (%)'].dropna()
            avg_return = returns.mean()
            
            #Calculate T-Stat against null hypothesis of zero mean return.
            t_stat = ttest_1samp(returns, 0).statistic if len(returns) > 1 else float('nan')
            
            #Calculate the Max drawdown. Here it's defined as (min_return / avg_return). 
            max_drawdown = (returns.min() / avg_return) if avg_return != 0 else float('nan')
            
            #Calculate the profitability metrics.
            num_profitable = (returns > 0).sum()
            total_count = len(returns)
            pop = num_profitable / total_count if total_count > 0 else 0

            stats.append({
                "Period": f'{period}D',
                "Average Return (%)": avg_return,
                "T-Stat": t_stat,
                "Max Drawdown": max_drawdown,
                "Number of Times Profitable": num_profitable,
                "Probability of Profit (POP)": pop,
                "Total Count": total_count
            })

        #Send stats to DataFrame and set period as index.
        stats_df = pd.DataFrame(stats)
        stats_df.set_index("Period", inplace=True)

        #Send parameters to DataFrame.
        parameters_df = pd.DataFrame.from_dict(parameters, orient='index', columns=['Value'])


        #Sheet 1: Parameters
        #Sheet 2: Average Returns
        #Sheet 3: Detailed Breakout Days
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            parameters_df.to_excel(writer, sheet_name="Parameters")
            stats_df.to_excel(writer, sheet_name="Average Returns")
            breakout_days.to_excel(writer, sheet_name="Breakout Days")

        return breakout_days, stats_df



 