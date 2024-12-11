import time
import yfinance as yf
import pandas as pd
from scipy.stats import ttest_1samp
import gspread
from oauth2client.service_account import ServiceAccountCredentials

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
            self._dataframe_stock = LoadData(self.start_date, self.end_date, self.ticker).dataframe_stock
        return self._dataframe_stock

    def _get_median_rolling_data(self):
        """
        This function is to calculate a 22-day rolling median of the trading volume and add it as a new column in the df.
        """
        dataframe_stock = self.dataframe_stock.copy()
        dataframe_stock['Median Volume'] = dataframe_stock['Volume'].rolling(window=22).median()
        return dataframe_stock

    def generate_excel(self, output_file):
        """
        Generates an Excel report containing:
        - Parameters used for the analysis.
        - Statistics of average returns over multiple holding periods post-breakout, including Max Return.
        - Detailed information on each breakout day.
        
        'output_file' is the file path where the Excel output should be saved.
        
        Returns:
            - breakout_days: pd.DataFrame containing rows for each identified breakout day along with corresponding returns.
            - stats_df: pd.DataFrame containing statistical summaries of returns for various holding periods post-breakout.
        """
        dataframe_stock = self._get_median_rolling_data()

        dataframe_stock['Volume Breakout'] = dataframe_stock['Volume'] > self.volume_threshold * dataframe_stock['Median Volume']

        dataframe_stock['Daily Return (%)'] = dataframe_stock['Close'].pct_change() * 100

        dataframe_stock['Next Day Close'] = dataframe_stock['Close'].shift(-1)

        return_periods = [1, 2, 5, 10, 15, 20, 25, 30]

        for period in return_periods:
            if period == 1:
                dataframe_stock[f'{period}D Return (%)'] = (
                    dataframe_stock['Next Day Close'] - dataframe_stock['Close']
                ) / dataframe_stock['Close'] * 100
            else:
                dataframe_stock[f'{period}D Return (%)'] = (
                    dataframe_stock['Close'].shift(-period) - dataframe_stock['Close']
                ) / dataframe_stock['Close'] * 100

        breakout_days = dataframe_stock[dataframe_stock['Volume Breakout']].copy()

        breakout_days['Date'] = breakout_days.index.astype(str) if breakout_days.index.name == 'Date' else breakout_days.index.map(str)
        breakout_days.reset_index(drop=True, inplace=True)

        positive_breakout_days = breakout_days[breakout_days['Daily Return (%)'] > 0].copy()

        def calculate_stats(data):
            stats = []
            for period in return_periods:
                returns = data[f'{period}D Return (%)'].dropna()
                avg_return = returns.mean()

                t_stat = ttest_1samp(returns, 0).statistic if len(returns) > 1 else float('nan')

                max_return = returns.max() if len(returns) > 0 else float('nan')

                max_drawdown = (returns.min() / avg_return) if avg_return != 0 else float('nan')

                num_profitable = (returns > 0).sum()
                total_count = len(returns)
                pop = num_profitable / total_count if total_count > 0 else 0

                stats.append({
                    "Period": f'{period}D',
                    "Average Return (%)": avg_return,
                    "Max Return (%)": max_return,
                    "T-Stat": t_stat,
                    "Max Drawdown": max_drawdown,
                    "Number of Times Profitable": num_profitable,
                    "Probability of Profit (POP)": pop,
                    "Total Count": total_count
                })
            return pd.DataFrame(stats).set_index("Period")

        stats_df = calculate_stats(breakout_days)
        positive_stats_df = calculate_stats(positive_breakout_days)

        parameters = {
            "Volume Threshold (%)": self.volume_threshold * 100,
            "Ticker": self.ticker,
            "Start Date": self.start_date,
            "End Date": self.end_date,
        }
        parameters_df = pd.DataFrame.from_dict(parameters, orient='index', columns=['Value'])

        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            parameters_df.to_excel(writer, sheet_name="Parameters")
            
            stats_df.to_excel(writer, sheet_name="Average Returns", startrow=2, startcol=0)
            positive_stats_df.to_excel(writer, sheet_name="Average Returns", startrow=len(stats_df) + 8, startcol=0)
            
            worksheet = writer.sheets["Average Returns"]
            worksheet.write(0, 0, "Average Returns")
            worksheet.write(len(stats_df) + 6, 0, "Positive Average Returns")
            
            breakout_days.to_excel(writer, sheet_name="Breakout Days")

        return breakout_days, stats_df

    def write_to_google_sheets(self, sheet_name):
        """
        Writes the analysis data to a Google Sheets document.
        - The data includes: Parameters, Average Returns, and Positive Average Returns.

        'sheet_name': Name of the Google Sheet to create or update.

        Returns:
            - breakout_days: pd.DataFrame containing rows for each identified breakout day.
            - stats_df: pd.DataFrame containing statistics for all breakout days.
            - positive_stats_df: pd.DataFrame containing statistics for breakout days with positive daily returns.
        """
        dataframe_stock = self._get_median_rolling_data()

        dataframe_stock['Volume Breakout'] = dataframe_stock['Volume'] > self.volume_threshold * dataframe_stock['Median Volume']

        dataframe_stock['Daily Return (%)'] = dataframe_stock['Close'].pct_change() * 100

        dataframe_stock['Next Day Close'] = dataframe_stock['Close'].shift(-1)

        return_periods = [1, 2, 5, 10, 15, 20, 25, 30]

        for period in return_periods:
            if period == 1:
                dataframe_stock[f'{period}D Return (%)'] = (
                    dataframe_stock['Next Day Close'] - dataframe_stock['Close']
                ) / dataframe_stock['Close'] * 100
            else:
                dataframe_stock[f'{period}D Return (%)'] = (
                    dataframe_stock['Close'].shift(-period) - dataframe_stock['Next Day Close']
                ) / dataframe_stock['Next Day Close'] * 100

        breakout_days = dataframe_stock[dataframe_stock['Volume Breakout']].copy()
        breakout_days['Date'] = breakout_days.index.astype(str)
        breakout_days.reset_index(drop=True, inplace=True)

        positive_breakout_days = breakout_days[breakout_days['Daily Return (%)'] > 0].copy()

        def calculate_stats(data):
            stats = []
            for period in return_periods:
                returns = data[f'{period}D Return (%)'].dropna()
                avg_return = returns.mean()
                max_return = returns.max()
                t_stat = ttest_1samp(returns, 0).statistic if len(returns) > 1 else float('nan')
                max_drawdown = (returns.min() / avg_return) if avg_return != 0 else float('nan')
                num_profitable = (returns > 0).sum()
                total_count = len(returns)
                pop = num_profitable / total_count if total_count > 0 else 0
                stats.append({
                    "Period": f'{period}D',
                    "Average Return (%)": avg_return,
                    "Max Return (%)": max_return,
                    "T-Stat": t_stat,
                    "Max Drawdown": max_drawdown,
                    "Number of Times Profitable": num_profitable,
                    "Probability of Profit (POP)": pop,
                    "Total Count": total_count
                })
            return pd.DataFrame(stats).set_index("Period")

        stats_df = calculate_stats(breakout_days)
        positive_stats_df = calculate_stats(positive_breakout_days)

        parameters = {
            "Volume Threshold (%)": self.volume_threshold * 100,
            "Ticker": self.ticker,
            "Start Date": self.start_date,
            "End Date": self.end_date,
        }
        parameters_df = pd.DataFrame.from_dict(parameters, orient='index', columns=['Value'])

        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name("path_to_your_credentials.json", scope)
        client = gspread.authorize(credentials)

        try:
            sheet = client.open(sheet_name)
        except gspread.SpreadsheetNotFound:
            sheet = client.create(sheet_name)

        def write_dataframe_to_sheet(dataframe, worksheet_name):
            worksheet = sheet.add_worksheet(title=worksheet_name, rows="1000", cols="50")
            worksheet.clear()
            worksheet.update([dataframe.columns.values.tolist()] + dataframe.values.tolist())

        write_dataframe_to_sheet(parameters_df, "Parameters")
        write_dataframe_to_sheet(stats_df.reset_index(), "Average Returns")
        write_dataframe_to_sheet(positive_stats_df.reset_index(), "Positive Average Returns")
        write_dataframe_to_sheet(breakout_days, "Breakout Days")

        return breakout_days, stats_df, positive_stats_df
