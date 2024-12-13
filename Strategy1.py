import time
import yfinance as yf
import pandas as pd
from scipy.stats import ttest_1samp
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from gspread_formatting import CellFormat, TextFormat, format_cell_range



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


class VolumeBreakoutDetect:
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



    def generate_google_sheet(self, spreadsheet_name):
        """
        Updates or creates a Google Sheet with:
        - Parameters used for the analysis.
        - Statistics of average returns over multiple holding periods post-breakout, including Max Return.
        - Detailed information on each breakout day.
        """
        # Authenticate and connect to Google Sheets
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        credentials = Credentials.from_service_account_file("credentials.json", scopes=scope)
        client = gspread.authorize(credentials)

        try:
            # Open the existing spreadsheet by name
            spreadsheet = client.open(spreadsheet_name)
        except gspread.SpreadsheetNotFound:
            print(f"Spreadsheet '{spreadsheet_name}' not found. Ensure the file exists and is shared with the service account.")
            return

        #Get or create the worksheets.
        worksheet_params = spreadsheet.worksheet("Parameters") if "Parameters" in [ws.title for ws in spreadsheet.worksheets()] else spreadsheet.add_worksheet(title="Parameters", rows="100", cols="20")
        worksheet_stats = spreadsheet.worksheet("Average Returns") if "Average Returns" in [ws.title for ws in spreadsheet.worksheets()] else spreadsheet.add_worksheet(title="Average Returns", rows="100", cols="20")
        worksheet_breakout = spreadsheet.worksheet("Breakout Days") if "Breakout Days" in [ws.title for ws in spreadsheet.worksheets()] else spreadsheet.add_worksheet(title="Breakout Days", rows="100", cols="20")

        #Clear the existing content.
        worksheet_params.clear()
        worksheet_stats.clear()
        worksheet_breakout.clear()

        #Generate the data.
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

        parameters = {
            "Volume Threshold (%)": self.volume_threshold * 100,
            "Ticker": self.ticker,
            "Start Date": self.start_date,
            "End Date": self.end_date,
        }
        parameters_df = pd.DataFrame.from_dict(parameters, orient='index', columns=['Value'])

        peak_returns = []
        for _, breakout in breakout_days.iterrows():
            peak_row = {}
            breakout_index = dataframe_stock.index.get_loc(pd.to_datetime(breakout['Date']))
            for period in return_periods:
                end_index = breakout_index + period
                if end_index > len(dataframe_stock) - 1:
                    peak_row[f'{period}D Peak Return (%)'] = float('nan')
                    continue

                sliced_data = dataframe_stock.iloc[breakout_index:end_index + 1]
                if sliced_data.empty or 'Close' not in sliced_data:
                    peak_row[f'{period}D Peak Return (%)'] = float('nan')
                    continue

                sub_period_returns = []
                for i in range(1, len(sliced_data)):
                    initial_close = sliced_data['Close'].iloc[0]
                    later_close = sliced_data['Close'].iloc[i]
                    sub_period_returns.append(((later_close - initial_close) / initial_close) * 100)

                peak_row[f'{period}D Peak Return (%)'] = max(sub_period_returns) if sub_period_returns else float('nan')

            peak_returns.append(peak_row)

        peak_returns_df = pd.DataFrame(peak_returns, index=breakout_days.index)
        breakout_days = pd.concat([breakout_days, peak_returns_df], axis=1)
        positive_peak_returns_df = peak_returns_df.loc[positive_breakout_days.index]
        positive_breakout_days = pd.concat([positive_breakout_days, positive_peak_returns_df], axis=1)


        def calculate_stats(data, include_peak=False):
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

                row = {
                    "Period": f'{period}D',
                    "Average Return (%)": avg_return,
                    "Max Return (%)": max_return,
                    "T-Stat": t_stat,
                    "Max Drawdown": max_drawdown,
                    "Number of Times Profitable": num_profitable,
                    "Probability of Profit (POP)": pop,
                    "Total Count": total_count
                }

                if include_peak:
                    peak_col = f'{period}D Peak Return (%)'
                    peak_values = data[peak_col].dropna() if peak_col in data else pd.Series(dtype=float)
                    row["Average Peak Return (%)"] = peak_values.mean() if not peak_values.empty else float('nan')

                stats.append(row)

            return pd.DataFrame(stats).set_index("Period")

        stats_df = calculate_stats(breakout_days, include_peak=True)
        positive_stats_df = calculate_stats(positive_breakout_days, include_peak=True)
    
        #Write DataFrames to the Google Sheets.
        set_with_dataframe(worksheet_params, parameters_df, include_index=True)
        set_with_dataframe(worksheet_stats, stats_df, include_index=True)
        set_with_dataframe(worksheet_stats, positive_stats_df, row=len(stats_df) + 3, include_index=True)
        set_with_dataframe(worksheet_breakout, breakout_days, include_index=True)
        uniform_format = CellFormat(
            textFormat=TextFormat(bold=False),
            horizontalAlignment="LEFT"
        )

        #Apply uniform formatting to each worksheet.
        for worksheet in [worksheet_params, worksheet_stats, worksheet_breakout]:
            sheet_range = f"A1:Z1000"
            format_cell_range(worksheet, sheet_range, uniform_format)

        print(f"Spreadsheet '{spreadsheet_name}' successfully updated.")

    # def generate_excel(self, output_file):
    #     """
    #     Generates an Excel report containing:
    #     - Parameters used for the analysis.
    #     - Statistics of average returns over multiple holding periods post-breakout, including Max Return.
    #     - Detailed information on each breakout day.
        
    #     'output_file' is the file path where the Excel output should be saved.
        
    #     Returns:
    #         - breakout_days: pd.DataFrame containing rows for each identified breakout day along with corresponding returns.
    #         - stats_df: pd.DataFrame containing statistical summaries of returns for various holding periods post-breakout.
    #     """
    #     dataframe_stock = self._get_median_rolling_data()

    #     dataframe_stock['Volume Breakout'] = dataframe_stock['Volume'] > self.volume_threshold * dataframe_stock['Median Volume']

    #     dataframe_stock['Daily Return (%)'] = dataframe_stock['Close'].pct_change() * 100

    #     dataframe_stock['Next Day Close'] = dataframe_stock['Close'].shift(-1)

    #     return_periods = [1, 2, 5, 10, 15, 20, 25, 30]

    #     for period in return_periods:
    #         if period == 1:
    #             dataframe_stock[f'{period}D Return (%)'] = (
    #                 dataframe_stock['Next Day Close'] - dataframe_stock['Close']
    #             ) / dataframe_stock['Close'] * 100
    #         else:
    #             dataframe_stock[f'{period}D Return (%)'] = (
    #                 dataframe_stock['Close'].shift(-period) - dataframe_stock['Close']
    #             ) / dataframe_stock['Close'] * 100

    #     breakout_days = dataframe_stock[dataframe_stock['Volume Breakout']].copy()

    #     breakout_days['Date'] = breakout_days.index.astype(str) if breakout_days.index.name == 'Date' else breakout_days.index.map(str)
    #     breakout_days.reset_index(drop=True, inplace=True)

    #     positive_breakout_days = breakout_days[breakout_days['Daily Return (%)'] > 0].copy()

    #     parameters = {
    #         "Volume Threshold (%)": self.volume_threshold * 100,
    #         "Ticker": self.ticker,
    #         "Start Date": self.start_date,
    #         "End Date": self.end_date,
    #     }
    #     parameters_df = pd.DataFrame.from_dict(parameters, orient='index', columns=['Value'])

    #     peak_returns = []
    #     for _, breakout in breakout_days.iterrows():
    #         peak_row = {}
    #         # Get the index of the current breakout in the dataframe_stock
    #         breakout_index = dataframe_stock.index.get_loc(pd.to_datetime(breakout['Date']))
    #         for period in return_periods:
    #             # Determine the range of rows to consider
    #             end_index = breakout_index + period
    #             if end_index > len(dataframe_stock) - 1:
    #                 peak_row[f'{period}D Peak Return (%)'] = float('nan')
    #                 continue

    #             # Slice the data using row-based indexing
    #             sliced_data = dataframe_stock.iloc[breakout_index:end_index + 1]
    #             if sliced_data.empty or 'Close' not in sliced_data:
    #                 peak_row[f'{period}D Peak Return (%)'] = float('nan')
    #                 continue

    #             # Calculate sub-period returns
    #             sub_period_returns = []
    #             for i in range(1, len(sliced_data)):
    #                 initial_close = sliced_data['Close'].iloc[0]
    #                 later_close = sliced_data['Close'].iloc[i]
    #                 sub_period_returns.append(((later_close - initial_close) / initial_close) * 100)

    #             # Assign the maximum sub-period return
    #             peak_row[f'{period}D Peak Return (%)'] = max(sub_period_returns) if sub_period_returns else float('nan')

    #         peak_returns.append(peak_row)

    #     # Combine Peak Returns with Breakout Days DataFrame
    #     peak_returns_df = pd.DataFrame(peak_returns, index=breakout_days.index)
    #     breakout_days = pd.concat([breakout_days, peak_returns_df], axis=1)
    #     positive_peak_returns_df = peak_returns_df.loc[positive_breakout_days.index]
    #     positive_breakout_days = pd.concat([positive_breakout_days, positive_peak_returns_df], axis=1)

    #     def calculate_stats(data, include_peak=False):
    #         stats = []
    #         for period in return_periods:
    #             returns = data[f'{period}D Return (%)'].dropna()
    #             avg_return = returns.mean()

    #             t_stat = ttest_1samp(returns, 0).statistic if len(returns) > 1 else float('nan')

    #             max_return = returns.max() if len(returns) > 0 else float('nan')

    #             max_drawdown = (returns.min() / avg_return) if avg_return != 0 else float('nan')

    #             num_profitable = (returns > 0).sum()
    #             total_count = len(returns)
    #             pop = num_profitable / total_count if total_count > 0 else 0

    #             row = {
    #                 "Period": f'{period}D',
    #                 "Average Return (%)": avg_return,
    #                 "Max Return (%)": max_return,
    #                 "T-Stat": t_stat,
    #                 "Max Drawdown": max_drawdown,
    #                 "Number of Times Profitable": num_profitable,
    #                 "Probability of Profit (POP)": pop,
    #                 "Total Count": total_count
    #             }

    #             if include_peak:
    #                 peak_col = f'{period}D Peak Return (%)'
    #                 peak_values = data[peak_col].dropna() if peak_col in data else pd.Series(dtype=float)
    #                 row["Average Peak Return (%)"] = peak_values.mean() if not peak_values.empty else float('nan')

    #             stats.append(row)

    #         return pd.DataFrame(stats).set_index("Period")

    #     # Calculate statistics for both normal and positive breakout days
    #     stats_df = calculate_stats(breakout_days, include_peak=True)
    #     positive_stats_df = calculate_stats(positive_breakout_days, include_peak=True)

    #     # Write to Excel with Peak Returns
    #     with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    #         parameters_df.to_excel(writer, sheet_name="Parameters")
            
    #         stats_df.to_excel(writer, sheet_name="Average Returns", startrow=2, startcol=0)
    #         positive_stats_df.to_excel(writer, sheet_name="Average Returns", startrow=len(stats_df) + 8, startcol=0)
           
    #         worksheet = writer.sheets["Average Returns"]
    #         worksheet.write(0, 0, "Average Returns")
    #         worksheet.write(len(stats_df) + 6, 0, "Positive Average Returns")

            
    #         breakout_days.to_excel(writer, sheet_name="Breakout Days")


    #     return breakout_days, stats_df
