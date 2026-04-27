from src.data.download import download_multiple_tickers
from src.data.preprocess import preprocess_multiple_tickers

if __name__ == "__main__":
    # 1. 데이터 다운로드
    download_multiple_tickers(
        tickers=["SPY"],
        start_date="2015-01-01",
        end_date="2024-12-31",
        interval="1d"
    )

    # 2. 전처리
    preprocess_multiple_tickers(
        tickers=["SPY"],
        normalize=True
    )