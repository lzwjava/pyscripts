from tigeropen.common.consts import Language, Market, BarPeriod, QuoteRight
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.quote.quote_client import QuoteClient
import os
from tigeropen.trade.trade_client import TradeClient
from tigeropen.common.util.order_utils import market_order
from tigeropen.common.util.contract_utils import stock_contract
import sys


def get_client_config(sandbox=False):
    client_config = TigerOpenClientConfig(sandbox_debug=sandbox)
    client_config.private_key = read_private_key(os.environ.get("TIGER_PEM"))
    client_config.tiger_id = os.environ.get("TIGER_TIGER_ID")
    client_config.account = os.environ.get("TIGER_ACCOUNT")
    client_config.language = Language.zh_CN
    return client_config


def get_stocks_briefs():
    client_config = get_client_config()
    quote_client = QuoteClient(client_config)

    hong_kong_biggest_companies_codes = [
        "00700",  # Tencent Holdings Ltd.
        "09988",  # Alibaba Group Holding Ltd.
        "01299",  # AIA Group Ltd.
        "00941",  # China Mobile Ltd.
        "00388",  # Hong Kong Exchanges and Clearing Ltd.
        "03690",  # Meituan
        "01398",  # Industrial and Commercial Bank of China Ltd.
        "02318",  # Ping An Insurance Group Co. of China Ltd.
        "01810",  # Xiaomi Corporation
        "03988",  # Bank of China Ltd.
    ]

    stock_prices = quote_client.get_stock_briefs(hong_kong_biggest_companies_codes)
    print(stock_prices)


def place_order():
    client_config = get_client_config()
    trade_client = TradeClient(client_config)
    contract = stock_contract(symbol="NVDA", currency="USD")
    stock_order = market_order(
        account=client_config.account, contract=contract, action="BUY", quantity=4
    )
    trade_client.place_order(stock_order)
    print(stock_order)


def cancel_order(order_id):
    client_config = get_client_config()
    trade_client = TradeClient(client_config)
    is_cancelled = trade_client.cancel_order(id=order_id)
    print(f"Order cancellation status: {is_cancelled}")


if __name__ == "__main__":

    job = sys.argv[1]

    if job == "get_stocks_briefs":
        get_stocks_briefs()
    elif job == "place_order":
        place_order()
    elif job == "cancel_order":
        order_id = int(sys.argv[2])
        cancel_order(order_id)
