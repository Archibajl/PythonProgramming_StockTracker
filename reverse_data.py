import csv
import numpy as np

def ReverseStockData():
    with open('upload_DJIA_table.csv', newline='\n') as price_data_file:
        stock_price = list(csv.reader(price_data_file, delimiter=',', skipinitialspace=True))
        stock_price=np.array(stock_price)
    with open('upload_DJIA_table.csv', 'w', newline='\n') as price_data_file2:
        j=(len(stock_price))
        print(len(stock_price))
        print(j)
        csvwriter= csv.writer(price_data_file2, skipinitialspace=True)
        print(stock_price[0])
        for i in reversed(range(1, j)):
            print(stock_price[i])
            csvwriter.writerow(stock_price[i])
    price_data_file2.close()
    price_data_file.close()
    return
def ReverseRedditData():
    with open('RedditNews.csv', 'r', newline='') as DJIA_news_file:
        stock_newsdata = list(csv.reader(DJIA_news_file, delimiter=',', skipinitialspace=True))
        stock_newsdata=np.array(stock_newsdata)
    with open('RedditNews.csv', 'w', newline='\n') as DJIA_news_file2:
        j=(len(stock_newsdata))
        print(len(stock_newsdata))
        print(j)
        csvwriter= csv.writer(DJIA_news_file2, skipinitialspace=True)
        print(stock_newsdata[0])
        for i in reversed(range(1, j)):
            print(stock_newsdata[i])
            csvwriter.writerow(stock_newsdata[i])
    DJIA_news_file.close()
    DJIA_news_file2.close()
    return

#ReverseStockData()
#ReverseRedditData()