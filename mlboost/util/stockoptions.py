''' eval stock at exit 
example: python stockoptions.py --shares 283333 --total 10000000 --option 3 --exit 150000000'''
def profit(shares, total, option_price, exit_price):
    price = float(exit_price)/total
    perc = float(shares)/total*100
    return "%.2f%% -> $%.1fM" %(perc, shares*(price-option_price)/1000000)

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s','--shares', type=int, help='nb of shares you own')
    parser.add_argument('-t','--total', type=int, help='total nb of shares')
    parser.add_argument('-o','--option', type=int, help='share option price')
    parser.add_argument('-e','--exit', type=int, help='exit price')

    args = parser.parse_args()
    print(profit(args.shares, args.total, args.option, args.exit))
    
