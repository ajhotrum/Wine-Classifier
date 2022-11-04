import pandas as pd
import numpy as np
from sklearn.utils import resample
from collections import Counter


def balance(path):
    df = pd.read_csv(path, header=None)
    df.columns = [*df.columns[:-1], 'Y']

    # Test for balance
    seq = np.array(df['Y'])
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)
    H = -sum([(count/n) * np.log((count/n)) for clas, count in classes])

    # If already balanced, get out
    if H/np.log(k) > 0.75:
        print("Already Balanced")
        return 0

    else:
        u = np.unique(seq)

        #Get the number of instances of most frequent y-value
        maximum = df['Y'].value_counts().iloc[0]

        #Get highest instance classification (classification with most occurrences)
        highest = df['Y'].value_counts().index[0]

        #remove highest from list of unique
        new_u = np.delete(u, np.where(u==highest))

        #Make df of just the most frequent y
        df_majority = df[df.Y == highest]

        for i in range(len(new_u)):
            # make df of all rows with current y-value
            df_minority = df[df.Y == new_u[i]]
            # if it has the fewer of instances, just add it to the final
            if df['Y'].value_counts()[new_u[i]] < maximum:
                df_minority = resample(df_minority,
                                    replace=True,     # sample with replacement
                                    n_samples=maximum,    # to match majority class
                                    random_state=123) # reproducible results

            # Add minority df to combined df
            df_majority = pd.concat([df_majority, df_minority])

        # Print combined df to csv
        df_majority.to_csv('page-blocks_balanced.csv', header=False, index=False)


if __name__ == '__main__':

    path = 'page-blocks.csv'
    balance(path)
