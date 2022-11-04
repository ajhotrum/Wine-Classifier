import pandas as pd

path = r'C:\Users\Alonso.Torres\Downloads\data\data'
df_majority = pd.DataFrame()

print(df_majority)

for i in range(1,15):
    if i < 10:
        temp_path = path + r"\a0"+ str(i)+"output.csv"
    else:
        temp_path = path + r"\a"+ str(i)+"output.csv"

    df = pd.read_csv(temp_path, header=None)
    df['a'] = i


    # Add minority df to combined df
    df_majority = pd.concat([df_majority, df])

    print(df_majority)


# Print combined df to csv
df_majority.to_csv(r'C:\Users\Alonso.Torres\Downloads\data\data\final_output.csv', header=False, index=False)
