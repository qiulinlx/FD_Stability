'''

Scratchfile of things I will look at later

Attempts at looking at 
1. Periodicity and volume under each curbe 
2. Shifts in lowest EVI across winters


'''


winter_year = []
min_dates = []
AUC_list=[]

sub = df[df['PID'] == pid]

sub = sub.dropna(axis=1)
sub.drop(columns=['PID'], inplace=True)


date_cols = pd.to_datetime([c for c in df.columns if c not in ['PID']], errors='coerce')

winter_mask = date_cols.month.isin([12, 1, 2])
winter_cols = df.columns[1:][winter_mask]   

date_cols = [c for c in df.columns if c not in ['PID']]
dates = pd.to_datetime(date_cols, errors='coerce')


sub.columns = pd.to_datetime(sub.columns)

winter_df=pd.DataFrame({'PID':df['PID']})

for d in dates:
    if d.month == 12:
        winter_year.append(d.year)       
    else: 
        winter_year.append(d.year - 1)    

        
winter_groups = {}
for col, wy in zip(date_cols, winter_year):
    if wy not in winter_groups:
        winter_groups[wy] = []
    winter_groups[wy].append(col)

for wy, cols in winter_groups.items():
    winter_df[f'winter_{wy}_col'] = df[cols].idxmin(axis=1)

winter_sub=winter_df[winter_df['PID'] == pid]
winter_sub.drop(columns=['PID'], inplace=True)
winter_sub=np.array(winter_sub)
winter_sub = pd.to_datetime(winter_sub[0]).date

for i in range(len(winter_sub)-1):
    start = winter_sub[i]
    end   =winter_sub[i+1]
    
    sub = sub.loc[:, start:end]

    # X-axis as numeric days
    dates = pd.to_datetime(sub.columns)
    print(dates[0])

    x = (dates - dates[0]).days
    y = sub.values.flatten()
    
    area = np.trapezoid(y, x)
    AUC_list.append(area)