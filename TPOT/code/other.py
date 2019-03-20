## IGNORE THIS FILE - it is just notes

mlb = MultiLabelBinarizer()
CabinTrans = mlb.fit_transform([{str(val)} for val in data['Cabin'].values])

## write out encoder to file to reuse in production
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=mlb, filename='outputs/CabinEncoder.pkl')

## load and use
mlb = joblib.load('outputs/CabinEncoder.pkl')
CabinTrans = mlb.transform([{str(val)} for val in tpot_data['Cabin'].values])
