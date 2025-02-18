import pandas as pd

# Creiamo un DataFrame di esempio
data = {
    'Nome': ['Antonio', 'Marco', 'Lucia'],
    'Età': [29, 34, 28],
    'Città': ['Pescara', 'Roma', 'Milano']
}

# Creiamo il DataFrame
df = pd.DataFrame(data)

# Visualizziamo il DataFrame
print(df)
