# Faster Algorithms for HodgeRank

## Usage
All useful functions in this repository are in hodgerank_tools.py, examples of their use can be found in rank_nba.ipynb.
### Structuring Data
Most functions in hodgerank_tools, take data as input, which should be structured as a list of dictionaries, where each
dictionary represents a voter's preferences.

Example:
```
my_data = [{"Fever":3, "Sore Throat": 2, "Nausea": 5},
            {"Fever":6, "Sore Throat": 2, "Cough": 2, "Nausea": 1}, 
            ...]
```
