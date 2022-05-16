# Faster Algorithms for HodgeRank

## Usage
All useful functions in this repository are in hodgerank_tools.py, examples of their use can be found in rank_nba.ipynb.
### Structuring Data
In order to use most of the functions in hodgerank_tools, your data has to be in a particular format. The data frame that is expected
should have the elements being ranked (symptoms, basketball teams, etc.) as the columns and with each row representing a voter's opinions.

Example:  
         Fever | Sore Throat | Cough | Nausea  
(Person 1:)  3       2            2        5  
(Person 2:)  7       8            9       NaN  
(Person 3:)  1       2            1        3

Additionally, elements that weren't rated by a voter should receive a value of [np.nan](https://numpy.org/doc/stable/user/misc.html).
