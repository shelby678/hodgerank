import numpy as np
import pandas as pd
import math

def get_edges(elements):
    edges = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            edges.append((elements[i],elements[j]))
    return edges

# f is a vector representing pairwise differences between the nodes
# W is a diagonal matrix containing the weights of each edge
def get_f_W(df):
    #init edges, triangles, adj_matrix, curl, neg_divergence, f, and W
    elements = df.columns
    num_nodes = len(elements)
    edges = get_edges(elements)
   
    f = np.zeros((len(edges)))
    W = np.zeros((len(edges), len(edges)))
    for index, row in df.iterrows():
        row_elements = [element for element in elements if not np.isnan(row[element])]
        for edge in get_edges(row_elements):
            f[edges.index(edge)] += row[edge[1]] - row[edge[0]]
            W[edges.index(edge), edges.index(edge)] += 1
    for i in range(len(edges)):
        if W[i, i] != 0:
            f[i] = f[i]*1/W[i,i]
        else:
            f[i] = 0
    return (f, W)

def get_error(f, W, r, elements):
    edges = get_edges(elements)
    sum = 0
    for i in range(len(edges)):
        to_add = 1*(f[i] + (r[elements.index(edges[i][0])] - r[elements.index(edges[i][1])]))**2
        sum += to_add
    return sum

def get_neg_divergence(df):
    elements = list(df.columns)
    edges = get_edges(elements)
    neg_divergence = np.zeros((len(edges), len(elements)))        
    # neg_divergence
    for i, edge in enumerate(edges):
        for j, element in enumerate(elements):
            if edge[0] == element:
                neg_divergence[i,j] = -1
            elif edge[1] == element:
                neg_divergence[i,j] = 1
    return neg_divergence

def rank(df):
    elements = list(df.columns)
    edges = get_edges(elements)
    neg_divergence = get_neg_divergence(df)
    (f, W) = get_f_W(df)
    right_side = np.matmul(np.transpose(neg_divergence), np.matmul(W, f))
    left_side = np.matmul(np.matmul(np.transpose(neg_divergence), W), neg_divergence)
    r = np.matmul(np.linalg.pinv(left_side), right_side)
    rank_df = pd.DataFrame({'element': elements, 'r': r})
    rank_df = rank_df.sort_values(by =['r'],  ascending = False)
    rank_df = rank_df.reset_index(drop = True)
    #rank_df.to_csv('data/hodge_ranking.csv')
    return(rank_df, get_error(f, W, r, elements))
    
# mean score
def naive_rank_0(df):
    elements = list(df.columns)
    if len(list(set(elements))) != len(elements):
        raise Exception("All columns must have different names")
    naive_r = [0]*len(elements)
    for i, element in enumerate(elements):
        naive_r[i] = df[element].mean()
    naive_rank_df = pd.DataFrame({'element': elements, 'r': naive_r})
    naive_rank_df = naive_rank_df.sort_values(by =['r'],  ascending = False)
    naive_rank_df = naive_rank_df.reset_index(drop = True)
    return (naive_rank_df, get_error(get_f_W(df)[0], get_f_W(df)[1], naive_r, elements))

# mean pairwise difference
def naive_rank(df):
    elements = list(df.columns)
    if len(list(set(elements))) != len(elements):
        raise Exception("All columns must have different names")
    naive_r = [0]*len(elements)
    element_weights = [0]*len(elements)
    for index, row in df.iterrows():
        row_elements = [element for element in elements if not np.isnan(row[element])]
        for edge in get_edges(row_elements):
            naive_r[elements.index(edge[0])] += row[edge[0]] - row[edge[1]]
            naive_r[elements.index(edge[1])] += row[edge[1]] - row[edge[0]]
            element_weights[elements.index(edge[0])] += 1
            element_weights[elements.index(edge[1])] += 1
    naive_r = [naive_r[i]/element_weights[i] for i in range(len(naive_r))]
    naive_rank_df = pd.DataFrame({'element': elements,'r': naive_r})
    naive_rank_df = naive_rank_df.sort_values(by =['r'],  ascending = False)
    naive_rank_df = naive_rank_df.reset_index(drop = True)
    return (naive_rank_df, get_error(get_f_W(df)[0], get_f_W(df)[1], naive_r, elements))

# a (very convulated) way to almost evenly distribute all nodes into 
# groupings such that each group has at least 
# group_size number of nodes
def get_group_lengths(df, k):
    num_nodes = len(list(df.columns))
    group_size = math.floor(num_nodes/k)
    group_lengths = [group_size]*k
    for n in range(num_nodes - group_size * len(group_lengths)):
        group_lengths[n % len(group_lengths)] += 1
    return(group_lengths)

def group_similar_scoring(df, k):
    naive_r = naive_rank(df)[0]
    # sort by score
    scores_df = naive_r[['r', 'element']]
    scores_df = scores_df.sort_values(by =['r'],  ascending = False)
    scores_df = scores_df.reset_index(drop = True)
    ranked_teams = list(scores_df["element"])
    # fill list with groups of teams, sorted by score
    groupings = []
    for group_length in get_group_lengths(df, k):
        groupings.append(ranked_teams[0:group_length])
        ranked_teams = ranked_teams[group_length:]
    return(groupings)

def simple_group_rank(df, k):
    groupings = group_similar_scoring(df, k)
    r_groups = pd.DataFrame(columns = ['element', 'r'])
    error_groups = 0
    # create ranking for each grouping
    for grouping in groupings:
        small_game_df = df[grouping]
        small_game_df = small_game_df.dropna(how='all')
        (group_rank, group_error) = rank(small_game_df)
        r_groups = pd.concat([r_groups,group_rank])
        error_groups += group_error
    r_groups = r_groups.reset_index(drop = True)
    return(r_groups, error_groups)

# a specialized method that returns the sum across the first axis of a list of lists, 
# treating nan's as 0 unless the entire row is nans
# ex: nansum([[5, np.nan, 1],       =>    [15, np.nan, 2] 
#             [5, np.nan, np.nan],   
#             [5, np.nan, 1]])
def nansum(to_sum):
    sum = [np.nan]*len(to_sum[0])
    for j in range(len(sum)):
        for i in range(len(to_sum)):
            if not np.isnan(to_sum[i][j]):
                sum[j] = np.nansum([to_sum[i][j], sum[j]])
    return sum


def group_rank(df, k):
    groupings = group_similar_scoring(df, k) 
    r_groups = pd.DataFrame(columns = ['element', 'r'])
    fake_teams = set()
    error = 0
    # create ranking for each grouping
    for grouping in groupings:
        new_grouping = grouping.copy()
        small_game_df = df.copy()
        for i, other_grouping in enumerate(groupings):
            if other_grouping == grouping: 
                continue
            fake_team_col = [[np.nan]*len(small_game_df.index)]
            fake_team_name = 'OTHER_TEAM' + str(i)
            fake_teams.add(fake_team_name)
            for group in other_grouping:
                fake_team_col.append( list(small_game_df[group]))
                small_game_df = small_game_df.drop(columns = [group])
            fake_team_col = nansum(fake_team_col)
            small_game_df[fake_team_name] = fake_team_col
        (group_rank, group_error) = rank(small_game_df)
        r_groups = pd.concat([r_groups,group_rank])
        error += group_error
    r_groups = r_groups[~r_groups['element'].isin(fake_teams)]
    r_groups = r_groups.reset_index(drop = True)
    return(r_groups, error)
