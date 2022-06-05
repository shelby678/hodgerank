import numpy as np
import pandas as pd
import math

# returns a list of all edges in the form of tuples
#ex: get_edges(['a', 'b', 'c']) => [('a', 'b'), ('a', 'c'), ('b','c')]
def get_edges(nodes):
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            edges.append((nodes[i],nodes[j]))
    return edges

# Converts a voter df into a list of dictionaries, where each dictionary represents the 
# opinions of one voter
def df_to_dict_list(df):
    data = []
    nodes = df.columns
        #row_nodes = [node_index for node_index in range(len(nodes)) if not np.isnan(row[node_index])]
    for index, row in df.iterrows():
    # get group of nodes that this voter has scored
        row_nodes = [node for node in nodes if not np.isnan(row[node])]
        voter_dict = dict()
        for node in row_nodes:
            voter_dict[node] = row[node]
        data.append(voter_dict)
    return data

#TODO: preprocess

# gets all nodes from voter data structures as a list of dictionaries
def get_nodes(data):
    nodes = set()
    for voter in data:
        nodes.update(voter.keys())
    nodes = list(nodes)
    nodes.sort()
    return nodes

# returns:
# - f is a vector representing the edge flows between the nodes. It is indexed by 
# the edges from the get_edges function
#  - W is a diagonal matrix containing the weights of each edge
# where df is a pandas dataframe such that the columns represent the nodes to be ranked and each 
# row gives a voter's ratings
def get_f_W(data, nodes):
    
    edges = get_edges(nodes)
    f = np.zeros((len(edges)))
    W = np.zeros((len(edges), len(edges)))
    # iterate through each voter's ratings
    for voter in data:
    
        for edge in get_edges(list(voter.keys())):
            # for each edge in this voter's graph, add pairwise difference to f and add 1 to W
            f[edges.index(edge)] += voter[edge[1]] - voter[edge[0]]
            W[edges.index(edge), edges.index(edge)] += 1
    # weighting f by W
    for i in range(len(edges)):
        if W[i, i] != 0:
            f[i] = f[i]*1/W[i,i]
        else:
            f[i] = 0
    return (f, W)

#TODO: revise this to fit the data frame version of ranks <3
# returns the error of this ranking according to some ranking r, which is a list containing
# the overall (numerical) rating of each node, indexed by the nodes in nodes
def get_error(f, W, r, nodes):
    edges = get_edges(nodes)
    sum = 0
    for i in range(len(edges)):
        to_add = (f[i] + (r[nodes.index(edges[i][0])] - r[nodes.index(edges[i][1])]))**2
        sum += to_add
    return sum

# returns the negative divergence matrix, structured as a numpy array, where df is a 
# pandas dataframe such that the columns represent the nodes to be ranked and each 
# row gives a voter's ratings
def get_neg_divergence(nodes):
    edges = get_edges(nodes)
    neg_divergence = np.zeros((len(edges), len(nodes)))        
    for i, edge in enumerate(edges):
        for j, node in enumerate(nodes):
            if edge[0] == node:
                neg_divergence[i,j] = -1
            elif edge[1] == node:
                neg_divergence[i,j] = 1
    return neg_divergence

# ranks based on hodgerank
# returns:
# - ranking
# - error
def rank(data):
    nodes = get_nodes(data)
    # get edges, negative divergence, f, and W
    edges = get_edges(nodes)
    neg_divergence = get_neg_divergence(nodes)
    (f, W) = get_f_W(data, nodes)
    # solve for r
    right_side = np.matmul(np.transpose(neg_divergence), np.matmul(W, f))
    left_side = np.matmul(np.matmul(np.transpose(neg_divergence), W), neg_divergence)
    r = np.matmul(np.linalg.pinv(left_side), right_side)
    # put r into df and sort by score
    rank_df = pd.DataFrame({'node': nodes, 'r': r})
    rank_df = rank_df.sort_values(by =['r'],  ascending = False)
    rank_df = rank_df.reset_index(drop = True)
    #rank_df.to_csv('data/hodge_ranking.csv') # uncomment this line if you want to save the ranking
    return(rank_df, get_error(f, W, r, nodes))
    
# Ranks based on average score
# returns:
# - ranking
# - error
def naive_rank_0(data):
    #initialize nodes, sum_scores, total_votes
    nodes = get_nodes(data)
    sum_scores = dict()
    naive_r = dict()
    for node in nodes:
        sum_scores[node] = 0
    total_votes = sum_scores.copy()
    # get sum_scores, total_votes from data
    for voter in data:
        for node in list(voter.keys()):
            total_votes[node] += 1
            sum_scores[node] += voter[node]
    # get final naive_r
    for node in nodes:
        naive_r[node] = sum_scores[node]/total_votes[node]
    # format into a df
    rank_df = pd.DataFrame.from_dict(naive_r, orient = 'index', columns = ['r'])
    rank_df = rank_df.reset_index()
    rank_df = rank_df.rename(columns = {'index':'node'})
    rank_df = rank_df.sort_values(by =['r'],  ascending = False)
    rank_df = rank_df.reset_index(drop = True)
    #(f, W) = get_f_W(data, nodes)
    return (rank_df) #get_error(f, W, naive_r, nodes)

# Ranks by the mean pairwise difference of each node
def naive_rank(data):
    #initialize nodes, sum_scores, total_votes
    nodes = get_nodes(data)
    (f, W) = get_f_W(data, nodes)
    sum_scores = dict()
    naive_r = dict()
    for node in nodes:
        sum_scores[node] = 0
    total_votes = sum_scores.copy()
    edges = get_edges(nodes)
    for node in nodes:
        # populate total_votes, sum_scores
        for i, edge  in enumerate(edges):
            if edge[0] == node:
                sum_scores[node] -= f[i]
            if edge[1] == node:
                sum_scores[node] += f[i]
            else:
                continue
            total_votes[node] += W[i, i]
        # calculate final score
        if total_votes[node] == 0:
            naive_r[node] = 0
        else:
            naive_r[node] = sum_scores[node]/total_votes[node]
    # format into a df
    rank_df = pd.DataFrame.from_dict(naive_r, orient = 'index', columns = ['r'])
    rank_df = rank_df.reset_index()
    rank_df = rank_df.rename(columns = {'index':'node'})
    rank_df = rank_df.sort_values(by =['r'],  ascending = False)
    rank_df = rank_df.reset_index(drop = True)
    #(f, W) = get_f_W(data, nodes)
    return (rank_df) #get_error(f, W, naive_r, nodes)

# returns a list of groupings sorted by naive rank
# ex: group_similar_scoring({df with cols 'a', 'b', 'c'}, 2) => [['a', 'b'], ['c']]
def group_similar_scoring(data, k):
    # get naive rank
    naive_r = naive_rank(data)
    ranked_teams = list(naive_r["node"])
    # split up teams
    groupings = list(np.array_split(ranked_teams, k))
    groupings = [grouping.tolist() for grouping in groupings]
    return(groupings)

# returns intersection of two listss
def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

# Ranks the nodes in df by creating rankings on k different groups (organized by naive_rank()) 
# and stacking the final scores on top of eachother
def simple_group_rank(data, k):
    # get groupings
    groupings = group_similar_scoring(data, k)
    # inititialize r_groups, error
    r = pd.DataFrame(columns = ['node', 'r'])
    error = 0
    # create ranking for each grouping
    for grouping in groupings:
        # build grouping_data
        grouping_data = []
        for voter in data:
            intersect = intersection(list(voter.keys()), grouping)
            if len(intersect) > 2:
                subset = {node: score for node, score in voter.items() if node in intersect}
                grouping_data.append(subset)
        # get ranking
        (group_rank, group_error) = rank(grouping_data)
        # add group rank,  error to overall
        r = pd.concat([r,group_rank])
        error += group_error
    r = r.reset_index(drop = True)
    return(r, error)

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

#TODO: revise
def group_rank(df, k):
    groupings = group_similar_scoring(df, k) 
    r_groups = pd.DataFrame(columns = ['node', 'r'])
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
    r_groups = r_groups[~r_groups['node'].isin(fake_teams)]
    r_groups = r_groups.reset_index(drop = True)
    return(r_groups, error)
