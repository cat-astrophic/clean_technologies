# This script does the statistical analysis for club convergence in access to clean technology and energy

# Importing required modules

import pandas as pd
import statsmodels.api as stats
import numpy as np
from matplotlib import pyplot as plt

# Loading data

data = pd.read_csv('C:/Users/User/Documents/Data/ccc.csv')
unit_data = pd.read_csv('C:/Users/User/Documents/Data/ccc_unit_root.csv')
unit_data = unit_data.set_index('Country')

# Endogenous variable

Y = data.Rate

# Exogenous variables

X = data[['Initial', 'Saving', 'Population']]
X = stats.add_constant(X)

# Running statistical test

model1 = stats.OLS(Y, X)
results1 = model1.fit()
print(results1.summary())
file = open('C:/Users/User/Documents/Data/ccc/model1.txt', 'w')
file.write(results1.summary().as_text())
file.close()

# Using approach from (P&S 2007) to test for club convergence

# First, create the matrix X_{it}

years = [2000+i for i in range(17)]
vals = [sum(unit_data[str(year)]) for year in years]

LITTLE_H = np.zeros(np.shape(unit_data))
for i in range(len(unit_data)):
    for j in range(len(years)):
        LITTLE_H[i,j] = unit_data.values[i,j] / ((1 / len(unit_data)) * vals[j])

# Second, find the cross sectional variance of X_{it}

big_h = np.zeros((1,len(years)))
for i in range(len(years)):
    s = 0
    for j in range(len(unit_data)):
        s += (LITTLE_H[j,i] - 1) ** 2
    big_h[0,i] = s / len(unit_data)


# Third, run regression to obtain estiamtes \hat{a} and \hat{b}

var = 5
ratios = [np.log(big_h[0,0] / big_h[0,t]) for t in range(var,17)]
LHS = [ratios[t] - 2*(np.log(np.log(t+var))) for t in range(len(ratios))]
LHS = pd.DataFrame(LHS, columns = ['LHS'])
RHS = [np.log(t) for t in range(var,17)]
RHS = pd.DataFrame(RHS, columns = ['RHS'])
RHS = stats.add_constant(RHS)

club_model = stats.OLS(LHS, RHS)
club_results = club_model.fit()
print(club_results.summary())
file = open('C:/Users/User/Documents/Data/ccc/club_results.txt', 'w')
file.write(club_results.summary().as_text())
file.close()

# Since \hat{a} and \hat{b} are negative, we may use a club convergence algorithm to determine club membership

# We define a function here

def clubbing(idx):
    
    club_vals = [sum(unit_data[str(year)][0:idx]) for year in years]
    
    club_h = np.zeros(np.shape(unit_data))
    for i in range(idx):
        for j in range(len(years)):
            club_h[i,j] = unit_data.values[i,j] / ((1 / idx) * club_vals[j])
    
    club_H = np.zeros((1, len(years)))
    for i in range(len(years)):
        s = 0
        for j in range(idx):
            s += (club_h[j,i] - 1) ** 2
        club_H[0,i] = s / idx
    
    var = 5
    ratios = [np.log(club_H[0,0] / club_H[0,t]) for t in range(var,17)]
    LHS = [ratios[t] - 2*(np.log(np.log(t+var))) for t in range(len(ratios))]
    LHS = pd.DataFrame(LHS, columns = ['LHS'])
    RHS = [np.log(t) for t in range(var,17)]
    RHS = pd.DataFrame(RHS, columns = ['RHS'])
    RHS = stats.add_constant(RHS)
    
    club_model = stats.OLS(LHS, RHS)
    club_results = club_model.fit()
    beta = club_results.params[1]
    return beta

# Creating convergence clubs

idx = 0
while idx == 0:
    for i in range(len(unit_data)):
        if unit_data['2016'][i] < 90:
            idx = i-1
            break
        else:
            pass

a = 0
clubs = []
remaining = [i for i in range(len(unit_data))]
while len(remaining) > 0:
    beta = 1
    while beta > 0:
        idx += 1
        print(a+idx)
        if (a + idx) == max(remaining):
            clubs.append(remaining)
            remaining = []
            beta = -1
        else:
            beta = clubbing(idx)
            if beta < 0:
                club = [i for i in range(a,a+idx)]
                clubs.append(club)
                for item in club:
                    remaining.remove(item)

    unit_data = unit_data.iloc[idx:len(unit_data), :]
    print(unit_data)
    a += idx
    idx = 1

# List names of members of clubs

unit_data = pd.read_csv('C:/Users/User/Documents/Data/ccc_unit_root.csv')
nations = unit_data.Country
unit_data = unit_data.set_index('Country')

members = []
for club in clubs:
    group = [nations[idx] for idx in club]
    members.append(group)
    group_ids = [idx for idx in club]

# Create plots for time series of each club

from pylab import rcParams as measurement_device
measurement_device['figure.figsize'] = 8.5, 8.5
pretty_rainbows = plt.get_cmap('gist_rainbow')

basis = [2000+i for i in range(17)]
for i in range(len(clubs)):
    plt.figure(i)
    for j in range(len(clubs[i])):
        x_data = unit_data.iloc[[clubs[i][j]]].values
        x_data = x_data.reshape(x_data.shape[1], x_data.shape[0])
        plt.plot(basis, x_data, label = members[i][j], color = pretty_rainbows(j/len(clubs[i])))
    plt.title('Convergence Club ' + str(i+1), loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
    plt.xlabel('Year')
    plt.ylabel('% population with access to clean technology')
    plt.legend(loc = 4, ncol = 2)
    plt.savefig('C:/Users/User/Documents/Data/ccc/ccc_plot_legend' + str(i+1)  + '.eps')
    k = 2*len(clubs)+i
    w = 3*len(clubs)+i
    q = 4*len(clubs)+i
    plt.figure(k)
    for j in range(len(clubs[i])):
        x_data = unit_data.iloc[[clubs[i][j]]].values
        x_data = x_data.reshape(x_data.shape[1], x_data.shape[0])
        plt.plot(basis, x_data, label = members[i][j], color = 'black')
    plt.title('Convergence Club ' + str(i+1), loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
    plt.xlabel('Year')
    plt.ylabel('% population with access to clean technology')
    plt.legend(loc = 4, ncol = 2)
    plt.savefig('C:/Users/User/Documents/Data/ccc/bw_plot_legend' + str(i+1)  + '.eps')
    plt.figure(w)
    for j in range(len(clubs[i])):
        x_data = unit_data.iloc[[clubs[i][j]]].values
        x_data = x_data.reshape(x_data.shape[1], x_data.shape[0])
        plt.plot(basis, x_data, label = members[i][j], color = pretty_rainbows(j/len(clubs[i])))
    plt.title('Convergence Club ' + str(i+1), loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
    plt.xlabel('Year')
    plt.ylabel('% population with access to clean technology')
    plt.savefig('C:/Users/User/Documents/Data/ccc/ccc_plot' + str(i+1)  + '.eps')
    plt.figure(q)
    for j in range(len(clubs[i])):
        x_data = unit_data.iloc[[clubs[i][j]]].values
        x_data = x_data.reshape(x_data.shape[1], x_data.shape[0])
        plt.plot(basis, x_data, label = members[i][j], color = 'black')
    plt.title('Convergence Club ' + str(i+1), loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
    plt.xlabel('Year')
    plt.ylabel('% population with access to clean technology')
    plt.savefig('C:/Users/User/Documents/Data/ccc/bw_plot' + str(i+1)  + '.eps')

# Create time series plot with club members sharing a common color

plt.figure(999)
for i in range(len(nations)):
    x = unit_data.iloc[[i]].values
    x = x.reshape(x.shape[1], x.shape[0])
    label_maker = [0,0,0,0,0,0,0,0]
    for j in range(len(members)):
        if nations[i] in members[j]:
            grp = j
            if label_maker[grp] == 0:
                lab = 'Convergence Club' + str(grp+1)
                label_maker[grp] = 1
            else:
                lab = ''
    plt.plot(basis, x, label = lab, color = pretty_rainbows(grp/8))
plt.title('All Nations by Convergence Club', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('% population with access to clean technology')
plt.savefig('C:/Users/User/Documents/Data/ccc/all_plot_ccc.eps')

# Write clubs with members to txt file for reference

members = pd.DataFrame(members)
members.to_csv('C:/Users/User/Documents/Data/ccc/club_membership.txt', index = False, header = False)

# Plot of all growth rates vs. initial values

plt.figure(545454554)
x_vals = data.Initial
plt.scatter(x_vals, Y, color = 'black')
xx_vals = stats.add_constant(x_vals)
tempmod = stats.OLS(Y, xx_vals)
tempres = tempmod.fit()
b = tempres.params[0]
m = tempres.params[1]
base_line = m * x_vals + b
plt.plot(x_vals, base_line, color = 'black')
plt.xlabel('Initial Level (Ln)')
plt.ylabel('Growth Rate')
plt.title('Growth Rate via Initial Value')
plt.savefig('C:/Users/User/Documents/Data/ccc/trends_all.eps')

# Plots of club level growth rates vs. initial values

plt.figure(34343434)
for club in clubs:
    x_club = data.Initial.iloc[club]
    y_club = Y.iloc[club]
    plt.scatter(x_club, y_club, color = pretty_rainbows(clubs.index(club)/8))
    xx_club = stats.add_constant(x_club)
    tempmod = stats.OLS(y_club, xx_club)
    tempres = tempmod.fit()
    b = tempres.params[0]
    m = tempres.params[1]
    line = m * x_club + b
    plt.plot(x_club, line, color = pretty_rainbows(clubs.index(club)/8))
plt.plot(data.Initial, base_line, color = 'black')
plt.xlabel('Initial Level (Ln)')
plt.ylabel('Growth Rate')
plt.title('Club Level: Growth Rate via Initial Value')
plt.savefig('C:/Users/User/Documents/Data/ccc/trends_at_club_level.eps')

# Club level beta convergence testing

for club in clubs:
    x_club = X.iloc[club,:]
    y_club = Y.iloc[club]
    x_club = stats.add_constant(x_club)
    mod_club = stats.OLS(y_club, x_club)
    res_club = mod_club.fit()
    print(res_club.summary())
    file = open('C:/Users/User/Documents/Data/ccc/club_' + str(clubs.index(club) + 1) + '.txt', 'w')
    file.write(res_club.summary().as_text())
    file.close()
    
# Generate statistics on covariates

df = pd.DataFrame()
for club in clubs:
    x_club = data.iloc[club,[1,2,3,4,5,6,7,8,9,10]]
    mu = np.mean(x_club)
    sigma = np.std(x_club)
    mu = pd.DataFrame(mu, columns = ['mu'])
    sigma = pd.DataFrame(sigma, columns = ['sigma'])
    df = pd.concat([df, mu, sigma], axis = 1)

df.to_csv('C:/Users/User/Documents/Data/ccc/clubstats.txt', 'w')

# Club level regressions on broader factor base to get rates of growth for access variable

for club in clubs:
    try:
        x_club = data.iloc[club,[1,2,3,4,5,6,7,8,9,10]]
        dummies = pd.get_dummies(x_club.Region)
        x_club = data.iloc[club,[3,4,5,6,7,8,9,10]]
        x_xlub = pd.concat([x_club, dummies], axis = 1)
        x_club = stats.add_constant(x_club)
        y_club = Y.iloc[club]
        club_mod = stats.OLS(y_club, x_club)
        club_res = club_mod.fit()
        print(club_res.summary())
        file = open('C:/Users/User/Documents/Data/ccc/club_model_' + str(clubs.index(club) + 1) + '.txt', 'w')
        file.write(club_res.summary().as_text())
        file.close()
    except:
        x_club = data.iloc[club,[1,2,3,4,5,6,7,8,9]]
        dummies = pd.get_dummies(x_club.Region)
        x_club = data.iloc[club,[3,4,5,6,7,8,9]]
        x_xlub = pd.concat([x_club, dummies], axis = 1)
        x_club = stats.add_constant(x_club)
        y_club = Y.iloc[club]
        club_mod = stats.OLS(y_club, x_club)
        club_res = club_mod.fit()
        print(club_res.summary())
        file = open('C:/Users/User/Documents/Data/ccc/club_model_' + str(clubs.index(club) + 1) + '.txt', 'w')
        file.write(club_res.summary().as_text())
        file.close()

# For creating a table of club level summary statistics

df = pd.DataFrame()
for club in clubs:
    mu = np.mean(data.iloc[club])
    mu = pd.DataFrame(mu)
    df = pd.concat([df, mu], axis = 1)
df.columns = ['Club ' + str(i) for i in range(1,9)]
df.to_csv('C:/Users/User/Documents/Data/ccc/summary_stats_at_club_level.txt', 'w')

# Corelation between access growth rate and HDI

# Club level regression for rates

A = pd.DataFrame(df.loc['Rate'].values, columns = ['Access Rate'])
B = pd.DataFrame(df.loc['HDI_Rate'].values, columns = ['HDI Rate'])
E = stats.add_constant(B)
mo = stats.OLS(A,B)
res = mo.fit()
print(res.summary())
file = open('C:/Users/User/Documents/Data/ccc/zresults_clubs_rates.txt', 'w')
file.write(res.summary().as_text())
file.close()

# Club level regression for initial values

A = pd.DataFrame(df.loc['Initial'].values, columns = ['Access Level'])
B = pd.DataFrame(df.loc['HDI_Init'].values, columns = ['Initial HDI'])
E = stats.add_constant(B)
mo = stats.OLS(A,B)
res = mo.fit()
print(res.summary())
file = open('C:/Users/User/Documents/Data/ccc/zresults_clubs_init.txt', 'w')
file.write(res.summary().as_text())
file.close()

# National level regression for rates

A = data.Rate
B = data.HDI_Rate
mo = stats.OLS(A,B)
res = mo.fit()
print(res.summary())
file = open('C:/Users/User/Documents/Data/ccc/zresults_nats_rates.txt', 'w')
file.write(res.summary().as_text())
file.close()

# National level regression for initial values

A = data.Initial
B = data.HDI_Init
mo = stats.OLS(A,B)
res = mo.fit()
print(res.summary())
file = open('C:/Users/User/Documents/Data/ccc/zresults_nats_init.txt', 'w')
file.write(res.summary().as_text())
file.close()

