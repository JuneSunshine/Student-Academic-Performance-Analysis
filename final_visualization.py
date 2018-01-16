'''
Data visualization
'''
import seaborn as sns
import matplotlib.pyplot as plt


def plot_cat(df,attrs):
    '''
    plot categorical attributes
    :param df: dataframe
    :param attrs: attributes
    :return: None
    '''
    fig = plt.figure(figsize=(20,30))
    fig.subplots_adjust(hspace=0.3, wspace = 0.2)

    for i in range(1,len(attrs)+1):
        ax = fig.add_subplot(5,2,i)
        sns.countplot(df[attrs[i-1]])
        ax.xaxis.label.set_size(20)
        plt.setp(ax.get_xticklabels(),rotation=45,horizontalalignment='right')
        total = float(len(df))
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.0, height+5,'{:1.1f}%'.format(100 * height/total), ha='center')
        sns.plt.show()

def plot_num(df,attrs):
    '''
    plot numerical attributes
    :param df: dataframe
    :param attrs: attributes
    :return: None
    '''
    fig = plt.figure(figsize = (10,10))
    # pair plot attributes
    sns.pairplot(df[attrs],hue='Class')
    sns.plt.show()

    for i in range(1,len(attrs)):
        ax = fig.add_subplot(2,2,i)
        sns.distplot(df[attrs[i-1]],kde=None)
        sns.plt.show()


def pair_grid(df):
    '''
    pair grid comparison
    :param df: dataframe
    :return: None
    '''
    pg = sns.PairGrid(df,hue = 'Class', palette='coolwarm', hue_kws= {'marker':['o','s','D']})
    pg = pg.map_diag(plt.hist)
    pg = pg.map_upper(plt.scatter, linewidths=1, edgecolor = 'w', s =40)
    pg = pg.map_lower(sns.kdeplot, lw=3, legend=False, cmap = 'coolwarm')
    pg = pg.add_legend()
    sns.plt.show()

def immigrants(df):
    '''
    plot "isImmigrant" attribute
    :param df: dataframe
    :return: None
    '''
    fig = plt.figure(figsize=(10,10))
    sns.pairplot(df[['isImmigrant','Class']],hue='Class')
    sns.plt.show()

