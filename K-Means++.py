import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getInitialCentoids(df,k,centroids):
    (rows,cols) = df.shape
    for i in range(1,k):
        SumProb = 0.0
        df['closest_centroids_{}'.format(i)]  = [getclosestDistance([df.iloc[j,0],df.iloc[j,1]],centroids) for  j in range(rows)]
        df['Prob_{}'.format(i)] = df['closest_centroids_{}'.format(i)]**2/np.sum(df["closest_centroids_{}".format(i)]**2)
        Prob_random = np.random.random()

        for n in range(rows):
            SumProb += df['Prob_{}'.format(i)][n]
            if SumProb>Prob_random:
                break
        centroids.update({i:[df.iloc[n,0],df.iloc[n,1]]})
        # print(df)
        # print(centroids)
    return centroids
def getclosestDistance(Points, centroids):
   # for i in centroids.keys():
    closet_Distance = np.min([np.sqrt((Points[0]-centroids[i][0])**2 + (Points[1] - centroids[i][1])**2) for i in centroids.keys()])
    return closet_Distance

def assignment(df,centoids,colormap):
    for i in centoids.keys():
        df["distance_from_{}".format(i)] = np.sqrt((df['x']-centoids[i][0])**2 + (df['y']-centoids[i][1])**2)
        #df['Prob_dis_{}'.format(i)] = (df["distance_from_{}".format(i)]**2)/np.sum(df["distance_from_{}".format(i)]**2)
    distance_ids = ["distance_from_{}".format(i) for i in centoids.keys()]
    df["closest"] = df.loc[:,distance_ids].idxmin(axis=1)
    df["closest"] =df["closest"].map(lambda x: int(x.replace("distance_from_","")))
    df['color'] = df["closest"].map(lambda x : colormap[int(x)])
#    df["closestValues"] = df.loc[:, distance_ids].min(axis=1)
    print(df)
    return  df

def update(df,centroids):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df["closest"] == i]["y"])
    return centroids

def main():
    df = pd.DataFrame({
        "x":[12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        "y":[39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })
    k = 3
    dfCopy = df.copy(deep=True)
    df_random = df.sample(n=1,axis=0)
    centroids = {0:[df_random.iloc[0,0],df_random.iloc[0,1]]}
    centoids = getInitialCentoids(dfCopy,k,centroids)
    print(centroids)
    colormap = { 0:'r', 1:"g",2:'b'}
    df = assignment(df,centoids,colormap)

    plt.scatter(df['x'],df['y'], color=df["color"],alpha=0.5, edgecolor = "k")
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colormap[i], linewidths=6)
    plt.xlim(0,80)
    plt.ylim(0,80)
    plt.show()

    for i in range(10):
        plt.close()
        closest_centroids = df["closest"].copy(deep=True)
        plt.scatter(df['x'],df["y"],color=df["color"], alpha=0.5, edgecolor='k')
        centroids = update(df,centroids)
        for j in centroids.keys():
            plt.scatter(*centroids[j],color=colormap[i], linewidths=6)
        plt.xlim(0,80)
        plt.ylim(0,80)
        plt.show()
        df = assignment(df, centroids,colormap)
        if closest_centroids.equals(df["closest"]):
            break

if __name__=="__main__":
    main()