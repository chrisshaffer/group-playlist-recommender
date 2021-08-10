import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

rcParams.update({'font.size': 18})

results = pd.read_csv('data/results/results.csv')

hyperparameters = False

if hyperparameters == False:
    model_labels = ['SVD', 'NMF','KNN',
                        'Co-cluster','ALS Base','Normal','Global Mean']

    models_names = results.name[results.name.str.contains(
        'svd_default|nmf|knn|cocluster|baseline|normal|global_mean')]

    xlabel = 'Model Name'
    
    comparison = results[results.name.isin(models_names)]

    comparison['label'] = model_labels
    comparison.sort_values(by='rmse',ascending=False,inplace=True)

    fig,ax = plt.subplots(figsize=(13,6))
    plt.bar(comparison['label'],comparison['rmse'],color = sns.color_palette("husl", 7))
    plt.xlabel('Model Type')
    plt.ylabel('RMSE')

    plt.show(block=False)
    plt.savefig('img/rmse_comparison_no_ensemble.jpg')
    plt.close('all')

    # comparison = comparison.iloc[:,[3,2]]
    # comparison.to_csv('data/results/comparison_results.csv',index=False)
else:
    hyp_names = results.name[results.name.str.contains(
        'svd|pred.csv')]
    
    hyp_df = results[results.name.isin(hyp_names)]
    
    hyp_df.drop(index=[9,10],inplace=True)
    
    hyp_df['lr'] = [.01, .005, .001, .001, .001, .001, .005, .005, .005, .005]
    hyp_df['epochs'] = [20, 20, 20, 50, 75, 125, 10, 50, 125, 75]
    
    print(hyp_df)

    fig,ax = plt.subplots(figsize=(8,5))
    plt.scatter(x=hyp_df['lr'], y=hyp_df['epochs'], s=400, 
                c=hyp_df['rmse'], cmap=sns.color_palette('plasma', as_cmap=True))
    plt.xlabel('Learning Rate')
    plt.ylabel('Number of Epochs')

    cb = plt.colorbar()
    cb.ax.set_title('RMSE')
    
    fig.tight_layout()
    plt.show(block=False)
    plt.savefig('img/rmse_svd_hyp.jpg')
    plt.close('all')

    hyp_df = hyp_df.iloc[:,[3,4,2]]
    hyp_df.to_csv('data/results/hyp_results.csv',index=False)