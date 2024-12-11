# import the usual suspects
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.counterfactual_generator import CounterfactualGenerator
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.utils import load_config,check_file_exists,print_balancing,write_time
import dice_ml
import time
import seaborn as sns

config = load_config("config.yaml")


dataset_name = 'german_credit'

data_processor = DataProcessor(config=config)

splits_path = f"data/processed/{dataset_name}/y_train.npz"
if not check_file_exists(splits_path):
    print(splits_path,"not found")
    print("Processing dataset")
    data = data_processor.load_dataset(dataset_name)
    processed_data = data_processor.preprocess_data(data,
                                                    dataset_name)
    splits = data_processor.split_data(features=processed_data[0],
                                        target=processed_data[1])
    data_processor.save_splits(splits, dataset_name)
else:
    print("Loading existing dataset")
    splits = data_processor.load_splits(dataset_name)

# print the balance of the splits
print("Train balance: \n")
print_balancing(splits['y_train'])
print("Test balance: \n")
print_balancing(splits['y_test'])
print("Calibration balance: \n")
print_balancing(splits['y_calibration'])

cf_generator = CounterfactualGenerator(config=config)
cf_generator.setup_dice(reference_data=pd.DataFrame([], columns=splits['X_train'].columns),
                        continuous_features=splits['X_train'].columns,
                        categorical_features=[],
                        target_name='target')

model_trainer = ModelTrainer(config=config)
random_F = model_trainer._load_model("random_forest", dataset_name)

xbg = model_trainer._load_model("xgboost", dataset_name)
mlp = model_trainer._load_model("mlp", dataset_name)
lgbm = model_trainer._load_model("lgbm", dataset_name)
models = {"random_forest": random_F,
          "xgboost": xbg,
          "mlp": mlp,
          "lgbm": lgbm}
experiments = {m:{meth:[0,[np.inf]] for meth in ["random","kdtree","genetic"]} for m in ["mlp","xgboost","random_forest","lgbm"]}

experiments

found=0
total_samples = 100
chosen_model = "mlp"
num_cf = 8
to_try = ["random","kdtree"]#,"genetic"]
percs = [1,10,50,100]
full_experiments = {p:{} for p in percs}
results = {}
for percentage in percs:
    results[percentage] = {}
    #########################
    #### CHOOSE THE DATA ####
    #########################
    if percentage==1:
        reference_data = splits['X_train'].sample(1)
        reference_data["target"] = 1
    else:
        # choose a percentage of the training data

        choosen_idxs = np.random.choice(splits['X_train'].index, int(len(splits['X_train'])*percentage/100))
        reference_data = splits['X_train'].loc[choosen_idxs]
        reference_data["target"] = splits['y_train'].loc[choosen_idxs]
    dice_data = dice_ml.Data(dataframe=reference_data,#pd.DataFrame([], columns=splits['X_train'].columns+["target"]),
                            continuous_features=splits['X_train'].columns.to_list(),
                            categorical_features=[],
                            outcome_name='target',
                            dataset_name='german_credit')

    #########################
    #### CHOOSE THE MODEL ###
    #########################
    experiments = {m:{meth:[0,[np.inf]] for meth in ["random","kdtree","genetic"]} for m in ["mlp","xgboost","random_forest","lgbm"]}

    for cf_method in to_try:
        results[percentage][cf_method] = {}
        for model in [chosen_model]:
            results[percentage][cf_method][model] = {}
            print("Generating counterfactuals for",model,"with",cf_method,"method with",percentage,"% of the training data")
            dice_model = dice_ml.Model(model=models["mlp"],
                                    backend="sklearn")
            explainer = dice_ml.Dice(dice_data, dice_model, method=cf_method)

            found=0
            times = []
            founded = []
            oupts = []
            for i in range(total_samples):
                e = splits['X_calibration'].loc[[i]]
                
                try:
                    start = time.time()
                    res = explainer.generate_counterfactuals(e, total_CFs=num_cf, desired_class="opposite",verbose=False)
                    print(len(res.cf_examples_list[0].final_cfs_df))
                    oupts.append(res)
                    found+=1
                    founded.append(1)
                except Exception as exc:
                    print(exc)
                    founded.append(0)
                    continue

                elapsed = time.time()-start
                
                '''
                # to test the time and the founds invent the data
                foundq = np.random.randint(0,2)
                found+=foundq
                founded.append(foundq)
                elapsed = np.random.rand()*10
                '''
                times.append(elapsed)
            print("Found",found,"counterfactuals", "out of",total_samples)
            print("percentage",found/total_samples)

            experiments[model][cf_method][0] = np.array(founded)
            experiments[model][cf_method][1] = np.array(times)
            results[percentage][cf_method][model]["found"] = founded
            results[percentage][cf_method][model]["times"] = times
            results[percentage][cf_method][model]["outputs"] = oupts
    full_experiments[percentage] = experiments


# define the figure
for meth in to_try:
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax2 = ax.twinx()
    all_percs = []
    all_times = []
    labs = []
    boxes = []

    for j,i in enumerate(percs):
        # get the experiment:
        exp = full_experiments[i]
        # plot the bars relatively to the total samples
        # at the j-th position
        percentages = exp[chosen_model][meth][0].sum()/total_samples*100
        print(percentages)
        ax.bar(j-0.2,percentages,0.2,label=str(round(percentages,2))+"% succes")
        text_position = (j-0.25,percentages/4)
        # write in white the percentage of found counterfactuals
        ax.text(*text_position,str(round(percentages,2))+"% success",color="white",fontsize=16,rotation=90)
        ax.set_ylabel("Percentage of found counterfactuals")
        # plot the boxplot of the times
        # get the ax2
        cmap = sns.color_palette("tab10", as_cmap=True)
        times = exp[chosen_model][meth][1]
        pathc = ax2.boxplot(times,positions=[j],widths=0.2,patch_artist=True,showfliers=True)
        if times.sum()>0:
            labs.append(write_time(np.mean(times))+"±"+
                    write_time(np.std(times)))
        else:
            labs.append("No time")
        all_times.append(times)
        for patch in pathc['boxes']:
            patch.set_facecolor(cmap(j))
        boxes.append(pathc["boxes"][0])
        ax2.set_ylabel("Time (s)",fontsize=18)
        all_percs.append(percentages)

    ax.set_ylim(0,100)
    # make the legend external
    ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1),fontsize=16)
    ax2.legend(boxes,labs,loc='upper left', bbox_to_anchor=(1.2, 0.4),fontsize=16)

    # set the y ticks to be at 1 second, 30, 60, 120.
    # use as labels 1s, 30s, 1m, 2m
    ax2.set_yscale("log")
    ax2.set_yticks([1,30,60,120,60*5])
    ax2.set_yticklabels(["1s","30s","1m","2m","5m"],fontsize=16)
    all_times_s = [i.sum() for i in all_times]

    additional_label = "Total time needed: "+" ".join(map(write_time,all_times_s))
    xticks = [str(p)+"%\n"+(t_s) for p,t_s in zip(percs,map(write_time,all_times_s))]
    #ax.set_xticks(np.arange(len(percs))-.1,[str(p)+"%" for p in percs],fontsize=16)
    ax.set_xticks(np.arange(len(percs))-.1,xticks,fontsize=16)
    ax.set_xlabel("Percentage of the training data used as reference and total time needed "+write_time(np.sum(all_times_s)),fontsize=16,loc="left")
    plt.suptitle("model "+chosen_model+"\n Ablation for dice_ml \""+
                 str(meth)+
                 "\" \nfor "+str(total_samples)+" samples "+f'({num_cf} each)',fontsize=18)
    plt.tight_layout()
    # show the grid
    plt.grid(True)
    date = time.strftime("%Y-%m-%d-%H-%M-%S")
    date.replace("-","")
    spath = f"plots/ablation_{date}_{chosen_model}_{meth}_dice_ml.pdf"
    plt.savefig(spath)
    print("saved in",spath)
    print(additional_label)

    # save everything in a folder called debug with pickle
    import pickle
    with open(spath.replace("plots","debug").replace(".pdf",".pkl"),"wb") as f:
        pickle.dump({"all_percs":all_percs,
                    "all_times":all_times,
                    "labs":labs,
                    "boxes":boxes,
                    "percs":percs,
                    "total_samples":total_samples,
                    "full_experiments":full_experiments,
                    "results":results},f)
    