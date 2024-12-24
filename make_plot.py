import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("/home/kshitiz/scratch/MAMMO/FUSION_NET")
from calc_metrics2 import calc_froc
import torch
from matplotlib import cm
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler



def get_sample_scores(pred_list, threshold=0.1, key = "new_scores", topk=5):
    def true_positive(gt, pred):
        # If center of pred is inside the gt, it is a true positive
        box_pascal_gt = ( gt[0]-(gt[2]/2.) , gt[1]-(gt[3]/2.), gt[0]+(gt[2]/2.), gt[1]+(gt[3]/2.) )
        if (pred[0] >= box_pascal_gt[0] and pred[0] <= box_pascal_gt[2] and
                pred[1] >= box_pascal_gt[1] and pred[1] <= box_pascal_gt[3]):
            return True
        return False
    
    positive_samples = []
    negative_samples = []
    for i, data_item in enumerate(tqdm(pred_list)):
        gt_data = data_item['target']
        # print(gt_data)
        pred = data_item['pred']
        scores = pred["scores"]
        select_mask = scores > threshold

        pred_boxes = pred['boxes'][select_mask]
        pred_scores = pred[key][select_mask]
        # pred_boxes = pred['boxes'][:topk]
        # pred_scores = pred[key][:topk]
        # print(scores.shape, pred_scores.shape)

        for k,pred in enumerate(pred_boxes):
            flag = False
            for j, gt_box in enumerate(gt_data['boxes']):
                if true_positive(gt_box, pred):
                    # print("Hello from inside", )
                    flag = True
            if(flag):
                positive_samples.append(pred_scores[k].item())
            else:
                negative_samples.append(pred_scores[k].item())
    # import pdb; pdb.set_trace()
    return positive_samples, negative_samples



def make_plot(data1, data2, ax_lims, fpi_data=None, ax=None, title = "FND+OURS", data1_label = "Postive BOXES", data2_label = "Negative BOXES"):
    x1,x2,y = ax_lims 
    # Create density plot
    sns.kdeplot(data1, label=data1_label, shade=True, ax=ax)
    sns.kdeplot(data2, label=data2_label, shade=True, ax=ax)

    if(fpi_data):
        fpi, thresh_data = fpi_data
        cmap = cm.get_cmap('coolwarm')  
        for i,thresh in enumerate(thresh_data):
            color = cmap(i / (len(thresh_data) - 1))
            ax.axvline(x=thresh, color=color, linestyle='--', linewidth=1)
            # ax.axvline(x=thresh, color=color, linestyle='--', linewidth=1, label=f'fpi={fpi[i]}')
            # ax.axvline(x=thresh, color=color, linestyle='--', linewidth=1)
            # ax.text(fpi[i], 3, f'fpi={fpi[i]}', rotation=90, color=color, ha='center', va='center')

    ax.set_xlim(x1, x2)
    ax.set_ylim(0, y)
    # Set plot labels and title
    # ax.text(0.5, -0.9,'{} Proposals'.format(title), ha='center', va='center', fontsize=12)
    # ax.text(0.5, 1.1, '{} Proposals'.format(title), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
    # ax.set_title('{} Proposals'.format(title), loc='center', pad=-20)
    ax.set_xlabel('CONFIDENCE (Standardized)', fontsize=14, weight='bold')
    ax.set_ylabel('PDF', fontsize=14, weight='bold')
    

    # Show legend
    legend = ax.legend(fontsize=14, title_fontsize=18, title=title, prop={'weight': 'bold'})
    legend.get_title().set_fontweight('bold')

    # Show the plot
    # ax.savefig("den_plots/{}_{}_plot.png".format(data,key))
    # ax.clf()


def min_max_scale(data1, data2, thresh):
    full_data = np.array(data1+data2); data1 = np.array(data1); data2 = np.array(data2); thresh = np.array(thresh) 
    data1 = data1-full_data.min() ; data1 = data1/full_data.max()
    data2 = data2-full_data.min() ; data2 = data2/full_data.max()
    thresh = thresh-full_data.min() ; thresh = thresh/full_data.max()
    return data1, data2, thresh

def standard_scale(data1, data2, thresh):
    full_data = np.array(data1+data2); data1 = np.array(data1); data2 = np.array(data2); thresh = np.array(thresh) 
    mean = np.mean(full_data, axis=0)
    std = np.std(full_data, axis=0)
    
    data1 = data1-mean ; data1 = data1/std
    data2 = data2-mean ; data2 = data2/std
    thresh = thresh-mean ; thresh = thresh/std
    
    return data1, data2, thresh


def plot_inbreast(inbreast_pred_file, fpi):
    axes_lims = [-2,4,0.8]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    # plt.title('Density Plot for Positive and Negative Proposals')
    
    inbreast_pred_list = torch.load(inbreast_pred_file)
    _, sens, thresh = calc_froc(inbreast_pred_list, fps_req=fpi)
    
    inbreast_pos_data, inbreast_neg_data = get_sample_scores(inbreast_pred_list, key="scores")    
    print(len(inbreast_pos_data), len(inbreast_neg_data))

    inbreast_pos_data, inbreast_neg_data, thresh = standard_scale(inbreast_pos_data, inbreast_neg_data, thresh)
    make_plot(inbreast_pos_data, inbreast_neg_data, axes_lims, fpi_data=[fpi, thresh], key="before", data="inbreast", ax=ax1, title="FND")

    for data_item in inbreast_pred_list:
        data_item["pred"]["old_scores"] = data_item["pred"]["scores"]
        data_item["pred"]["scores"] = data_item["pred"]["new_scores"]
    
    _, sens, thresh = calc_froc(inbreast_pred_list, fps_req=fpi)
    for data_item in inbreast_pred_list:
        data_item['pred']["scores"] = data_item['pred']["old_scores"]
        
    inbreast_pos_data, inbreast_neg_data = get_sample_scores(inbreast_pred_list, key="new_scores")    
    print(len(inbreast_pos_data), len(inbreast_neg_data))
    inbreast_pos_data, inbreast_neg_data, thresh = standard_scale(inbreast_pos_data, inbreast_neg_data, thresh)
    make_plot(inbreast_pos_data, inbreast_neg_data, axes_lims, fpi_data=[fpi, thresh], key="after", data="inbreast", ax=ax2, title='FND+OURS')
    # fig.suptitle('INBreast Dataset')
    plt.tight_layout()
    plt.savefig("den_plots/inbreast_density_plot.png")
    plt.clf()

def plot_aiims(aiims_pred_file, fpi):
    axes_lims = [-2,3,1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    aiims_pred_list = torch.load(aiims_pred_file)
    _, sens, thresh = calc_froc(aiims_pred_list, fps_req=fpi)
    
    aiims_pos_data, aiims_neg_data = get_sample_scores(aiims_pred_list, key="scores")    
    print(len(aiims_pos_data), len(aiims_neg_data))

    aiims_pos_data, aiims_neg_data, thresh = standard_scale(aiims_pos_data, aiims_neg_data, thresh)
    make_plot(aiims_pos_data, aiims_neg_data, axes_lims, fpi_data=[fpi, thresh], key="before", data="aiims", ax=ax1, title="FND")

    for data_item in aiims_pred_list:
        data_item["pred"]["old_scores"] = data_item["pred"]["scores"]
        data_item["pred"]["scores"] = data_item["pred"]["new_scores"]
    
    _, sens, thresh = calc_froc(aiims_pred_list, fps_req=fpi)
    for data_item in aiims_pred_list:
        data_item['pred']["scores"] = data_item['pred']["old_scores"]
        
    aiims_pos_data, aiims_neg_data = get_sample_scores(aiims_pred_list, key="new_scores")    
    print(len(aiims_pos_data), len(aiims_neg_data))
    aiims_pos_data, aiims_neg_data, thresh = standard_scale(aiims_pos_data, aiims_neg_data, thresh)
    make_plot(aiims_pos_data, aiims_neg_data, axes_lims, fpi_data=[fpi, thresh], key="after", data="aiims", ax=ax2)
    plt.tight_layout()
    plt.savefig("den_plots/aiims_density_plot.png")
    plt.clf()

def plot_ddsm(ddsm_pred_file, fpi):
    axes_lims = [-2,5,1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ddsm_pred_list = torch.load(ddsm_pred_file)
    _, sens, thresh = calc_froc(ddsm_pred_list, fps_req=fpi)
    
    ddsm_pos_data, ddsm_neg_data = get_sample_scores(ddsm_pred_list, key="scores")    
    print(len(ddsm_pos_data), len(ddsm_neg_data))

    ddsm_pos_data, ddsm_neg_data, thresh = standard_scale(ddsm_pos_data, ddsm_neg_data, thresh)
    make_plot(ddsm_pos_data, ddsm_neg_data, axes_lims, fpi_data=[fpi, thresh], key="before", data="ddsm", ax=ax1, title="FND")

    for data_item in ddsm_pred_list:
        data_item["pred"]["old_scores"] = data_item["pred"]["scores"]
        data_item["pred"]["scores"] = data_item["pred"]["new_scores"]
    
    _, sens, thresh = calc_froc(ddsm_pred_list, fps_req=fpi)
    for data_item in ddsm_pred_list:
        data_item['pred']["scores"] = data_item['pred']["old_scores"]
        
    ddsm_pos_data, ddsm_neg_data = get_sample_scores(ddsm_pred_list, key="new_scores")    
    print(len(ddsm_pos_data), len(ddsm_neg_data))
    ddsm_pos_data, ddsm_neg_data, thresh = standard_scale(ddsm_pos_data, ddsm_neg_data, thresh)
    make_plot(ddsm_pos_data, ddsm_neg_data, axes_lims, fpi_data=[fpi, thresh], key="after", data="ddsm", ax=ax2)
    plt.tight_layout()
    plt.savefig("den_plots/ddsm_density_plot.png")
    plt.clf()

def plot_inbreast_2(inbreast_pred_file, fpi):
    axes_lims = [-2,4,0.8]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    # plt.title('Density Plot for Positive and Negative Proposals')
    
    inbreast_pred_list = torch.load(inbreast_pred_file)
    _, sens, thresh = calc_froc(inbreast_pred_list, fps_req=fpi)
    
    inbreast_pos_data_orig, inbreast_neg_data_orig = get_sample_scores(inbreast_pred_list, key="scores")    
    print(len(inbreast_pos_data_orig), len(inbreast_neg_data_orig))
    inbreast_pos_data_orig, inbreast_neg_data_orig, thresh = standard_scale(inbreast_pos_data_orig, inbreast_neg_data_orig, thresh)
    # make_plot(inbreast_pos_data_orig, inbreast_neg_data_orig, axes_lims, fpi_data=[fpi, thresh], key="before", data="inbreast", ax=ax1, title="FND")

    for data_item in inbreast_pred_list:
        data_item["pred"]["old_scores"] = data_item["pred"]["scores"]
        data_item["pred"]["scores"] = data_item["pred"]["new_scores"]
    
    _, sens, thresh = calc_froc(inbreast_pred_list, fps_req=fpi)
    for data_item in inbreast_pred_list:
        data_item['pred']["scores"] = data_item['pred']["old_scores"]
        
    inbreast_pos_data_prop, inbreast_neg_data_prop = get_sample_scores(inbreast_pred_list, key="new_scores")    
    print(len(inbreast_pos_data_prop), len(inbreast_neg_data_prop))
    inbreast_pos_data_prop, inbreast_neg_data_prop, thresh = standard_scale(inbreast_pos_data_prop, inbreast_neg_data_prop, thresh)
    # make_plot(inbreast_pos_data_prop, inbreast_neg_data_prop, axes_lims, fpi_data=[fpi, thresh], key="after", data="inbreast", ax=ax2, title='FND+OURS')
    
    make_plot( inbreast_neg_data_prop, inbreast_neg_data_orig, axes_lims, ax=ax1, title='NEGATIVE', data1_label="FDN+OURS", data2_label="FDN")
    make_plot( inbreast_pos_data_prop, inbreast_pos_data_orig, axes_lims, ax=ax2, title='POSITIVE', data1_label="FDN+OURS", data2_label="FDN")
    
    # fig.suptitle('INBreast Dataset')
    plt.tight_layout()
    plt.savefig("den_plots/inbreast_density_plot_2.png")
    plt.clf()
    
def plot_ddsm_2(ddsm_pred_file, fpi):
    axes_lims = [-2,5,1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    # plt.title('Density Plot for Positive and Negative Proposals')
    
    ddsm_pred_list = torch.load(ddsm_pred_file)
    _, sens, thresh = calc_froc(ddsm_pred_list, fps_req=fpi)
    
    ddsm_pos_data_orig, ddsm_neg_data_orig = get_sample_scores(ddsm_pred_list, key="scores")    
    print(len(ddsm_pos_data_orig), len(ddsm_neg_data_orig))
    ddsm_pos_data_orig, ddsm_neg_data_orig, thresh = standard_scale(ddsm_pos_data_orig, ddsm_neg_data_orig, thresh)
    # make_plot(ddsm_pos_data_orig, ddsm_neg_data_orig, axes_lims, fpi_data=[fpi, thresh], key="before", data="ddsm", ax=ax1, title="FND")

    for data_item in ddsm_pred_list:
        data_item["pred"]["old_scores"] = data_item["pred"]["scores"]
        data_item["pred"]["scores"] = data_item["pred"]["new_scores"]
    
    _, sens, thresh = calc_froc(ddsm_pred_list, fps_req=fpi)
    for data_item in ddsm_pred_list:
        data_item['pred']["scores"] = data_item['pred']["old_scores"]
        
    ddsm_pos_data_prop, ddsm_neg_data_prop = get_sample_scores(ddsm_pred_list, key="new_scores")    
    print(len(ddsm_pos_data_prop), len(ddsm_neg_data_prop))
    ddsm_pos_data_prop, ddsm_neg_data_prop, thresh = standard_scale(ddsm_pos_data_prop, ddsm_neg_data_prop, thresh)
    # make_plot(ddsm_pos_data_prop, ddsm_neg_data_prop, axes_lims, fpi_data=[fpi, thresh], key="after", data="ddsm", ax=ax2, title='FND+OURS')
    
    make_plot( ddsm_neg_data_prop, ddsm_neg_data_orig, axes_lims, ax=ax1, title='NEGATIVE', data1_label="FDN+OURS", data2_label="FDN")
    make_plot( ddsm_pos_data_prop, ddsm_pos_data_orig, axes_lims, ax=ax2, title='POSITIVE', data1_label="FDN+OURS", data2_label="FDN")
    
    # fig.suptitle('ddsm Dataset')
    plt.tight_layout()
    plt.savefig("den_plots/ddsm_density_plot_2.png")
    plt.clf()

def plot_aiims_2(aiims_pred_file, fpi):
    axes_lims = [-2,3,1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    # plt.title('Density Plot for Positive and Negative Proposals')
    
    aiims_pred_list = torch.load(aiims_pred_file)
    _, sens, thresh = calc_froc(aiims_pred_list, fps_req=fpi)
    
    aiims_pos_data_orig, aiims_neg_data_orig = get_sample_scores(aiims_pred_list, key="scores")    
    print(len(aiims_pos_data_orig), len(aiims_neg_data_orig))
    aiims_pos_data_orig, aiims_neg_data_orig, thresh = standard_scale(aiims_pos_data_orig, aiims_neg_data_orig, thresh)
    # make_plot(aiims_pos_data_orig, aiims_neg_data_orig, axes_lims, fpi_data=[fpi, thresh], key="before", data="aiims", ax=ax1, title="FND")

    for data_item in aiims_pred_list:
        data_item["pred"]["old_scores"] = data_item["pred"]["scores"]
        data_item["pred"]["scores"] = data_item["pred"]["new_scores"]
    
    _, sens, thresh = calc_froc(aiims_pred_list, fps_req=fpi)
    for data_item in aiims_pred_list:
        data_item['pred']["scores"] = data_item['pred']["old_scores"]
        
    aiims_pos_data_prop, aiims_neg_data_prop = get_sample_scores(aiims_pred_list, key="new_scores")    
    print(len(aiims_pos_data_prop), len(aiims_neg_data_prop))
    aiims_pos_data_prop, aiims_neg_data_prop, thresh = standard_scale(aiims_pos_data_prop, aiims_neg_data_prop, thresh)
    # make_plot(aiims_pos_data_prop, aiims_neg_data_prop, axes_lims, fpi_data=[fpi, thresh], key="after", data="aiims", ax=ax2, title='FND+OURS')
    
    make_plot( aiims_neg_data_prop, aiims_neg_data_orig, axes_lims, ax=ax1, title='NEGATIVE', data1_label="FDN+OURS", data2_label="FDN")
    make_plot( aiims_pos_data_prop, aiims_pos_data_orig, axes_lims, ax=ax2, title='POSITIVE', data1_label="FDN+OURS", data2_label="FDN")
    
    # fig.suptitle('aiims Dataset')
    plt.tight_layout()
    plt.savefig("den_plots/aiims_density_plot_2.png")
    plt.clf()



if __name__=="__main__":
    aiims_pred_file = "DENSITY/aiims_density_plot.dict"
    inbreast_pred_file = "DENSITY/inbreast_density_plot.dict"
    ddsm_pred_file = "DENSITY/ddsm_density_plot.dict"
    fpi = [0.025, 0.05, 0.1, 0.3, 0.5, 1]    
    
    # plot_inbreast(inbreast_pred_file, fpi)
    # plot_ddsm(ddsm_pred_file, fpi)
    # plot_aiims(aiims_pred_file, fpi)
    # plot_inbreast_2(inbreast_pred_file, fpi)
    plot_ddsm_2(ddsm_pred_file, fpi)
    plot_aiims_2(aiims_pred_file, fpi)
    
    
    






