import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

#returns associated string label for integer label
def get_label_name(label_id):
    #Ade20k indoor label space
    num_word_label = {
        '0': 'Unknown', '1': 'Wall', '2': 'Floor', '3': 'Ceiling',
        '4': 'Door', '5': 'Stairs', '6': 'Structure', '7': 'Shelf',
        '8': 'Plant', '9': 'Bed', '10': 'Storage', '11': 'Table',
        '12': 'Chair', '13': 'Wall_Decoration', '14': 'Couch',
        '15': 'Light', '16': 'Appliance', '17': 'Thing',
        '18': 'Deformable', '19': 'Dynamic_NonHuman', '20': 'Human'
    }

    return num_word_label.get(str(label_id), 'Unknown')

#returns count of all labels in the graph
def get_label_distribution(graph):
    labels = []
    #object layer
    for node in graph.get_layer(2).nodes:
        #add atttribute if exists
        if hasattr(node.attributes, 'semantic_label'):
            label_name = get_label_name(node.attributes.semantic_label)
            labels.append(f"{label_name}({node.attributes.semantic_label})")
        else:
            labels.append("Unknown")

    return Counter(labels)

#return all object centroids, grouped by label
def get_labelled_centroids(graph):
    centroids = {}
    #object layer
    for node in graph.get_layer(2).nodes:
        if hasattr(node.attributes, 'semantic_label'):
            label = get_label_name(node.attributes.semantic_label)
            pos = getattr(node.attributes, 'position', None)
            #store if position centroid is recorded and valid, store according to label
            if pos is not None:
                pos_array = np.array(pos)
                if not np.any(np.isinf(pos_array)) and not np.any(np.isnan(pos_array)):
                    centroids.setdefault(label, []).append(pos_array)
    return centroids

#return all object bounding boxes, grouped by label
def get_labelled_bboxes(graph):
    bboxes = {}

    #object layer
    for node in graph.get_layer(2).nodes:

        #attribute extraction and checks
        if not hasattr(node.attributes, 'semantic_label'):
            continue
        label = node.attributes.semantic_label
        bbox = getattr(node.attributes, 'bounding_box', None)
        if bbox is None:
            continue
        centroid = getattr(bbox, 'world_P_center', None)
        dimensions = getattr(bbox, 'dimensions', None)
        if centroid is None or dimensions is None:
            continue
        
        #add centroids and bounding box to numpy arrays
        centroid_arr = np.array(centroid)
        dimensions_arr = np.array(dimensions)

        if np.any(np.isinf(centroid_arr)) or np.any(np.isnan(centroid_arr)):
            continue
        if np.any(np.isinf(dimensions_arr)) or np.any(np.isnan(dimensions_arr)):
            continue

        #store according to label
        bboxes.setdefault(label, []).append({
            'centroid': centroid_arr,
            'bbox': dimensions_arr
        })
    return bboxes

#returns intersection over union of 2 bounding boxes
def _iou_bbox(box1, box2):
    #extract centroids and bboxs
    c1, s1 = np.array(box1['centroid']), np.array(box1['bbox'])
    c2, s2 = np.array(box2['centroid']), np.array(box2['bbox'])

    #half sizes for each box
    h1, h2 = s1 / 2, s2 / 2

    #calculate min and max corners for each box
    min1, max1 = c1 - h1, c1 + h1
    min2, max2 = c2 - h2, c2 + h2

    #intersection dimensions
    inter_dims = np.maximum(np.minimum(max1, max2) - np.maximum(min1, min2), 0)

    #intersection volume
    inter_vol = np.prod(inter_dims)
    #union volume
    union_vol = np.prod(s1) + np.prod(s2) - inter_vol

    #iou = intersection / union, catches zero division
    return 0.0 if union_vol == 0 else inter_vol / union_vol

#handles cases where no bounding box was registered, return nan instead of erroring
def _iou_3d_safe(box1, box2):
    try:
        vals = list(box1['centroid']) + list(box1['bbox']) + list(box2['centroid']) + list(box2['bbox'])
        if any(v != v for v in vals):
            return float('nan')
        return _iou_bbox(box1, box2)
    except Exception:
        return float('nan')


##############################################
#  STAT FUNCTIONS                            #
##############################################

#METRICS CALCULATED:

#    OVERALL NODE INFO
#    LABEL DISTRIBUTION
#    ROOM SEGMENTATION
#    DRIFT BETWEEN RUNS
#    BOUNDING BOX IoU
#    DISTANCE BETWEEN CENTROIDS OF OBJECT AND GROUND TRUTH
#    OBJECT RECALL AND PRECISION
#    SEMANTIC LABEL CONFUSION MATRIX


#returns dataframe containing stats for total, room and object nodes over 5 runs
def get_node_statistics(graphs):
    import spark_dsg as dsg

    nodes, rooms, objects = [], [], []

    #gets counts for each run
    for graph in graphs.values():
        obj_layer = graph.get_layer(dsg.DsgLayers.OBJECTS)
        room_layer = graph.get_layer(dsg.DsgLayers.ROOMS)
        nodes.append(graph.num_nodes())
        rooms.append(room_layer.num_nodes())
        objects.append(obj_layer.num_nodes())

    

    # #returns stats for list of counts
    # def _stats(vals):
    #     mean = np.mean(vals)
    #     std = np.std(vals)
    #     cv = std / mean if mean != 0 else 0
    #     return {'Mean': round(mean, 2), 'Std': round(std, 2), 'CV': round(cv, 2)}

    stats = {}

    #convert to numpy array for easy calculations
    nodes = np.array(nodes)
    rooms = np.array(rooms)
    objects = np.array(objects)

    #gets stats
    stats = {
        'Total Nodes':  {'Mean': round(nodes.mean(), 2),   'Std': round(nodes.std(), 2),   'CV': round(nodes.std() / nodes.mean(), 2)},
        'Room Nodes':   {'Mean': round(rooms.mean(), 2),   'Std': round(rooms.std(), 2),   'CV': round(rooms.std() / rooms.mean(), 2)},
        'Object Nodes': {'Mean': round(objects.mean(), 2), 'Std': round(objects.std(), 2), 'CV': round(objects.std() / objects.mean(), 2)},
    }

    #makes transposed dataframe for each section as a column
    df = pd.DataFrame(stats).T

    return df

#returns dataframe of counts for each semantic label across 5 runs
def get_label_statistics(graphs):

    #gets info on all labels
    all_labels = [get_label_distribution(g) for g in graphs.values()]
    label_df = pd.DataFrame(all_labels).fillna(0)

    #get stats
    stats = pd.DataFrame()
    stats['Mean'] = round(label_df.mean(), 2)
    stats['Std']  = round(label_df.std(), 2)
    stats['CV']   = round(stats['Std'] / stats['Mean'], 2)
    stats = stats.sort_values(by='Mean', ascending=False)
    stats.loc['Mean Row'] = round(stats.mean(), 2)

    return stats

#returns dataframe with stats for each run on room detection
#true positive is classes as (threshold) distance between gt and detected centroid
def get_room_statistics(gt_room_centroids, room_centroids, threshold=4.0):

    gt_array = np.array(gt_room_centroids)
    rows = []

    #for each run
    for i in sorted(room_centroids.keys()):
        used_gts = set()
        true_positive = 0

        #gets centroids for that run
        detected_in_run = np.array(room_centroids[i])

        if len(detected_in_run) > 0:
            #calculates distance between all detected centroids, with every other gt centroid
            dist_matrix = distance.cdist(detected_in_run[:, :2], gt_array[:, :2])
            
            for d_idx in range(len(detected_in_run)):
                dists = dist_matrix[d_idx]
                closest = np.argmin(dists)

                #if threshold distance away from a centroid
                if dists[closest] < threshold and closest not in used_gts:
                    true_positive += 1

                    #ensures 1 match per gt centroids
                    used_gts.add(closest)

        #recall stats
        num_detected = len(detected_in_run)
        num_gt = len(gt_array)
        fp = num_detected - true_positive
        fn = num_gt - true_positive

        recall    = true_positive / (true_positive + fn) if (true_positive + fn) > 0 else 0.0
        precision = true_positive / (true_positive + fp) if (true_positive + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append({'Run': f'Run {i}', 'Recall': recall, 'Precision': precision, 'F1-Score': f1})

    #put data into dataframe for viewing
    df = pd.DataFrame(rows).set_index('Run')
    mean_row = df.mean().to_frame().T
    mean_row.index = ['Mean (Overall)']
    std_row = df.std().to_frame().T
    std_row.index = ['Std']
    cv_vals = np.where(mean_row.values != 0, std_row.values / mean_row.values, float('nan'))
    cv_row = pd.DataFrame(cv_vals, columns=df.columns, index=['CV'])

    return pd.concat([df, mean_row, std_row, cv_row])

#returns drift stats for each semantic label across 5 runs
#uses same logic as rooms stats above, just stores distance instead of true positive
def get_drift_statistics(all_run_centroids):
    run_ids = sorted(all_run_centroids.keys())
    label_drifts = {}

    #implement pairwise comparison between runs, to get drift between runs for system consistency
    for idx, i in enumerate(run_ids):
        for j in run_ids[idx + 1:]:

            #for each label used, calculate the drift from nearest centroid from other run
            for label, positions_i in all_run_centroids[i].items():
                if label in all_run_centroids[j] and len(all_run_centroids[j][label]) > 0:
                    dist_matrix = distance.cdist(positions_i, all_run_centroids[j][label])
                    best_match = np.min(dist_matrix, axis=1)
                    label_drifts.setdefault(label, []).extend(best_match)

    all_drifts = []
    rows = []
    for label, drifts in label_drifts.items():
        all_drifts.extend(drifts)
        rows.append({
            'Label': label,
            'Mean': np.mean(drifts),
            'Max':  np.max(drifts),
            'Std':  np.std(drifts)
        })

    rows.append({
        'Label': 'Mean (Overall)',
        'Mean': np.mean(all_drifts),
        'Max':  np.max(all_drifts),
        'Std':  np.std(all_drifts)
    })

    return pd.DataFrame(rows).set_index('Label')

#returns mean euclidean distance between detected object centroids and nearest gt centroid. for each run
#same distance logic as room validation, but stores the distance instead of recall
def get_med_statistics(all_run_centroids, gt_by_label):
    
    all_gt_centroids = [obj['centroid'] for instances in gt_by_label.values() for obj in instances]
    rows = []

    for i in sorted(all_run_centroids.keys()):

        label_distances = []
        for label, positions in all_run_centroids[i].items():
            dist_matrix = distance.cdist(positions, all_gt_centroids)
            label_distances.extend(np.min(dist_matrix, axis=1))

        mean = np.mean(label_distances)
        std  = np.std(label_distances)
        rows.append({'Run': f'Run {i}', 'Mean': mean, 'Std': std, 'CV': std / mean})

    df = pd.DataFrame(rows).set_index('Run')
    mean_row = df.mean().to_frame().T
    mean_row.index = ['Mean (Overall)']

    return pd.concat([df, mean_row])

#returns stats for bounding box IoU per run between detected objects and nearest ground truth objects
def get_iou_statistics(all_run_bboxes, gt_by_label):

    #get ground truth values
    all_gt_centroids = [obj['centroid'] for instances in gt_by_label.values() for obj in instances]
    all_gt_bboxes    = [obj['bbox']     for instances in gt_by_label.values() for obj in instances]
    rows = []

    for i in sorted(all_run_bboxes.keys()):
        iou_scores = []

        for label, boxes in all_run_bboxes[i].items():
            bbox_positions = np.array([b['centroid'] for b in boxes])
            dist_matrix = distance.cdist(bbox_positions, all_gt_centroids)

            #calculates IoU for each box with closest gt object by centroid
            for d_idx in range(len(boxes)):
                closest_gt_idx = np.argmin(dist_matrix[d_idx])
                gt_box = {
                    'centroid': np.array(all_gt_centroids[closest_gt_idx]),
                    'bbox':     np.array(all_gt_bboxes[closest_gt_idx])
                }
                iou_scores.append(_iou_3d_safe(boxes[d_idx], gt_box))

        valid = [x for x in iou_scores if not np.isnan(x)]
        mean = np.mean(valid)
        std  = np.std(valid)
        nan_count = sum(1 for x in iou_scores if np.isnan(x))
        cv = std / mean if mean != 0 else float('nan')
        rows.append({'Run': f'Run {i}', 'Mean': mean, 'Std': std, 'CV': cv, 'Nan': nan_count})

    df = pd.DataFrame(rows).set_index('Run')
    mean_row = df.mean().to_frame().T
    mean_row.index = ['Mean (Overall)']

    return pd.concat([df, mean_row])

#returns recall precision and f1 stats per run for objects
def get_recall_statistics(all_run_centroids, gt_by_label, threshold=1.0):
    
    all_gt_centroids = [obj['centroid'] for instances in gt_by_label.values() for obj in instances]
    rows = []

    for i in sorted(all_run_centroids.keys()):
        used_gts = set()
        true_positive = 0

        for label, positions in all_run_centroids[i].items():
            #calculates distance between all detected centroids, with every other gt centroid
            dist_matrix = distance.cdist(positions, all_gt_centroids)

            for d_idx in range(len(positions)):
                dists = dist_matrix[d_idx]
                closest = np.argmin(dists)

                #true positive is within threshold distance (default 1)
                if dists[closest] < threshold and closest not in used_gts:
                    true_positive += 1
                    used_gts.add(closest)

        #get stats
        total_detected = sum(len(p) for p in all_run_centroids[i].values())
        recall    = true_positive / len(all_gt_centroids) if all_gt_centroids else 0.0
        precision = true_positive / total_detected if total_detected > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append({'Run': f'Run {i}', 'Recall': recall, 'Precision': precision, 'F1': f1})

    #make dataframe
    df = pd.DataFrame(rows).set_index('Run')
    mean_row = df.mean().to_frame().T
    mean_row.index = ['Mean (Overall)']
    std_row = df.std().to_frame().T
    std_row.index = ['Std']
    cv_vals = np.divide(std_row.values, mean_row.values, out=np.full_like(mean_row.values, float('nan')), where=mean_row.values != 0)
    cv_row = pd.DataFrame(cv_vals, columns=df.columns, index=['CV'])

    return pd.concat([df, mean_row, std_row, cv_row])

#returns confusion matrix for object labels, 2 plots for normalised and raw count values
def get_confusion_matrix(all_run_centroids, gt_by_label, threshold=1.0, normalise=True):

    all_gt_centroids = [obj['centroid'] for instances in gt_by_label.values() for obj in instances]
    all_gt_labels    = [label           for label, instances in gt_by_label.items() for _ in instances]

    all_predicted, all_actual = [], []

    #same logic as object and room detection
    for i in sorted(all_run_centroids.keys()):
        used_gts = set()
        for label, positions in all_run_centroids[i].items():
            dist_matrix = distance.cdist(positions, all_gt_centroids)
            for d_idx in range(len(positions)):
                dists = dist_matrix[d_idx]
                closest = np.argmin(dists)
                if dists[closest] < threshold and closest not in used_gts:
                    used_gts.add(closest)
                    #stores predicted and actual labels for confusion matrix
                    all_predicted.append(label)
                    all_actual.append(all_gt_labels[closest])

    if not all_actual:
        return None, None

    #make confusion matrix with mean counts across runs
    labels = sorted(set(all_actual + all_predicted))
    num_runs = len(all_run_centroids)
    cm = confusion_matrix(all_actual, all_predicted, labels=labels)
    cm_mean = cm / num_runs

    # Normalised matrix by row (gt labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    #makes normalised figure
    fig_norm, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(cm_norm, index=labels, columns=labels),
                annot=True, fmt='.2f', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual (GT)')
    ax.set_title('Mean Semantic Label Confusion Matrix (Normalised by Row)')
    fig_norm.tight_layout()

    #makes mean count figure
    fig_raw, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(cm_mean, index=labels, columns=labels),
                annot=True, fmt='.2f', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual (GT)')
    ax2.set_title('Mean Semantic Label Confusion Matrix (Raw Count)')
    fig_raw.tight_layout()

    print("Confusion Matrix Recall:")
    from sklearn.metrics import classification_report
    print(classification_report(all_actual, all_predicted, zero_division=0))

    return fig_norm, fig_raw