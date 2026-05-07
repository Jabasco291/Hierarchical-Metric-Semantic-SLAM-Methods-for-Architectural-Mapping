import spark_dsg as dsg
from get_gt3 import get_gt_by_label, get_gt_rooms

#get functions for calulcating stats
from stats_functions import (
    get_labelled_centroids,
    get_labelled_bboxes,
    get_node_statistics,
    get_label_statistics,
    get_room_statistics,
    get_drift_statistics,
    get_med_statistics,
    get_iou_statistics,
    get_recall_statistics,
    get_confusion_matrix,
)

#loads graphs for given method and stores in dictionary with keys as run numbers
def load_graphs(method):
    paths = {
        "hydra": "../graphs/hydra/eval_runs_20260327_1838/run_{i}/backend/dsg.json",
        "clio":  "../graphs/clio/eval_runs_20260328_1754/run_{i}/backend/dsg.json",
    }

    graphs = {}
    for i in range(1, 6):
        path = paths[method].format(i=i)
        graphs[i] = dsg.DynamicSceneGraph.load(path)
    return graphs

#get and print stats for each method
def run_evaluation(method):

    #GET DATA

    #get all graphs
    graphs = load_graphs(method)

    # compute per-run centroids/bboxes
    all_run_centroids = {i: get_labelled_centroids(g) for i, g in graphs.items()}
    all_run_bboxes    = {i: get_labelled_bboxes(g)    for i, g in graphs.items()}

    # get ground truth data
    gt_by_label = get_gt_by_label()
    gt_rooms = get_gt_rooms()
    gt_room_centroids = [v['centroid'] for v in gt_rooms.values()]

    # room centroids per run
    room_centroids = {}
    for i, graph in graphs.items():
        room_layer = graph.get_layer(dsg.DsgLayers.ROOMS)
        room_centroids[i] = [node.attributes.position.copy() for node in room_layer.nodes]

    #CALC AND DISPLAY RESULTS

    print(f"--- {method.upper()} Evaluation Results ---")

    # ── Node Statistics ──────────────────────────────────────
    print("\n--- Node Statistics ---")
    print(get_node_statistics(graphs).round(2))

    # ── Label Distribution ───────────────────────────────────
    print("\n--- Label Distribution ---")
    print(get_label_statistics(graphs).round(2))

    # ── Room Recognition ─────────────────────────────────────
    print("\n--- Room Recognition ---")
    print(get_room_statistics(gt_room_centroids, room_centroids, threshold=4.0).round(2))

    # ── Drift ────────────────────────────────────────────────
    print("\n--- Drift Statistics ---")
    print(get_drift_statistics(all_run_centroids).round(2))

    # ── Mean Euclidean Distance ───────────────────────────────
    print("\n--- Mean Euclidean Distance (MED) ---")
    print(get_med_statistics(all_run_centroids, gt_by_label).round(2))

    # ── Bounding Box IoU ─────────────────────────────────────
    print("\n--- Bounding Box IoU ---")
    print(get_iou_statistics(all_run_bboxes, gt_by_label).round(2))

    # ── Recall & Precision ────────────────────────────────────
    print("\n--- Object Recall and Precision ---")
    print(get_recall_statistics(all_run_centroids, gt_by_label, threshold=1.0).round(2))

    # ── Confusion Matrix ─────────────────────────────────────
    fig_norm, fig_raw = get_confusion_matrix(all_run_centroids, gt_by_label, threshold=1.0)
    
    #print confusion matrix, uncomment if desired
    # if fig_norm:
    #     fig_norm.savefig(f"confusion_matrix_{method}_norm.png", dpi=150)
    #     fig_raw.savefig(f"confusion_matrix_{method}_raw.png",  dpi=150)

if __name__ == "__main__":
    #runs evalation on both methods
    run_evaluation("hydra")
    run_evaluation("clio")