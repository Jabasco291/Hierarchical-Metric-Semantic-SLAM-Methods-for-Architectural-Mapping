import json
import sys
from pathlib import Path
import get_gt3
import pandas as pd
import stats_functions

def get_sgraph_statistics():

    gt_rooms = get_gt3.get_gt_rooms()
    gt_room_centroids = [v['centroid'] for v in gt_rooms.values()]

    room_centroids = {}
    rooms_table = []
    node_stats = []

    for i in range(1,6):
        #get graph
        file_path = f"../graphs/s-graphs/sgraphs_eval_20260426_1657/run_{i}_sgraph.json"
        path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(file_path).expanduser()
        
        with open(path) as f:
            graph = json.load(f)

        #get nodes centroids for room recognition
        room_centroids[i] = [
                node['attributes']['position'][:3]
                for node in graph['nodes']
                    if node['layer'] == 3 and node['attributes']['position'][2] > 0
        ]

        #get room node count
        num_rooms = sum(1 for node in graph['nodes'] if node['layer'] == 3)
        rooms_table.append({"Run": f"Run {i}", "Room Nodes": num_rooms})

        #layer to number
        mapping = {"keyframe": 1, "floor": 2, "room": 3, "wall": 4, "plane": 5}
        #get node counts for each layer
        counts = {
            "Run": i,
            "Total Nodes": len(graph['nodes']),
            "Keyframes": sum(1 for node in graph['nodes'] if node.get('layer') == mapping["keyframe"]),
            "Rooms": sum(1 for node in graph['nodes'] if node.get('layer') == mapping["room"]),
            "Planes": sum(1 for node in graph['nodes'] if node.get('layer') == mapping["plane"])
        }

        node_stats.append(counts)


    #build nodes dataframe
    df_nodes = pd.DataFrame(node_stats).set_index("Run")

    overall_row = df_nodes.mean().to_frame().T
    overall_row.index = ['Mean (Overall)']

    std_row = df_nodes.std().to_frame().T
    std_row.index = ['Std']

    cv_row = pd.DataFrame(std_row.values / overall_row.values, 
                        columns=df_nodes.columns, 
                        index=['CV'])

    df_nodes_final = pd.concat([df_nodes, overall_row, std_row, cv_row])

    print("--- Node Summary Table ---")
    print(df_nodes_final.round(2))

    #build rooms dataframe
    df_rooms = pd.DataFrame(rooms_table).set_index("Run")

    overall_row = df_rooms.mean().to_frame().T
    overall_row.index = ['Mean (Overall)']

    std_row = df_rooms.std().to_frame().T
    std_row.index = ['Std']

    cv_row = pd.DataFrame(std_row.values / overall_row.values, columns=df_rooms.columns, index=['CV'])

    df_rooms_final = pd.concat([df_rooms, overall_row, std_row, cv_row])

    print("--- Room Node Table ---")
    print(df_rooms_final.round(2))

    
    df_rooms_final = stats_functions.get_room_statistics(gt_room_centroids, room_centroids, 4)

    print("--- Room Recognition ---")
    print(df_rooms_final.round(2))

get_sgraph_statistics()