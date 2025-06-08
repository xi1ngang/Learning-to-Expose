import json

scenarios_indoor = ['bright', 'dim', 'neutral']
scenarios_outdoor = ['outdoor']
selected_models_indoor = dict()
selected_models_outdoor = dict()
for scenario in scenarios_indoor:
    for scene_index in range(1,6):
        file_path = f".\\higher_flicker\\results_higherflicker_{scenario}_scene{scene_index}.json"
        with open(file_path, 'r') as file:
            data = json.load(file)
            for item in data:
                episode_number = item['episode_number']
                if episode_number in selected_models_indoor:
                    selected_models_indoor[episode_number] += 1
                else:
                    selected_models_indoor[episode_number] = 1

# for scenario in scenarios_outdoor:
#     for scene_index in range(1,16):
#         file_path = f".\\new_2100_model\\results_2100_{scenario}_scene{scene_index}.json"
#         with open(file_path, 'r') as file:
#             data = json.load(file)
#             for item in data:
#                 episode_number = item['episode_number']
#                 if episode_number in selected_models_outdoor:
#                     selected_models_outdoor[episode_number] += 1
#                 else:
#                     selected_models_outdoor[episode_number] = 1


sorted_indoor_models = dict(sorted(selected_models_indoor.items(), key=lambda item: item[1], reverse=True))
# sorted_outdoor_models = dict(sorted(selected_models_outdoor.items(), key=lambda item: item[1], reverse=True))

# outdoor_model = list(sorted_outdoor_models.keys())[0:20]
indoor_model = list(sorted_indoor_models)[0:20]


common = {}

# for key in sorted_outdoor_models:
#     if key in sorted_indoor_models:
#         common[key] = (sorted_indoor_models[key], sorted_outdoor_models[key])


print(sorted_indoor_models)
