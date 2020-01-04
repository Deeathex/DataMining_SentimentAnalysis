training = [[["sunny", 85, 85, "weak"], "no"],
				[["sunny", 80, 90, "strong"], "no"],
				[["overcast", 83, 78, "weak"], "yes"],
				[["rain", 70, 96, "weak"], "yes"],
				[["rain", 68, 80, "weak"], "yes"],
				[["rain", 65, 70, "strong"], "no"],
				[["overcast", 64, 65, "strong"], "yes"],
				[["sunny", 72, 95, "weak"], "no"],
				[["sunny", 69, 70, "weak"], "yes"],
				[["rain", 75, 80, "weak"], "yes"],
				[["sunny", 75, 70, "strong"], "yes"],
				[["overcast", 72, 90, "strong"], "yes"],
				[["overcast", 81, 75, "weak"], "yes"],
				[["rain", 71, 80, "strong"], "no"],
				]

ch = {"outlook":1, "temperature": 1, "humidity": 1, "wind": 1}
attr = {0: "outlook", 1:"temperature", 2:"humidity", 3:"wind"}
attributes_index = {"outlook": 0, "temperature": 1, "humidity": 2, "wind": 3}
attributes = ["outlook", "temperature", "humidity", "wind"]
continuous = {"outlook": 0, "temperature": 1, "humidity": 1, "wind": 0}
testing = []
labels = ["pozitive", "negative"]