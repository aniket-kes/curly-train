import numpy as np 
import pandas as pd 
 
# Define a new small dataset 
data_dict = { 
    'Symptom1': ['Yes', 'Yes', 'No', 'Yes', 'No', 'No'], 
    'Symptom2': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'], 
    'Symptom3': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes'], 
    'Disease': ['DiseaseA', 'DiseaseA', 'DiseaseB', 'DiseaseA', 'DiseaseB', 'DiseaseB'] 
} 
 
data = pd.DataFrame(data_dict) 
 
# Display the dataset 
print("Data:\n", data) 
 
# Prepare concepts and target arrays 
concepts = np.array(data.iloc[:, :-1]) 
target = np.array(data.iloc[:, -1]) 
 
print("\nTarget:\n", target) 
print("\nConcepts:\n", concepts) 
 
# Candidate Elimination Algorithm 
def learn(concepts, target): 
    # Initialize specific and general hypotheses 
    specific_h = concepts[0].copy() 
    general_h = [['?' for _ in range(len(specific_h))] for _ in range(len(specific_h))] 
 
    print("\nInitialization of specific_h and general_h") 
    print("Specific_h: ", specific_h) 
    print("General_h: ", general_h) 
 
    # Iterate through each instance 
    for i, h in enumerate(concepts): 
        if target[i] == "DiseaseA":  # Positive example 
            print(f"\nIf instance is Positive \n Step {i+1}") 
            for x in range(len(specific_h)): 
                if h[x] != specific_h[x]: 
                    specific_h[x] = '?' 
                    for j in range(len(general_h)): 
                        general_h[j][x] = '?' 
        elif target[i] == "DiseaseB":  # Negative example 
            print(f"\nIf instance is Negative \n Step {i+1}") 
            for x in range(len(specific_h)): 
                if h[x] != specific_h[x]: 
                    for j in range(len(general_h)): 
                        general_h[j][x] = specific_h[x] 
                else: 
                    for j in range(len(general_h)): 
                        general_h[j][x] = '?' 
 
        print(f"\nAfter Step {i+1}") 
        print("Specific_h: ", specific_h) 
        print("General_h: ", general_h) 
 
    # Remove overly general hypotheses 
    general_h = [h for h in general_h if any(attr != '?' for attr in h)] 
 
    return specific_h, general_h 
 
# Run the Candidate Elimination algorithm 
s_final, g_final = learn(concepts, target) 
 
# Print final results 
print("\nFinal Specific_h:", s_final, sep="\n") 
print("Final General_h:", g_final, sep="\n")
