import joblib
import numpy as np

model = joblib.load('trained_model.pkl')

feature_names = ['Total_Minutes', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PTS', 'PTOV', 'SFD', 'PGA', 'AND1', 'TS%', 'USG%', 'WS', 'BPM', 'VORP', 'ORtg']
feature_min = [200, 0.3, 0.8, 0, 0, 0.4, 0, 1, 0, 0, 2, 0, 0.312, 10, -2.1, -9, -2, 79]
feature_max = [3125, 11.5, 23.6, 9.8, 12.3, 16, 11.7, 33.9, 243, 385, 1947, 105, 0.7, 38.8, 16.4, 10, 8.2, 147]

# Define the PER min-max range
per_min = 0  # Adjust if PER has a specific minimum
per_max = 30  # Adjust if PER has a specific maximum


# Function to normalize features
def normalize_features(raw_features):
    normalized = [(raw_features[i] - feature_min[i]) / (feature_max[i] - feature_min[i]) for i in range(len(raw_features))]
    return normalized


# Function to denormalize the predicted PER
def denormalize_per(normalized_per):
    real_per = normalized_per * (per_max - per_min) + per_min
    return real_per


# Function to predict PER based on real-world stats
def predict_per(raw_features):
    normalized_features = normalize_features(raw_features) # Normalize the features
    input_array = np.array(normalized_features).reshape(1, -1)
    normalized_per = model.predict(input_array)[0] # Make prediction (normalized PER)
    real_per = denormalize_per(normalized_per) # Convert normalized PER back to real-world PER
    return real_per

# Function to calculate and display differences
def display_with_difference(real_per, predicted_per):
    difference = round(abs(real_per - predicted_per), 1)
    print(f"Difference: {difference}")


# 97 Olajuwon 22.7
example_input3 = [2852, 9.3, 18.3, 4.5, 5.7, 9.2, 3, 23.2, 113, 186, 590, 48, 0.558, 30.4, 9.1, 3.5, 3.9, 105] # Input a player's statistics
predicted_per3 = predict_per(example_input3) # Gives PER prediction based on player's statistics
print(f"\n1997 Hakeem Olajuwon\nReal PER: 22.7\nPredicted PER: {predicted_per3:.1f}") # Output predicted PER
display_with_difference(22.7, predicted_per3) # Computes difference between real and predicted PER to demonstrate the model's accuracy

# 98 Jordan 25.2
example_input1 = [3181, 10.7, 23.1, 6.9, 8.8, 5.8, 3.5, 28.7, 89, 278, 607, 70, 0.533, 33.7, 15.8, 6.9, 7.1, 114]
predicted_per1 = predict_per(example_input1)
print(f"\n1998 Michael Jordan\nReal PER: 25.2\nPredicted PER: {predicted_per1:.1f}")
display_with_difference(25.2, predicted_per1)

# 99 Iverson 22.2
example_input11 = [1990, 9.1, 22, 7.4, 9.9, 4.9, 4.6, 26.8, 81, 163, 458, 31, 0.508, 32.6, 7.2, 5.7, 3.9, 105]
predicted_per11 = predict_per(example_input11)
print(f"\n1999 Allen Iverson\nReal PER: 22.2\nPredicted PER: {predicted_per11:.1f}")
display_with_difference(22.2, predicted_per11)

# 00 Mutombo 19.4
example_input12 = [2984, 3.9, 7, 3.6, 5.1, 14.1, 1.3, 11.5, 32, 143, 229, 34, 0.621, 13.8, 9.9, 2.4, 3.3, 116]
predicted_per12 = predict_per(example_input12)
print(f"\n2000 Dikembe Mutombo\nReal PER: 19.4\nPredicted PER: {predicted_per12:.1f}")
display_with_difference(19.4, predicted_per12)

# 01 O'Neal 30.2
example_input13 = [2924, 11, 19.2, 6.7, 13.1, 12.7, 3.7, 28.7, 72, 365, 644, 115, 0.574, 31.6, 14.9, 7.7, 7.1, 114]
predicted_per13 = predict_per(example_input13)
print(f"\n2001 Shaquille O'Neal\nReal PER: 30.2\nPredicted PER: {predicted_per13:.1f}")
display_with_difference(30.2, predicted_per13)

# 02 Payton 22.9
example_input4 = [3301, 9, 19.2, 3.3, 4.1, 4.8, 9, 22.1, 106, 119, 1666, 34, 0.526, 27.2, 12.6, 5.1, 5.9, 114]
predicted_per4 = predict_per(example_input4)
print(f"\n2002 Gary Payton\nReal PER: 22.9\nPredicted PER: {predicted_per4:.1f}")
display_with_difference(22.9, predicted_per4)

# 03 McGrady 30.3
example_input14 = [2954, 11.1, 24.2, 7.7, 9.7, 6.5, 5.5, 32.1, 95, 303, 943, 66, 0.564, 35.2, 16.1, 10.5, 9.3, 116]
predicted_per14 = predict_per(example_input14)
print(f"\n2003 Tracy McGrady\nReal PER: 30.3\nPredicted PER: {predicted_per14:.1f}")
display_with_difference(30.3, predicted_per14)

# 04 Stojaković 21.8
example_input5 = [3264, 8.2, 17.1, 4.9, 5.2, 6.3, 2.1, 24.2, 58, 137, 381, 32, 0.624, 23.9, 13.5, 3.9, 4.9, 120]
predicted_per5 = predict_per(example_input5)
print(f"\n2004 Peja Stojaković\nReal PER: 21.8\nPredicted PER: {predicted_per5:.1f}")
display_with_difference(21.8, predicted_per5)

# 05 Duncan 27
example_input15 = [2203, 7.8, 15.8, 4.6, 6.9, 11.1, 2.7, 20.3, 44, 200, 404, 41, 0.540, 28.9, 11.2, 7.6, 5.4, 112]
predicted_per15 = predict_per(example_input15)
print(f"\n2005 Tim Duncan\nReal PER: 27.0\nPredicted PER: {predicted_per15:.1f}")
display_with_difference(27.0, predicted_per15)

# 06 Nowitzki 28.1
example_input6 = [3089, 9.3, 19.3, 6.7, 7.4, 9, 2.8, 26.6, 71, 219, 502, 57, 0.589, 30, 17.7, 8.1, 7.9, 123]
predicted_per6 = predict_per(example_input6)
print(f"\n2006 Dirk Nowitzki\nReal PER: 28.1\nPredicted PER: {predicted_per6:.1f}")
display_with_difference(28.1, predicted_per6)

# 07 Garnett 24.1
example_input16 = [2995, 8.4, 17.6, 5.5, 6.6, 12.8, 4.1, 22.4, 76, 195, 696, 38, 0.546, 27.4, 10.7, 5.4, 5.5, 110]
predicted_per16 = predict_per(example_input16)
print(f"\n2007 Kevin Garnett\nReal PER: 24.1\nPredicted PER: {predicted_per16:.1f}")
display_with_difference(24.1, predicted_per16)

# 08 Paul 28.3
example_input17 = [3006, 7.9, 16.1, 4.2, 4.9, 4, 11.6, 21.1, 93, 119, 2123, 32, 0.576, 25.7, 17.8, 10.4, 9.3, 125]
predicted_per17 = predict_per(example_input17)
print(f"\n2008 Chris Paul\nReal PER: 28.3\nPredicted PER: {predicted_per17:.1f}")
display_with_difference(28.3, predicted_per17)

# 09 Howard 25.4
example_input18 = [2821, 7.1, 12.4, 6.4, 10.7, 13.8, 1.4, 20.6, 33, 378, 296, 94, 0.600, 26.1, 13.8, 4.5, 4.7, 113]
predicted_per18 = predict_per(example_input18)
print(f"\n2009 Dwight Howard\nReal PER: 25.4\nPredicted PER: {predicted_per18:.1f}")
display_with_difference(25.4, predicted_per18)

# 10 Gallinari 14.8
example_input19 = [2747, 4.8, 11.4, 3.1, 3.8, 4.9, 1.7, 15.1, 35, 121, 306, 18, 0.575, 19.3, 5.6, 0.5, 1.7, 113]
predicted_per19 = predict_per(example_input19)
print(f"\n2010 Danilo Gallinari\nReal PER: 14.8\nPredicted PER: {predicted_per19:.1f}")
display_with_difference(14.8, predicted_per19)

# 11 Ellis 18.6
example_input7 = [3227, 9.1, 20.1, 4.3, 5.4, 3.5, 5.6, 24.1, 104, 183, 1057, 40, 0.536, 28.1, 6, 0.8, 2.3, 107]
predicted_per7 = predict_per(example_input7)
print(f"\n2011 Monta Ellis\nReal PER: 18.6\nPredicted PER: {predicted_per7:.1f}")
display_with_difference(18.6, predicted_per7)

# 12 Aldridge 22.7
example_input20 = [1994, 8.8, 17.1, 4.1, 5.0, 8, 2.4, 21.7, 22, 128, 340, 30, 0.560, 27, 7, 3, 2.5, 113]
predicted_per20 = predict_per(example_input20)
print(f"\n2012 LaMarcus Aldridge\nReal PER: 22.7\nPredicted PER: {predicted_per20:.1f}")
display_with_difference(22.7, predicted_per20)

# 13 George 16.8
example_input21 = [2972, 6.2, 14.9, 2.8, 3.5, 7.6, 4.1, 17.4, 96, 107, 736, 26, 0.531, 23.5, 9, 3.8, 4.4, 104]
predicted_per21 = predict_per(example_input21)
print(f"\n2013 Paul George\nReal PER: 16.8\nPredicted PER: {predicted_per21:.1f}")
display_with_difference(16.8, predicted_per21)

# 14 Durant 29.8
example_input22 = [3122, 10.5, 20.8, 8.7, 9.9, 7.4, 5.5, 32, 123, 311, 1004, 68, 0.635, 33, 19.2, 10.2, 9.6, 123]
predicted_per22 = predict_per(example_input22)
print(f"\n2014 Kevin Durant\nReal PER: 29.8\nPredicted PER: {predicted_per22:.1f}")
display_with_difference(29.8, predicted_per22)

# 15 Millsap 20
example_input8 = [2390, 6.1, 12.7, 3.5, 4.6, 7.8, 3.1, 16.7, 53, 142, 529, 32, 0.565, 23.8, 8.3, 3.4, 3.3, 109]
predicted_per8 = predict_per(example_input8)
print(f"\n2015 Paul Millsap\nReal PER: 20.0\nPredicted PER: {predicted_per8:.1f}")
display_with_difference(20.0, predicted_per8)

# 16 Thomas 21.5
example_input23 = [2644, 7.2, 16.9, 5.8, 6.6, 3, 6.2, 22.2, 107, 212, 1187, 44, 0.562, 29.6, 9.7, 4.3, 4.2, 113]
predicted_per23 = predict_per(example_input23)
print(f"\n2016 Isaiah Thomas\nReal PER: 21.5\nPredicted PER: {predicted_per23:.1f}")
display_with_difference(21.5, predicted_per23)

# 17 Westbrook 30.6
example_input24 = [2802, 10.2, 24, 8.8, 10.4, 10.7, 10.4, 31.6, 297, 350, 1886, 65, 0.554, 41.7, 13.1, 11.1, 9.3, 112]
predicted_per24 = predict_per(example_input24)
print(f"\n2017 Russell Westbrook\nReal PER: 30.6\nPredicted PER: {predicted_per24:.1f}")
display_with_difference(30.6, predicted_per24)

# 18 James 28.6
example_input2 = [3026, 10.5, 19.3, 4.7, 6.5, 8.6, 9.1, 27.5, 217, 252, 1836, 85, 0.621, 31.6, 14, 8.7, 8.2, 118]
predicted_per2 = predict_per(example_input2)
print(f"\n2018 LeBron James\nReal PER: 28.6\nPredicted PER: {predicted_per2:.1f}")
display_with_difference(28.6, predicted_per2)

# 19 Harden 30.6
example_input9 = [2868, 10.8, 24.5, 9.7, 11, 6.6, 7.5, 36.1, 214, 358, 1390, 76, 0.616, 40.5, 15.2, 11, 9.3, 118]
predicted_per9 = predict_per(example_input9)
print(f"\n2019 James Harden\nReal PER: 30.6\nPredicted PER: {predicted_per9:.1f}")
display_with_difference(30.6, predicted_per9)

# 20 Dončić 27.6
example_input25 = [2047, 9.5, 20.6, 7, 9.2, 9.4, 8.8, 28.8, 153, 248, 1320, 80, 0.585, 36.8, 8.8, 8.4, 5.4, 116]
predicted_per25 = predict_per(example_input25)
print(f"\n2020 Luka Dončić\nReal PER: 27.6\nPredicted PER: {predicted_per25:.1f}")
display_with_difference(27.6, predicted_per25)

# 21 Young 23.0
example_input26 = [2125, 7.7, 17.7, 7.7, 8.7, 3.9, 9.4, 25.3, 179, 183, 1403, 26, 0.589, 33, 7.2, 3.7, 3, 117]
predicted_per26 = predict_per(example_input26)
print(f"\n2021 Trae Young\nReal PER: 23.0\nPredicted PER: {predicted_per26:.1f}")
display_with_difference(23.0, predicted_per26)

# 22 Jokic 32.8
example_input10 = [2476, 10.3, 17.7, 5.1, 6.3, 13.8, 7.9, 27.1, 173, 188, 1356, 72, 0.661, 31.9, 15.2, 13.7, 9.8, 126]
predicted_per10 = predict_per(example_input10)
print(f"\n2022 Nikola Jokic\nReal PER: 32.8\nPredicted PER: {predicted_per10:.1f}")
display_with_difference(32.8, predicted_per10)

# 23 McDaniels 12.0
example_input27 = [2416, 4.7, 9.1, 1.3, 1.8, 3.9, 1.9, 12.1, 47, 75, 351, 23, 0.611, 15.8, 4.4, -1.2, 0.5, 115]
predicted_per27 = predict_per(example_input27)
print(f"\n2023 Jaden McDaniels\nReal PER: 12.0\nPredicted PER: {predicted_per27:.1f}")
display_with_difference(12.0, predicted_per27)

# 24 Leonard 23.2
example_input28 = [2330, 9, 17.1, 3.7, 4.2, 6.1, 3.6, 23.7, 53, 122, 607, 44, 0.626, 26.5, 8.9, 5.5, 4.4, 124]
predicted_per28 = predict_per(example_input28)
print(f"\n2024 Kawhi Leonard\nReal PER: 23.2\nPredicted PER: {predicted_per28:.1f}")
display_with_difference(23.2, predicted_per28)


# Function to calculate the average of differences
def calculate_average_difference(real_per_list, predicted_per_list):
    differences = [abs(real - predicted) for real, predicted in zip(real_per_list, predicted_per_list)]
    average_difference = sum(differences) / len(differences)
    return round(average_difference, 2)

real_per_list = [22.7, 25.2, 22.2, 19.4, 30.2, 22.9, 30.3, 21.8, 27.0, 28.1, 24.1, 28.3, 25.4, 14.8, 18.6, 22.7, 16.8, 29.8, 20.0, 21.5, 30.6, 28.6, 30.6, 27.6, 23.0, 32.8, 12.0, 23.2]
predicted_per_list = [22.9, 26.2, 21.9, 16.5, 29.7, 20.4, 27.3, 17.8, 26.8, 27.5, 23.6, 25.3, 24.3, 10.2, 14.3, 21.6, 14.7, 29.1, 17.8, 18.7, 32.2, 27.3, 28.3, 27.8, 22.4, 35.8, 9.3, 21.8]

average_difference = calculate_average_difference(real_per_list, predicted_per_list)
print(f"\nThe average difference is: {average_difference}\n")

