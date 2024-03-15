import matplotlib.colors as mcolors
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json

# Function to read TSV file


def read_tsv(file_path):
    return pd.read_csv(file_path, delimiter='\t')

# Function to read JSON file


def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Paths to the files
demo_data_path = 'demo_data.tsv'
text_data_path = 'text_data.tsv'
desc_data_path = 'desc_data.json'
meta_data_path = 'meta_data.json'

# Reading the files
demo_data = read_tsv(demo_data_path)
text_data = read_tsv(text_data_path)
desc_data = read_json(desc_data_path)
meta_data = read_json(meta_data_path)

# Display the first few rows of the dataframes to check
print(text_data.head())
text_data.columns
demo_data.columns
# Display the dictionaries
print(desc_data)
print(meta_data)


# Assuming demo_data has already been read into a DataFrame
# Let's describe the 'EEG Papers' column to see its distribution

# First, we'll check the basic statistics and distribution of the 'EEG Papers' column
eeg_papers_description = demo_data['EEG Papers'].describe()

# To get a frequency table of the 'EEG Papers' column to understand the distribution better
eeg_papers_frequency = demo_data['EEG Papers'].value_counts().sort_index()

eeg_papers_description, eeg_papers_frequency

# Start building the table


start_col_pos = demo_data.columns.get_loc("Predictions_2")
end_col_pos = demo_data.columns.get_loc("Predictions_42") + 1

# Step 1: Identify and select the prediction columns
# Assuming prediction columns are named from 'Predictions_2' to 'Predictions_42'
prediction_columns = demo_data.columns[start_col_pos:end_col_pos]

# Step 2 & 3: Calculate averages and valid percentages for each prediction item
averages = demo_data[prediction_columns].mean()
valid_counts = demo_data[prediction_columns].count()  # Count non-NaN values
# Total number of rows for percentage calculation
total_counts = len(demo_data)
valid_percentages = (valid_counts / total_counts) * 100

# Step 4: Calculate confidence intervals for each prediction item


def calculate_ci(column):
    confidence_level = 0.95
    degrees_freedom = column.count() - 1
    mean = column.mean()
    standard_error = stats.sem(column, nan_policy='omit')
    ci_range = stats.t.ppf((1 + confidence_level) / 2.,
                           degrees_freedom) * standard_error
    return (mean - ci_range, mean + ci_range)


ci_bounds = demo_data[prediction_columns].apply(calculate_ci)

# Organizing results into a DataFrame
results_df = pd.DataFrame({
    'Average Prediction (Years)': averages,
    'Valid Percentage (%)': valid_percentages,
    'CI Lower Bound': ci_bounds.apply(lambda x: x[0]),
    'CI Upper Bound': ci_bounds.apply(lambda x: x[1])
})


# Create a new column for descriptions in results_df
results_df['Description'] = [desc_data.get(
    col, "") for col in prediction_columns]

# Update the 'Description' column by keeping only the part of each description after the first '-'
results_df['Description'] = results_df['Description'].apply(
    lambda x: x.split('-', 1)[-1] if '-' in x else x)

print(results_df)

predict_item_dict = {'EEG is routinely used in the diagnosis and monitoring of sleep disorders, such as sleep apnea and insomnia.': ['Diagnosis & monitoring of sleep disorders', 'Clinical applications', '#8a2a86', '1'],
                     'Portable, consumer-grade EEG devices become widely available for personal use, such as for relaxation or focus training.': ['Personal use common', 'Clinical applications', '#8a2a86', '1'],
                     'EEG data analysis enables real-time, reliable detection of brain abnormalities such as seizures or tumours.': ['Detection of brain abnormalities', 'Clinical applications', '#8a2a86', '1'],
                     'EEG devices are utilised to study the impact of various pharmacological interventions on brain activity.': ['Study impact of pharmacological interventions', 'Clinical applications', '#8a2a86', '1'],
                     'Automatic preprocessing pipelines outperform human-in-the-loop workflows.': ['Automatic pipelines outperform human-in-the-loop', 'Unified standards', '#0100ff', '2'],
                     'There are widely agreed definitions on how to collect, analyse and interpret resting state EEG data.': ['Widely agreed definitions for resting state data', 'Unified standards', '#0100ff', '2'],
                     'There is a scientific consensus on best practices for preprocessing EEG data.': ['Consensuses on best practices for preprocessing', 'Unified standards', '#0100ff', '2'],
                     'High quality EEG data can be acquired from anyone, irrespective of their hair type and other physical characteristics.': ['High quality data possible from everyone', 'Accessibility', '#ff8000', '3'],
                     'EEG is used as a reliable tool for detecting and monitoring the progression of traumatic brain injuries (TBI) and concussions.': ['Progression of traumatic brain injuries', 'Clinical applications', '#8a2a86', '1'],
                     'Closed-loop EEG systems are developed for adaptive and personalized neuromodulation therapies, such as deep brain stimulation and transcranial magnetic stimulation.': ['Personalized neuromodulation therapies', 'Clinical applications', '#8a2a86', '1'],
                     'EEG-based brain-computer interfaces are widely adopted in gaming and virtual reality applications.': ['EEG-based BCIs for gaming & VR', 'Accessibility', '#ff8000', '3'],
                     'Ethical considerations are incorporated into the development of brain-computer interfaces, taking into account potential implications on personal identity, autonomy, and agency.': ['Ethical BCI development', 'Unified standards', '#0100ff', '2'],
                     'EEG-guided neurofeedback therapy becomes a standard treatment option for mental health disorders such as ADHD and anxiety.': ['Standard treatment for mental health', 'Clinical applications', '#8a2a86', '1'],
                     'EEG-based technology enables early detection and intervention for learning disabilities in children.': ['Early detection & intervention for learning disabilities', 'Clinical applications', '#8a2a86', '1'],
                     'EEG systems are sufficiently robust and user-friendly that untrained individuals are able to use them to reliably collect high quality data.': ['Untrained users can collect high quality data', 'Accessibility', '#8a2a86', '3'],
                     'There is widely agreed ontology mapping ERP components to cognitive processes.': ['Ontology for ERPs mapped to cognitive processes', 'Decoding', '#0e9a24', '4'],
                     'EEG technology enables accurate and efficient detection of early-stage neurodegenerative diseases, such as Alzheimer\'s and Parkinson\'s.': ['Early detection of neurodegeneration', 'Clinical applications', '#8a2a86', '1'],
                     'EEG technology is sufficiently low cost that it is widely accessible to researchers and clinicians to use in every part of the globe.': ['Widely accessible in every part of the globe', 'Accessibility', '#8a2a86', '3'],
                     'Improvements in machine learning algorithms allow for seamless decoding of  cognitive states and emotions using EEG data.': ['Decoding of cognitive states & emotions', 'Decoding', '#0e9a24', '4'],
                     'Machine learning-driven EEG analysis contributes to the development of personalized treatment plans for mental health disorders.': ['Personalized treatment for mental health disorders', 'Clinical applications', '#8a2a86', '1'],
                     'EEG-based technology is used as a primary tool for communication by individuals with severe motor disabilities or locked-in syndrome.': ['Primary communication tool for motor disabilities', 'Clinical applications', '#8a2a86', '1'],
                     'EEG technology allows for the non-invasive assessment of brain health during prenatal and neonatal stages.': ['Brain health during prenatal & neonatal stages', 'Clinical applications', '#8a2a86', '1'],
                     'EEG technology is integrated into workplace safety protocols to monitor worker fatigue and cognitive function in high-risk environments.': ['Monitor worker fatigue in high-risk environments', 'Decoding', '#0e9a24', '4'],
                     'Advancements in EEG technology enable the development of precise and individualized neuromodulation therapies for various neurological and psychiatric disorders.': ['Precise & individualized neuromodulation therapies', 'Clinical applications', '#8a2a86', '1'],
                     'Consumer-grade EEG devices are widely used to monitor and enhance cognitive performance in professional and educational settings.': ['Consumer-devices monitor & enhance cognition', 'Accessibility', '#8a2a86', '3'],
                     'Advancements in EEG technology enable remote and telemedicine applications for neurological assessments and treatments.': ['Remote neurological assessments & treatments', 'Clinical applications', '#8a2a86', '1'],
                     'Guidelines for the responsible use of EEG technology in marketing, advertising, and entertainment industries are established, preventing the potential manipulation of users\' emotions, cognitive states, and decision-making processes.': ['Guidelines for marketing & entertainment', 'Unified standards', '#0100ff', '2'],
                     'EEG technology is widely used with daily-use devices like smartphones and smartwatches, providing continuous brain monitoring.': ['Integrated with smartphones', 'Accessibility', '#8a2a86', '3'],
                     'EEG-driven passive biometric authentication is integrated into AR, VR, and metaverse platforms, ensuring secure access to personal data and virtual assets.': ['Biometric identification', 'Decoding', '#0e9a24', '4'],
                     'The integration of EEG-based brain-computer interfaces with AR, VR, and metaverse platforms enables intuitive and seamless control of virtual environments using thought alone.': ['Control of virtual environments', 'Decoding', '#0e9a24', '4'],
                     'Integration of EEG data in AR, VR, and metaverse platforms leads to adaptive environments that respond to users\' cognitive states, emotions, and mental workload.': ['Virtual environments in aid of cognitive support', 'Decoding', '#8a2a86', '1'],
                     'EEG is widely used as a lie detector': ['Used as a lie detector', 'Decoding', '#0e9a24', '4'],
                     'EEG is used to identify students in need of additional support to unlock their full educational potential.': ['Identify students in need of additional support', 'Clinical applications', '#8a2a86', '1'],
                     'EEG is used widely across the globe (including in lower and middle income countries) as an indicator of brain health.': ['Brain health measure used across the globe', 'Accessibility', '#8a2a86', '3'],
                     'EEG can be used to read the content of your dreams': ['Reading content of dreams', 'Decoding', '#0e9a24', '4'],
                     'EEG can be used to read the content of your long-term memories': ['Reading long-term memories', 'Decoding', '#0e9a24', '4']
                     }


# We need to sort 'results_df' by 'Average Prediction (Years)' in ascending order
sorted_results_df = results_df.sort_values('Average Prediction (Years)')

# THIS IS WHERE THE CHANGE HAPPENS


# Now, we can safely assign the new column names since the order will match
sorted_results_df.index = new_column_order

# Renaming the index to reflect the new labels
sorted_results_df.rename_axis('Prediction', inplace=True)

# Let's display the sorted and renamed DataFrame
sorted_results_df

# Assuming the start date is January 1, 2024
start_date = pd.Timestamp('2024-01-01')

# Let's convert the years to dates starting from 01.01.2024.
# We'll treat the 'Average Prediction (Years)' as the number of years away from the start date.
# The 'CI Lower Bound' will be the start of the event, and 'CI Upper Bound' will be the end of the event.


# Since timedelta does not directly support years (due to variable number of days in a year),
# we will convert the duration to days approximating 1 year as 365.25 days (including leap years)

# Adjusting for leap years in the calculation
sorted_results_df['Time (from)'] = start_date + \
    pd.to_timedelta(
    sorted_results_df['CI Lower Bound']*365.25, unit='D')
sorted_results_df['Time (to)'] = start_date + pd.to_timedelta(
    sorted_results_df['CI Upper Bound']*365.25, unit='D')


#####
# Define a function to calculate the date given a number of years away from the start date
def years_to_date(years, start_date):
    # Calculate the number of days (accounting for leap years)
    days = years * 365.25
    # Add the number of days to the start date
    return start_date + pd.to_timedelta(days, unit='D')

# Define a function to format the date in the specified "Time Graphics" format


def format_timeline_date(date):
    if date.year < 0:
        return f"-{-date.year} {date.month} {date.day}"
    elif date.year == 0:
        return f"-0 {date.month} {date.day}"
    else:
        return f"{date.year} {date.month} {date.day}"


# Apply the conversion and formatting functions to the lower and upper CI bounds
sorted_results_df['Time (from)'] = sorted_results_df['CI Lower Bound'].apply(
    lambda x: years_to_date(x, start_date))
sorted_results_df['Time (to)'] = sorted_results_df['CI Upper Bound'].apply(
    lambda x: years_to_date(x, start_date))

# Format the dates for the timeline
sorted_results_df['Time (from)'] = sorted_results_df['Time (from)'].apply(
    format_timeline_date)
sorted_results_df['Time (to)'] = sorted_results_df['Time (to)'].apply(
    format_timeline_date)

# Show the results
sorted_results_df[['Time (from)', 'Time (to)']]

timeline_data = sorted_results_df[['Time (from)', 'Time (to)']].copy()
# Now we create a new DataFrame that matches the structure expected by the timeline graphics site.
# Use the Prediction names as titles
timeline_data['Title'] = timeline_data.index
# Directly use the descriptions added earlier
timeline_data['Description'] = sorted_results_df['Description'].values
timeline_data['Image'] = ""  # Placeholder for image URLs
timeline_data['Video'] = ""  # Placeholder for video URLs
timeline_data['Position'] = "top"  # Placeholder for position
timeline_data['Type'] = "thin_line"  # Placeholder for type
timeline_data['Common color'] = ""  # Placeholder for common color
timeline_data['Text color'] = "#000000"  # Placeholder for text color
timeline_data['Tags'] = ""  # Placeholder for tags
timeline_data['Tags IDs'] = ""  # Placeholder for tags IDs

# Let's display the new DataFrame that is structured for the timeline graphics site.
# Reset index to remove the prediction names from index
timeline_data.reset_index(drop=True, inplace=True)
timeline_data

# To reverse the colormap so that warmer colors indicate fewer respondents and cooler colors more respondents,
# we can simply invert the colormap.

# Inverting the colormap
cmap_reversed = plt.get_cmap('coolwarm_r')

# Normalize the valid_percentages again, this time we don't need to change it, just apply the reversed colormap
# Mapping the normalized percentages to colors using the reversed colormap and converting to hex format
sorted_results_df['Common Color'] = sorted_results_df['Valid Percentage (%)'].apply(
    lambda x: mcolors.to_hex(cmap_reversed(norm(x))))

# Reset index
sorted_results_df.reset_index(drop=True, inplace=True)

# Update the timeline_data DataFrame with the 'Common Color' column using the reversed colormap
timeline_data['Common Color'] = sorted_results_df['Common Color']


# Creating a new legend with the reversed colormap
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cbar = fig.colorbar(plt.cm.ScalarMappable(
    norm=norm, cmap=cmap_reversed), cax=ax, orientation='horizontal')
cbar.set_label('Response Percent (%)')

# Save the colorbar as a .jpg image file with the reversed colormap
legend_path = 'C:/Users/richard1/OneDrive - Tartu Ülikool/KÄSIKIRJAD/#EEGManyLabs/EEG100/heatmap_legend.jpg'
plt.savefig(legend_path, format='jpg', dpi=300)
plt.close()  # Close the figure to prevent it from displaying in the notebook

# Return the path to the saved legend image with the reversed colormap
legend_path


# Export the timeline_data DataFrame to a .csv file
csv_path = 'timeline_data.csv'  # Specify your desired path
timeline_data.to_csv(csv_path, index=False)

# START
# Step 1: Identify and select the prediction columns
start_col_pos = demo_data.columns.get_loc("Predictions_2")
end_col_pos = demo_data.columns.get_loc("Predictions_42") + 1
prediction_columns = demo_data.columns[start_col_pos:end_col_pos]

# Step 2 & 3: Calculate averages and valid percentages for each prediction item
averages = demo_data[prediction_columns].mean()
valid_counts = demo_data[prediction_columns].count()  # Count non-NaN values
# Total number of rows for percentage calculation
total_counts = len(demo_data)
valid_percentages = (valid_counts / total_counts) * 100

# Step 4: Calculate confidence intervals for each prediction item


def calculate_ci(column):
    confidence_level = 0.95
    degrees_freedom = column.count() - 1
    mean = column.mean()
    standard_error = stats.sem(column, nan_policy='omit')
    ci_range = stats.t.ppf((1 + confidence_level) / 2.,
                           degrees_freedom) * standard_error
    return (mean - ci_range, mean + ci_range)


ci_bounds = demo_data[prediction_columns].apply(calculate_ci)

# Organizing results into a DataFrame
results_df = pd.DataFrame({
    'Average Prediction (Years)': averages,
    'Valid Percentage (%)': valid_percentages,
    'CI Lower Bound': ci_bounds.apply(lambda x: x[0]),
    'CI Upper Bound': ci_bounds.apply(lambda x: x[1])
})

# We need to sort 'results_df' by 'Average Prediction (Years)' in ascending order
sorted_results_df = results_df.sort_values('Average Prediction (Years)')

# Using 'desc_data' to update 'results_df' with descriptions and match with 'predict_item_dict'
for col in prediction_columns:
    # Extract description for each prediction column from 'desc_data'
    # Assume each column in 'desc_data' matches the prediction column names
    # Strip everything before and including the '-' from the descriptions
    # Taking the first row as example
    key = desc_data[col].split('-', 1)[1][1:]

    # Use the processed string as a key for 'predict_item_dict' to get new values
    if key in predict_item_dict:
        values = predict_item_dict[key]
        # Assuming the index of 'sorted_results_df' matches with the prediction column order
        # Update 'sorted_results_df' with new values based on the 'predict_item_dict'
        # Find the index matching the current column
        idx = sorted_results_df.index[sorted_results_df.index == col]
        sorted_results_df.loc[idx, 'Title'] = values[0]
        sorted_results_df.loc[idx, 'Tagging Category'] = values[1]
        sorted_results_df.loc[idx, 'Tag Color'] = values[2]
        sorted_results_df.loc[idx,
                              'Tags'] = f"{values[3]}:{values[2]}:{values[1]}"
        sorted_results_df.loc[idx, 'Description'] = key
        sorted_results_df.loc[idx,
                              'Position'] = 'top' if values[1] == 'Clinical applications' else 'bottom'


# Proceed with sorting, additional calculations, or formatting as required
# Assuming the start date is January 1, 2024
start_date = pd.Timestamp('2024-01-01')

# Let's convert the years to dates starting from 01.01.2024.
# We'll treat the 'Average Prediction (Years)' as the number of years away from the start date.
# The 'CI Lower Bound' will be the start of the event, and 'CI Upper Bound' will be the end of the event.


# Calculate the start and end dates for each event based on the confidence interval bounds
# sorted_results_df['Time (from)'] = start_date + pd.to_timedelta(sorted_results_df['CI Lower Bound'], unit='Y')
# sorted_results_df['Time (to)'] = start_date + pd.to_timedelta(sorted_results_df['CI Upper Bound'], unit='Y')

# Since timedelta does not directly support years (due to variable number of days in a year),
# we will convert the duration to days approximating 1 year as 365.25 days (including leap years)

# Adjusting for leap years in the calculation
sorted_results_df['Time (from)'] = start_date + \
    pd.to_timedelta(
    sorted_results_df['CI Lower Bound']*365.25, unit='D')
sorted_results_df['Time (to)'] = start_date + pd.to_timedelta(
    sorted_results_df['CI Upper Bound']*365.25, unit='D')


#####
# Define a function to calculate the date given a number of years away from the start date
def years_to_date(years, start_date):
    # Calculate the number of days (accounting for leap years)
    days = years * 365.25
    # Add the number of days to the start date
    return start_date + pd.to_timedelta(days, unit='D')

# Define a function to format the date in the specified "Time Graphics" format


def format_timeline_date(date):
    if date.year < 0:
        return f"-{-date.year} {date.month} {date.day}"
    elif date.year == 0:
        return f"-0 {date.month} {date.day}"
    else:
        return f"{date.year} {date.month} {date.day}"


# Apply the conversion and formatting functions to the lower and upper CI bounds
sorted_results_df['Time (from)'] = sorted_results_df['CI Lower Bound'].apply(
    lambda x: years_to_date(x, start_date))
sorted_results_df['Time (to)'] = sorted_results_df['CI Upper Bound'].apply(
    lambda x: years_to_date(x, start_date))

# Format the dates for the timeline
sorted_results_df['Time (from)'] = sorted_results_df['Time (from)'].apply(
    format_timeline_date)
sorted_results_df['Time (to)'] = sorted_results_df['Time (to)'].apply(
    format_timeline_date)

# Show the results
sorted_results_df[['Time (from)', 'Time (to)']]

timeline_data = sorted_results_df[['Time (from)', 'Time (to)']].copy()
# Now we create a new DataFrame that matches the structure expected by the timeline graphics site.
# Use the Prediction names as titles
timeline_data['Title'] = sorted_results_df['Title'].values
# Directly use the descriptions added earlier
timeline_data['Description'] = sorted_results_df['Description'].values
timeline_data['Position'] = sorted_results_df['Position'].values
timeline_data['Image'] = ""  # Placeholder for image URLs
timeline_data['Video'] = ""  # Placeholder for video URLs
timeline_data['Type'] = "thin_line"  # Placeholder for type
timeline_data['Text color'] = "#000000"  # Placeholder for text color
# Placeholder for tags
timeline_data['Tags'] = sorted_results_df['Tags'].values
# Placeholder for tags IDs
timeline_data['Tags IDs'] = sorted_results_df['Tags'].values

# Let's display the new DataFrame that is structured for the timeline graphics site.
# Reset index to remove the prediction names from index
timeline_data.reset_index(drop=True, inplace=True)
timeline_data

# To reverse the colormap so that warmer colors indicate fewer respondents and cooler colors more respondents,
# we can simply invert the colormap.

# Inverting the colormap
cmap_reversed = plt.get_cmap('coolwarm_r')

# Re-normalize if needed (using the same 'norm' as before)
norm = mcolors.Normalize(vmin=sorted_results_df['Valid Percentage (%)'].min(),
                         vmax=sorted_results_df['Valid Percentage (%)'].max())

# Normalize the valid_percentages again, this time we don't need to change it, just apply the reversed colormap
# Mapping the normalized percentages to colors using the reversed colormap and converting to hex format
sorted_results_df['Common Color'] = sorted_results_df['Valid Percentage (%)'].apply(
    lambda x: mcolors.to_hex(cmap_reversed(norm(x))))

# Reset index
sorted_results_df.reset_index(drop=True, inplace=True)

# Update the timeline_data DataFrame with the 'Common Color' column using the reversed colormap
timeline_data['Common Color'] = sorted_results_df['Common Color']


# Creating a new legend with the reversed colormap
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cbar = fig.colorbar(plt.cm.ScalarMappable(
    norm=norm, cmap=cmap_reversed), cax=ax, orientation='horizontal')
cbar.set_label('Response Percent (%)')

# Save the colorbar as a .jpg image file with the reversed colormap
legend_path = 'C:/Users/richard1/OneDrive - Tartu Ülikool/KÄSIKIRJAD/#EEGManyLabs/EEG100/heatmap_legend.jpg'
plt.savefig(legend_path, format='jpg', dpi=300)
plt.close()  # Close the figure to prevent it from displaying in the notebook

# Return the path to the saved legend image with the reversed colormap
legend_path

new_column_order = ['Time (from)', 'Time (to)', 'Title', 'Description', 'Image',
                    'Video', 'Position', 'Type', 'Common Color', 'Text color', 'Tags', 'Tags IDs']

# Reordering the columns in timeline_data
timeline_data = timeline_data.reindex(columns=new_column_order)


# Export the timeline_data DataFrame to a .csv file
csv_path = 'timeline_data.csv'  # Specify your desired path
timeline_data.to_csv(csv_path, index=False)
