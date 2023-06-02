# DICES Dataset: Diversity in Conversational AI Evaluation for Safety #

# Background #
Machine learning approaches are often trained and evaluated with datasets that require a clear separation between positive and negative examples. This approach overly simplifies the natural subjectivity present in many tasks and content items. It also obscures the inherent diversity in human perceptions and opinions. Often tasks that attempt to preserve the variance in content and diversity in humans are quite expensive and laborious. To fill in this gap and facilitate more in-depth model performance analyses we propose the **DICES dataset** **- a unique dataset with diverse perspectives on safety of AI generated conversations**. We focus on the task of safety evaluation of conversational AI systems. The  DICES  dataset contains detailed demographics information about each rater, extremely high replication of unique ratings per conversation to ensure statistical significance of further analyses and encodes rater votes as distributions across different demographics to allow for in-depth explorations of different rating aggregation strategies. 

This dataset is well suited to observe and measure variance, ambiguity and diversity in the context of safety of conversational AI. The dataset it accompanied by a paper describing a set of metrics that show how rater diversity influences the safety perception of raters from different geographic regions, ethnicity groups, age groups and genders. The goal of the DICES datasetis to be used as a shared benchmark for safety evaluation of conversational AIsystems.

# Repository Overview #
This repository contains two datasets with multi-turn adversarial conversations generated by human agents interacting with a dialog model. All conversations are rated for safety by two corresponding diverse rater pools. Details for all safety ratings can be found in the corresponding README.md files.

- **Dataset 990:** `990/diverse_safety_adversarial_dialog_990.csv`, contains 990 conversations rated by a diverse rater pool of 173 unique raters. Each conversation is rated with three safety top-level categories and one overall conversation comprehension question. Raters were recruited so that the number of raters for each conversation was balanced by gender (Man, Woman) and locale (US, India). Each rater rated only a sample of the conversation. Each conversation has 40 unique ratings. Total number of rows in this dataset is 72104.

- **Dataset 350:** `350/diverse_safety_adversarial_dialog_350.csv`, contains 350 conversations rated by a diverse rater pool of 123 unique raters. Each conversation is rated with five safety top-level categories and one overall comprehension question of the conversation. Raters were recruited were balanced by gender (man or woman), race/ethnicity (White, Black, Latine, Asian, Multiracial) and each rater rated all items.  Each rater rated all conversations. Each conversation has 104 unique ratings. Total number of rows in this dataset is 43050.

Each directory contains the dataset csv file and a README.md file describing the schema for the corresponding dataset.
