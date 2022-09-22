# Modelling with Leaf Reflectance Spectroscopy data

### Quickstart

1. Create an environment with Python 3.9, and install requirements (`pip -r requirements.txt`)
2. Execute `python start.py` and follow prompts in the terminal.
3. At the end of each program execution, the displayed results are saved
   to a folder called `results`.  The PNG contains the confusion matrix, and 
   the CSV file with the same name contains the performance summary (accuracy et al.).

-----

#### Using a different CSV of data

For example, if you'd like to manually filter the rows in a way that's not offered within the 
program itself.

Requirements: 
- The CSV file must have the same column headers -- no changes to the names or order, and no
  adding/removing columns.
- It should use the same naming convention in the `species` column,
  i.e. _2 charaters for genus, 2 characters for subgenus, 2 characters for
  subsection, a period, and the species name_.

To load your own custom CSV, specify the file path as an argument, e.g.:

`python start.py data/my_new_file.csv`.

-----
### Credits
Field Museum of Natural History & Grainger Bioinformatics Center

Ryan Fuller (Postdoctoral Researcher), Beth McDonald (Machine Learning Engineer), 
Dr. Rick Ree (Curator & Section Head of Flowering Plants)

Please contact Ryan Fuller for research inquiries and licensing.