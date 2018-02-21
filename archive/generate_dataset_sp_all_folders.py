"""This is just a quick and dirty solution to calculate from
all inputs in db folder
"""

import os

from generate_dataset_sp import main


database_folder = os.path.normpath("data_base/")
result_folder = os.path.normpath("input_data")

# aquire list of available source dirs
sources = [
    f for f in os.listdir(database_folder) if os.path.isdir(
        os.path.join(database_folder, f)
    )
]

# for each source do calculation. if it fails, who cares..
for source in sources:
    try:
        main(
            source=os.path.join(database_folder, source),
            destination=os.path.join(result_folder, source),
            number_of_processes=8
        )
    except:
        pass