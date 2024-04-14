import math
import numpy as np

y = [-1, 0, 3, 0, -1, 0, -3, 0, 0, 0, -2, 2, 2, 0, -2, 2, 0, 0, 4, 0, -2, -1, -1, 3, 3, -2, -3, 3, 1, 1, 0, 2, 4, -2, 3, -1]


unique_values = np.unique(y)
print(f"unique values: {unique_values}")
print(f"length: {len(unique_values)}")
is_continuous = False
is_multiclass = False

# Check if any unique value is a non-integer float.
if len(unique_values) > 4:
    print(f"\nCONTINUOUS")
    is_continuous = True

for value in unique_values:
    if isinstance(value, float) and not value.is_integer(): 
        print(f"\nCONTINUOUS")
        is_continuous = True
        break

# If not continuous, it's binary or multiclass.
if not is_continuous:
    print(f"\nMULTICLASS")
    is_multiclass = len(unique_values) > 2  # Multiclass if more than 2 unique values.
                                            # and less than 4
if not is_continuous and not is_multiclass:
    print(f"\nBINARY")
num_classes = math.inf if is_continuous else len(unique_values)

