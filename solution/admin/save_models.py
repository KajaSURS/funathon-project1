# %% 
# Run it from bash with uv run solution/admin/save_models.py intermediate_solutions/3_RF.py rf_model_final rf_model.joblib
import sys
import os

# Append parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import subprocess
import argparse

from utils import upload_file_s3

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("input_script", help="Path to the input script")
parser.add_argument("model_name", help="Name of the model variable")
parser.add_argument("s3_name", help="S3 destination filename")
args = parser.parse_args()

input_script = args.input_script
model_name = args.model_name
s3_name = args.s3_name

# Debug
# input_script = "intermediate_solutions/3_RF.py"
# model_name = "rf_model_final"
# s3_name = "rf_model_test.joblib"

# Step 1: Copy script to temp.py
os.makedirs("temp", exist_ok=True)
shutil.copy(input_script, "temp/temp.py")

# Step 2: Add lines to temp script to dump model
with open("temp/temp.py", "a") as f:
    f.write("\n\n# Save model\n")
    f.write("import joblib\n")
    f.write(f"joblib.dump({model_name}, 'temp/temp.joblib')\n")

# Step 3: Run temp.py
subprocess.run(["uv", "run", "temp/temp.py"])

# Step 4: Upload temp.joblib to S3
upload_file_s3("temp/temp.joblib", s3_name)

# Step 5: Cleanup temporary files
os.remove("temp/temp.py")
os.remove("temp/temp.joblib")

# %%