from rvalue_model import *
import subprocess as cmd
import pandas as pd

x = run_govukmodel()


today_str = str(pd.Timestamp.today().date())
#commit results to github
cp = cmd.run("git add .", check=True, shell=True)
message = "Results "+today_str
cp = cmd.run(f"git commit -m '{message}'", check=True, shell=True)
cp = cmd.run("git push -u origin master -f", check=True, shell=True)