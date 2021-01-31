from rvalue_model import *
import subprocess as cmd
import pandas as pd
import pickle

# Run the model
x = run_govukmodel()

# Specify folder to save the most recent model and plots
dirname = './results/recent'

# Save the output figures in a "recent"
# folder for updating the readme page
fig_plot = x.plot_model(return_figure=True, reference_level=1000)
plt.savefig(dirname+'/forecast.jpeg')
fig_cov = x.plot_covariance(return_figure=True)
plt.savefig(dirname+'/correlation.jpeg')

# Save the model
f = open(dirname + "/model.pkl", "wb")
pickle.dump({'model': x}, f)
f.close()

# Commit results to github
today_str = str(pd.Timestamp.today().date())
cp = cmd.run("git add .", check=True, shell=True)
message = "Results "+today_str
cp = cmd.run(f"git commit -m '{message}'", check=True, shell=True)
cp = cmd.run("git push -u origin master -f", check=True, shell=True)