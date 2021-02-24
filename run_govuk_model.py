from rvalue_model import *
import subprocess as cmd
import pandas as pd
import pickle
import time

# Check for new data. Abort if none
x_newdatacheck = rmodel_govuk()
new_data = x_newdatacheck.check_todays_update()
if new_data is False:
    raise Exception('No new data for '+str(pd.Timestamp.today().date())+'. Aborting')

# Run the model
x = run_govukmodel()

time.sleep(15)

# Specify folder to save the most recent model and plots
dirname = './results/recent'

# Save the output figures in a "recent"
# folder for updating the readme page
fig_plot = x.plot_model(return_figure=True, reference_level=1000)
plt.savefig(dirname+'/forecast.png',dpi=1200)
fig_cov = x.plot_covariance(return_figure=True)
plt.savefig(dirname+'/correlation.png',dpi=500)


# Plot the rolling r tracker
fig = plt.figure()
ax1 = fig.add_subplot(111)
x.plot_r_estimate(fig, ax1)
ax1.set_title('Rolling Reproduction Factor Calculation')

xann = {'date':[pd.Timestamp(2020, 3, 23),
        pd.Timestamp(2020, 11, 5),
        pd.Timestamp(2021, 1, 2)],
        'label':['Lockdown 1','2','3']}
idx = 1
for date, lab in zip(xann['date'],xann['label']):
    if idx == 1:
        label = 'UK Lockdowns'
    else:
        label = None
    ax1.axvline(date,ls=':',label=label,color='purple')
    idx += 1
plt.legend()
plt.tight_layout()
plt.savefig(dirname+'/rolling_r_plot.png',dpi=1000)


# Save the model
f = open(dirname + "/model.pkl", "wb")
pickle.dump({'model': x}, f)
f.close()

# Commit results to github
today_str = str(pd.Timestamp.today().date())
cp = cmd.run("git pull", check=True, shell=True)
cp = cmd.run("git add .", check=True, shell=True)
message = "Results "+today_str
cp = cmd.run(f"git commit -m '{message}'", check=True, shell=True)
cp = cmd.run("git push -u origin master -f", check=True, shell=True)