import pandas as pd
import plotly.express as px

df = pd.read_csv('/Users/stephenxie/Documents/GitHub/homework_spring2026/submission/src/hw1_imitation/wandb_export_2026-01-28T18_07_12.681-08_00.csv')

fig = px.line(df, x = 'Step', y = 'mse - train_loss', title='MSE Train Loss', labels={'Step': 'Training Steps', 'mse - train_loss': 'Training Loss'})
fig.write_image("MSE_loss.png")

df = pd.read_csv('/Users/stephenxie/Documents/GitHub/homework_spring2026/submission/src/hw1_imitation/wandb_export_2026-01-28T18_14_18.507-08_00.csv')

fig = px.line(df, x = 'Step', y = 'flow - train_loss', title='Flow-matching Train Loss', labels={'Step': 'Training Steps', 'flow - train_loss': 'Training Loss'})
fig.write_image("Flow_loss.png")

df = pd.read_csv('/Users/stephenxie/Documents/GitHub/homework_spring2026/submission/src/hw1_imitation/wandb_export_2026-01-28T18_09_25.581-08_00.csv')

fig = px.line(df, x = 'Step', y = 'mse - learning_rate', title='MSE Learning Rate', labels={'Step': 'Training Steps', 'mse - learning_rate': 'Learning Rate'})
fig.write_image("MSE_lr.png")
fig = px.line(df, x = 'Step', y = 'mse - learning_rate', title='Flow-matching Learning Rate', labels={'Step': 'Training Steps', 'mse - learning_rate': 'Learning Rate'})
fig.write_image("Flow_lr.png")

df = pd.read_csv('/Users/stephenxie/Documents/GitHub/homework_spring2026/submission/src/hw1_imitation/wandb_export_2026-01-28T18_10_38.150-08_00.csv')

fig = px.line(df, x = 'Step', y = 'mse - eval/mean_reward', title='MSE Mean Reward', labels={'Step': 'Training Steps', 'mse - eval/mean_reward': 'Mean Reward'})
fig.write_image("MSE_reward.png")


df = pd.read_csv('/Users/stephenxie/Documents/GitHub/homework_spring2026/submission/src/hw1_imitation/wandb_export_2026-01-28T18_13_12.842-08_00.csv')

fig = px.line(df, x = 'Step', y = 'flow - eval/mean_reward', title='Flow-matching Mean Reward', labels={'Step': 'Training Steps', 'flow - eval/mean_reward': 'Mean Reward'})
fig.write_image("Flow_reward.png")
