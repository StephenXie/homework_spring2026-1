import pandas as pd
import plotly.express as px

DATA_DIR = '/Users/stephenxie/Documents/GitHub/homework_spring2026/hw2/data'

eval_return_csv = f'{DATA_DIR}/wandb_export_2026-03-02T18_27_43.549-08_00.csv'
baseline_loss_csv = f'{DATA_DIR}/wandb_export_2026-03-02T18_28_27.287-08_00.csv'
lunarlander_csv = f'{DATA_DIR}/wandb_export_2026-03-02T18_35_20.549-08_00.csv'

LABELS = {
    'cheetah_baseline': 'baseline',
    'cheetah': 'no baseline',
}


def extract_series(df, metric, label_map, env_prefix='HalfCheetah-v4_'):
    """Pull per-run metric series, dropping NaN rows per run."""
    records = []
    for col in df.columns:
        if metric not in col or '__' in col:
            continue
        run_name = col.split(' - ')[0].split(env_prefix)[1]
        short = run_name.rsplit('_sd', 1)[0]
        label = label_map.get(short, short)
        sub = df[['Step', col]].dropna()
        sub = sub.rename(columns={col: metric})
        sub['Experiment'] = label
        records.append(sub)
    return pd.concat(records, ignore_index=True)


# --- HalfCheetah: Baseline Loss ---
import os
if os.path.exists(baseline_loss_csv):
    df_loss = pd.read_csv(baseline_loss_csv)
    runs_loss = extract_series(df_loss, 'Baseline Loss', LABELS)
    fig = px.line(
        runs_loss, x='Step', y='Baseline Loss', color='Experiment',
        title='HalfCheetah: Baseline Loss vs. Environment Steps',
        labels={'Step': 'Train EnvstepsSoFar', 'Baseline Loss': 'Baseline Loss'},
    )
    fig.write_image(f'{DATA_DIR}/halfcheetah_baseline_loss.png')

# --- HalfCheetah: Eval Average Return ---
if os.path.exists(eval_return_csv):
    df_eval = pd.read_csv(eval_return_csv)
    runs_eval = extract_series(df_eval, 'Eval_AverageReturn', LABELS)
    fig = px.line(
        runs_eval, x='Step', y='Eval_AverageReturn', color='Experiment',
        title='HalfCheetah: Average Return vs. Environment Steps',
        labels={'Step': 'Train EnvstepsSoFar', 'Eval_AverageReturn': 'Average Return'},
    )
    fig.write_image(f'{DATA_DIR}/halfcheetah_eval_return.png')


# --- LunarLander: λ comparison ---
LAMBDA_LABELS = {
    'lunar_lander_lambda0': 'λ = 0',
    'lunar_lander_lambda0.95': 'λ = 0.95',
    'lunar_lander_lambda0.98': 'λ = 0.98',
    'lunar_lander_lambda0.99': 'λ = 0.99',
    'lunar_lander_lambda1': 'λ = 1',
}

if os.path.exists(lunarlander_csv):
    df_ll = pd.read_csv(lunarlander_csv)
    runs_ll = extract_series(df_ll, 'Eval_AverageReturn', LAMBDA_LABELS,
                             env_prefix='LunarLander-v2_')

    lambda_order = ['λ = 0', 'λ = 0.95', 'λ = 0.98', 'λ = 0.99', 'λ = 1']
    runs_ll['Experiment'] = pd.Categorical(
        runs_ll['Experiment'], categories=lambda_order, ordered=True)

    fig = px.line(
        runs_ll, x='Step', y='Eval_AverageReturn', color='Experiment',
        title='LunarLander-v2: Effect of λ on Task Performance',
        labels={
            'Step': 'Train EnvstepsSoFar',
            'Eval_AverageReturn': 'Eval Average Return',
        },
        category_orders={'Experiment': lambda_order},
    )
    fig.update_layout(
        xaxis_title='Train EnvstepsSoFar',
        yaxis_title='Eval Average Return',
        legend_title='λ',
        template='plotly_white',
        width=900,
        height=550,
    )
    fig.write_image(f'{DATA_DIR}/lunarlander_lambda.png')


# --- InvertedPendulum: default vs tuned hyperparameters ---
pendulum_csv = f'{DATA_DIR}/wandb_export_2026-03-02T19_39_54.146-08_00.csv'

PENDULUM_LABELS = {
    '193411': 'tuned',
    '193849': 'default',
}

if os.path.exists(pendulum_csv):
    df_pend = pd.read_csv(pendulum_csv)
    records = []
    for col in df_pend.columns:
        if 'Train_AverageReturn' not in col or '__' in col:
            continue
        timestamp = col.split('_20260302_')[1].split(' - ')[0]
        label = PENDULUM_LABELS.get(timestamp, timestamp)
        sub = df_pend[['Step', col]].dropna()
        sub = sub.rename(columns={col: 'Train_AverageReturn'})
        sub['Experiment'] = label
        records.append(sub)
    runs_pend = pd.concat(records, ignore_index=True)

    fig = px.line(
        runs_pend, x='Step', y='Train_AverageReturn', color='Experiment',
        title='InvertedPendulum-v4: Default vs. Tuned Hyperparameters',
        labels={
            'Step': 'Train EnvstepsSoFar',
            'Train_AverageReturn': 'Average Return',
        },
        category_orders={'Experiment': ['default', 'tuned']},
    )
    fig.update_layout(
        xaxis_title='Train EnvstepsSoFar',
        yaxis_title='Average Return',
        legend_title='Hyperparameters',
        template='plotly_white',
        width=900,
        height=550,
    )
    fig.write_image(f'{DATA_DIR}/pendulum_hp_comparison.png')
