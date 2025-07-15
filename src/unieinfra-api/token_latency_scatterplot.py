import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(file: str):
    df_uploaded = pd.read_csv(file)

    # Create a new column for total tokens
    df_uploaded["total_tokens"] = df_uploaded["prompt_tokens"] + df_uploaded["completion_tokens"]

    # Set figure size for all plots
    plt.figure(figsize=(16, 5))

    # Prompt tokens vs. Request time
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df_uploaded, x="prompt_tokens", y="request_time", hue="model_name")
    plt.title("Prompt Tokens vs. Request Time")
    plt.xlabel("Prompt Tokens")
    plt.ylabel("Request Time")

    # Completion tokens vs. Request time
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df_uploaded, x="completion_tokens", y="request_time", hue="model_name")
    plt.title("Completion Tokens vs. Request Time")
    plt.xlabel("Completion Tokens")
    plt.ylabel("Request Time")

    # Total tokens vs. Request time
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df_uploaded, x="total_tokens", y="request_time", hue="model_name")
    plt.title("Total Tokens vs. Request Time")
    plt.xlabel("Total Tokens")
    plt.ylabel("Request Time")

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import fire
    fire.Fire(main)