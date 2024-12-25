import streamlit as st
import pandas as pd
import numpy as np
import os
from huggingface_hub import InferenceClient
from pathlib import Path


# Initialize constants and paths
DATASETS_DIR = Path(".datasets")
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_FILE = DATASETS_DIR / "original.csv"
CLEANED_FILE = DATASETS_DIR / "cleaned.csv"
PROBABILITY_FILE = DATASETS_DIR / "probability.csv"
OUTPUT_FILE = DATASETS_DIR / "output.txt"
CHAT_FILE = DATASETS_DIR / "chat_history.txt"


HF_TOKEN = os.getenv("secret2")
if not HF_TOKEN:
    st.error("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
    st.stop()

llm_model = "Qwen/Qwen2.5-Coder-32B-Instruct"
client = InferenceClient(api_key=HF_TOKEN)

def save_uploaded_file(uploaded_file):
    """Save the uploaded file and return the file path."""
    try:
        if not uploaded_file.name.endswith('.csv'):
            st.error("Only CSV files are supported.")
            return None

        file_path = ORIGINAL_FILE
        with file_path.open("wb") as f:
            f.write(uploaded_file.getvalue())

        return str(file_path)
    except Exception as e:
        st.error(f"Error saving uploaded file: {str(e)}")
        return None

def preprocess_data(label_column=None, columns_to_drop=None):
    """Preprocess the data with user input and return the cleaned DataFrame."""
    try:
        # Load the data
        data = pd.read_csv(ORIGINAL_FILE, low_memory=False)

        # Clean column names by stripping leading/trailing whitespaces
        data.columns = data.columns.str.strip()

        # Print cleaned column names for debugging
        st.write("Cleaned Columns in DataFrame:", data.columns.tolist())

        # Step 1: Rename the label column if provided
        if label_column:
            data = data.rename(columns={label_column: " Label"})
        else:
            data[" Label"] = None

        # Step 2: Drop specified columns if provided
        if columns_to_drop:
            # Step 2.1: Convert user input into a list and sanitize
            columns_to_drop = columns_to_drop.split(',') # Convert input to list by splitting commas
            st.write("Original User Input:", columns_to_drop)

            # Step 2.2: Sanitize user input (remove extra spaces, quotes, etc.)
            columns_to_drop_cleaned = []
            for col in columns_to_drop:
                # Strip spaces and remove single quotes, then append to the cleaned list
                cleaned_col = col.strip().replace("'", "").strip()
                columns_to_drop_cleaned.append(cleaned_col)
            
            st.write("Sanitized Columns to Drop:", columns_to_drop_cleaned)

            # Step 2.3: Check if the cleaned columns exist in the DataFrame
            columns_to_drop_final = [col for col in columns_to_drop_cleaned if col in data.columns]

            # Find any missing columns that are not found in the DataFrame
            missing_columns = [col for col in columns_to_drop_cleaned if col not in columns_to_drop_final]
            if missing_columns:
                st.error(f"Error: These columns are not present in the DataFrame (after sanitization): {missing_columns}")
                return None

            # Step 2.4: Drop the cleaned and validated columns
            data.drop(columns=columns_to_drop_final, errors='raise', inplace=True)
            st.write(f"Successfully dropped columns: {columns_to_drop_final}")

        else:
            st.markdown("No columns to drop.")

        # Step 3: Handle missing and infinite values
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        # Save the cleaned data
        data.to_csv(CLEANED_FILE, index=False)

        # Display the shape and preview of the data
        st.success("Data preprocessing complete.")
        st.write(f"Total Rows: {data.shape[0]}, Total Columns: {data.shape[1]}")
        return data

    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        return None



def analyze_and_infer():
    try:
        if not CLEANED_FILE.exists():
            st.error("Please preprocess data first.")
            return None

        ddos_data = pd.read_csv(CLEANED_FILE)
        ddos_data_without_label = ddos_data.drop([" Label"], axis=1)

        
        numeric_columns = ddos_data_without_label.select_dtypes(include=[np.number]).columns
        max_values = ddos_data_without_label[numeric_columns].max()
        min_values = ddos_data_without_label[numeric_columns].min()
        median_values = ddos_data_without_label[numeric_columns].median()
        mean_values = ddos_data_without_label[numeric_columns].mean()
        variance_values = ddos_data_without_label[numeric_columns].var()

        
        result = pd.DataFrame({
            'Max': max_values,
            'Min': min_values,
            'Median': median_values,
            'Mean': mean_values,
            'Variance': variance_values
        })

        
        know_prompt_pre = (
            "1. Supposed that you are now an experienced network traffic data analysis expert.\n"
            "2. You need to help me analyze the data in the DDoS dataset and determine whether the data is DDoS traffic or normal traffic.\n"
            "3. Next, I will give you the maximum, minimum, median, mean, and variance of all the data under each label in the data set, which may help you make your judgment.\n"
            "4. Do deep analysis, it's your work, and provide accurate answers along with the given task:\n"
        )

        know_prompt_back = ""
        for i, idx in enumerate(result.index):
            know_prompt_back += (f"{idx}'s max: {float(result.iloc[i].Max):0.1f}, min: {float(result.iloc[i].Min):0.1f}, "
                                 f"median: {float(result.iloc[i].Median):0.1f}, mean: {float(result.iloc[i].Mean):0.1f}, "
                                 f"variance: {float(result.iloc[i].Variance):0.1f}, "
                                 )

        know_prompt = know_prompt_pre + know_prompt_back + "JUST PROVIDE YES AS YOUR REPONSE AND JUST OBEY YOUR NETWORK TRAFFIC DATA ANALYST ROLE"

        with open(OUTPUT_FILE, "a") as f:
            f.write(know_prompt + "\n")

        messages = [
            {'role': 'user', 'content': know_prompt}
        ]

        completion = client.chat.completions.create(model=llm_model, messages=messages, max_tokens=10000)
        response = completion.choices[0].message.content
        with open(OUTPUT_FILE, "a") as f:
            f.write(response + "\n")
        st.write("Model's analysis:")
        st.write(response)

        
        with open(PROBABILITY_FILE, "w") as f:
            f.write("SRNO,DDOS,BENIGN,DDOS,BENIGN\n")

        
        for i, row in ddos_data.iterrows():
            token_prompt_back = ""
            for j, col in enumerate(ddos_data_without_label.columns):
                token_prompt_back += f"{col}:{row[col]}, "

            token_prompt = (
                "\n1. Next, I will give you a piece of data about network traffic information.\n"
                "\n2. You need to first tell me the probability of this data belonging to DDoS traffic data and normal traffic data directly and separately, and express the probability in the format of [0.xxx,0.xxx].\n"
                "\n3. The first number is the probability of DDoS traffic like [0.xxx] only where x are any numeral, and the second number is the probability of normal traffic like [0.xxx] only where x are any numeral.\n"
                "\n4. KEEP IN MIND: [BOTH NUMBERS ARE DECIMALS BETWEEN 0 AND 1, AND THE SUM OF THE TWO NUMBERS IS 1].\n"
                "\n5. CLEARLY NOTE THAT YOU HAVE TO PROVIDE ONLY PROBABILITY OF EACH DDOS AND BENIGN GIVEN THAT TOTAL PROBABILITY WHEN SUMMED MUST BE 1.\n"
                "\n6. AND YES IT MUST BE LIKE 0<=P([DDOS,BENGNIN])<=1 BUT YOU HAVE TO PROVIDE PROBABILITY IN THIS FORM: [0.xxx,0.xxx] ONLY.\n"
                "\n7. Let's think step by step and explain the reasons for your judgment. The characteristics of its network traffic data packets are\n"
            ) + "\n" + token_prompt_back + "\n" + "\n PLS GIVE PROBABILITY IN FORM [0.xyz , 0.pqr ] where x,y,z may be same or diff , also p,q,r may be same or diff ALWAYS REMEMBER THAT total probability = 1 and probability of event lies between 0 and 1 as 0<p(X)<1\n"

            st.write(f"ROW {i} DATA:")
            st.write(f"Token Prompt: {token_prompt}")

            with open(OUTPUT_FILE, "a") as f:
                f.write(token_prompt + "\n")

            messages = [
                {'role': 'user', 'content': token_prompt}
            ]

            completion = client.chat.completions.create(model=llm_model, messages=messages, max_tokens=10000)
            response = completion.choices[0].message.content
            with open(OUTPUT_FILE, "a") as f:
                f.write(response + "\n")
            st.write(f"Model's response for row {i}:")
            st.write(response)

            
            try:
                start_index = response.find("[")
                end_index = response.find("]", start_index)
                if start_index != -1 and end_index != -1:
                    prob_str = response[start_index:end_index+1]
                    prob_list = prob_str.replace("[", "").replace("]", "").split(",")

                    # Check if the number of probabilities is correct
                    if len(prob_list) != 2:
                        st.write(f"Row {i}: Invalid probability format.")
                        with open(PROBABILITY_FILE, "a") as f:
                            f.write(f"{i},None,None,None\n")
                        continue

                    # Convert probabilities to floats
                    try:
                        attack_prob, benign_prob = float(prob_list[0].strip()), float(prob_list[1].strip())
                    except ValueError:
                        st.write(f"Row {i}: Invalid probability values.")
                        with open(PROBABILITY_FILE, "a") as f:
                            f.write(f"{i},None,None,None\n")
                        continue

                    # Check if probabilities are within the valid range
                    if not (0.0 <= attack_prob <= 1.0) or not (0.0 <= benign_prob <= 1.0):
                        st.write(f"Row {i}: Probabilities out of range.")
                        with open(PROBABILITY_FILE, "a") as f:
                            f.write(f"{i},None,None,None\n")
                        continue

                    # Write the probabilities to the file
                    with open(PROBABILITY_FILE, "a") as f:
                        f.write(f"{i},{attack_prob},{benign_prob},{prob_str}\n")

            except Exception as e:
                st.write(f"Row {i}: Error parsing probabilities: {str(e)}")
                with open(PROBABILITY_FILE, "a") as f:
                    f.write(f"{i},None,None,None\n")

            st.write("--------------------------------------------------------")

        st.success("Analysis complete.")

        final_prompt = "Are the analysis and probabilities correct? If not, what improvements are needed?"
        with open(OUTPUT_FILE, "a") as f:
            f.write(final_prompt + "\n")
        messages = [{'role': 'user', 'content': final_prompt}]
        completion = client.chat.completions.create(model=llm_model, messages=messages, max_tokens=10000)
        response = completion.choices[0].message.content
        with open(OUTPUT_FILE, "a") as f:
            f.write(response + "\n")
        st.write("Model's confirmation:")
        st.write(response)

    except Exception as e:
        st.error(f"Error during analysis and inference: {str(e)}")
        return None

def save_chat_history(chat_history, file_path):
    """Save the chat history to a text file."""
    with open(file_path, "w") as f:
        for message in chat_history:
            role = message["role"].capitalize()
            content = message["content"]
            f.write(f"{role}: {content}\n")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("DrLLM")
st.title("Prompt-Enhanced Distributed Denial-of-Service Resistance Method with Large Language Models")

st.markdown("""
**Features:**
- **Upload Dataset**: Upload your network traffic dataset in CSV format.
- **Preprocess Data**: Clean and prepare the data for analysis.
- **Analyze Data**: Get insights and probability estimates.
- **Chat with the Model**: Engage in a conversation for further insights.
""")

st.markdown("Upload your network traffic dataset:")

uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"], key="file_uploader")

if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        st.success("File uploaded successfully.")

label_column = st.text_input("Label Column name", placeholder="e.g., Label", key="label_column", max_chars=50)
st.markdown("Do you want to drop the Columns (Yes/No)")
answer = st.text_input("Type :", placeholder="e.g , Yes , yes, No , no" ,key="answer",max_chars=5)
if answer == "yes" or answer == "Yes" or answer == "YES":
    columns_to_drop = st.text_input("Columns to drop", placeholder="e.g., Unnamed: 0, Flow ID", key="columns_to_drop", max_chars=500)
else:
    columns_to_drop = None

if st.button("Preprocess Data", key="preprocess_button"):
    preview = preprocess_data(label_column, columns_to_drop)
    if preview is not None:
        st.dataframe(preview)

if st.button("Analyze Data", key="analyze_button"):
    analyze_and_infer()

if os.path.exists(PROBABILITY_FILE):
    with open(PROBABILITY_FILE, "rb") as file:
        st.download_button("Download Probability Results", file, "probability.csv", "text/csv", key="download_probability")

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "rb") as file:
        st.download_button("Download Output", file, "output.txt", "text/plain", key="download_output")

if st.button("Chat with Model", key="chat_button"):
    if not os.path.exists(OUTPUT_FILE):
        st.error("Please analyze data first to chat with the model.")
    else:
        user_input = st.text_input("Your message:", key="user_input")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            messages = st.session_state.chat_history + [{"role": "user", "content": user_input}]
            completion = client.chat.completions.create(model=llm_model, messages=messages, max_tokens=10000)
            response = completion.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.write(response)

if st.button("Save Chat History", key="save_chat_button"):
    save_chat_history(st.session_state.chat_history, CHAT_FILE)
    st.success("Chat history saved to chat_history.txt.")

if os.path.exists(PROBABILITY_FILE) and os.path.exists(OUTPUT_FILE):
    if st.button("Files Downloaded?", key="download_confirm"):
        if st.button("Yes, Delete Files", key="delete_files"):
            os.remove(PROBABILITY_FILE)
            os.remove(OUTPUT_FILE)
            st.success("Files deleted successfully.")
