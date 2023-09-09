# https://github.com/facebookresearch/llama-recipes/blob/03faba661f079ee1ecaeb66deaa6bdec920a7bab/ft_datasets/alpaca_dataset.py

'''
Goal is to convert the chat jsons into the Samsum dataset format
https://huggingface.co/datasets/samsum
'''
import json
import os
import pandas as pd
import sys

def format_context(messages):
    context = ''
    for message in messages:
        user = list(message.keys())[0]
        context = context + str(user) + ': ' + str(message[user])
        context = context + '\n'
    return context

def format_output(message):
    user = list(message.keys())[0]
    return message[user]

if __name__=="__main__":
    if len(sys.argv) != 4:
        print("Usage: prepare_dataset.py <dataset_folder> <your_name> <save_file>")
        sys.exit(1)


    dataset_folder = sys.argv[1]#'AutomaticWhatsappReplying/json_dataset'
    user_name = sys.argv[2]#'Advaith'
    save_file = sys.argv[3]#'FinalDataset.csv'

    conv = []
    for file in os.listdir(dataset_folder):
        with open(dataset_folder + '/' + file, 'r') as f:
            data = json.load(f)
            count=0
            for message in data:
                if list(message.keys())[0]==user_name and count!=0: #expect the other person to start the conversation
                    if(count>=5):
                        conv.append([format_context(data[count-5:count]),format_output(message)])
                    else:
                        conv.append([format_context(data[0:count]),format_output(message)])
                count+=1

    df = pd.DataFrame(conv)
    df.columns = ['Context','Reply']
    df.to_csv(save_file)
