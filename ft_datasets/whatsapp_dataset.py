import datasets
from .utils import Concatenator

def get_preprocessed_whatsapp(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset('csv', data_files='final.csv')[split]

    prompt = (
        f"Reply to the following messages as the user Advaith. Provide just one reply, do not continue the conversation.\n{{context}}\n---\Advaith:\n{{reply}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                context=sample["Context"],
                reply=sample["Reply"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
        
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset
