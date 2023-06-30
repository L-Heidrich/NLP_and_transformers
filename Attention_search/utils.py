import string
import nltk
from nltk.corpus import stopwords
import torch
from transformers import BertTokenizer, AutoModel

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def remove_special_characters(text):
    # Remove special characters using string.punctuation
    cleaned_text = ''.join(
        char for char in text if char not in string.punctuation)

    return cleaned_text


def remove_stopwords(text):
    # Get the set of stopwords
    stopword_set = set(stopwords.words('english'))

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    filtered_tokens = [
        word for word in tokens if word.lower() not in stopword_set]

    # Join the filtered tokens back into a string
    cleaned_text = ' '.join(filtered_tokens)

    return cleaned_text


def filter_nouns(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Perform POS tagging
    tagged_tokens = nltk.pos_tag(tokens)

    # Filter for nouns
    nouns = [word for word, pos in tagged_tokens if pos.startswith('N')]

    # Join the filtered nouns back into a string
    filtered_text = ' '.join(nouns)

    return filtered_text


def format_attention(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def num_layers(attention):
    return len(attention)


def retrieve_attention_scores(text, seed_words):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained(
        'bert-base-uncased', output_attentions=True)

    # Input sentence or text sequence
    #text = remove_special_characters(remove_stopwords(filter_nouns(text)))
    text = remove_special_characters(remove_stopwords(text))

    # Tokenize the text and convert to token IDs

    results = []

    sentence_tokens = tokenizer.convert_ids_to_tokens(
        tokenizer(text, return_tensors="pt").input_ids[0])
    sentence_tokens = sentence_tokens[1:len(sentence_tokens)]

    data = {
        "text": text,
        "tokens": sentence_tokens,
        "results": results
    }

    for seed_word in seed_words:
        inputs = tokenizer(seed_word, text, return_tensors="pt")
        attention = model(**inputs).attentions

        sentence_b_start = (inputs.token_type_ids == 0).sum(dim=1)
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        attention = model(**inputs).attentions

        include_layers = list(range(num_layers(attention)))
        attention = format_attention(attention, include_layers)

        slice_a = slice(0, sentence_b_start)
        slice_b = slice(sentence_b_start, len(tokens))

        scores = attention[:, :, slice_a, slice_b]
        results.append(
            {"seed_word": seed_word,
             "scores": scores,
             "tokens": tokens,
             }
        )

    return data


def sum_up_attention_over_heads_and_layers(scores):
    totals = [0 for _ in range(len(scores[0][0][0])-1)]
    for i in range(len(scores[0])):
        for j in range(len(scores[i][0])):

            current_values = scores[i][j][1].tolist()
            # print(current_values)
            #max_attention = max(current_values)
            #totals[current_values.index(max_attention)] += max_attention
            totals = [a + b for a, b in zip(totals, current_values[:-1])]

    return totals


def prettify_results(text, scores, tokens):
    print("Token attention totals:")
    min_val = min(scores)
    max_val = max(scores)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in scores]

    max_norm_val = max(normalized_data)
    print(tokens[normalized_data.index(max_norm_val)], max_norm_val)
    for token, total in zip(tokens, normalized_data):
        if total > 0.5:
            print(token, total)
