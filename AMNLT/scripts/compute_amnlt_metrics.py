from collections import Counter
import os
import sys
import jiwer
import re
import json
import functools

from datasets import load_dataset
from gabcparser import GabcParser
from gabcparser.utils import separate_lyrics_music
import gabcparser.grammars as grammars
from AMNLT.utils.dc_base_unfolding_utils.dataset import make_vocabulary, Separation

import argparse

if __name__ == "__main__":
    supported_datasets = ["PRAIG/GregoSynth_staffLevel", "PRAIG/Solesmes_staffLevel", "PRAIG/Einsiedeln_staffLevel", "PRAIG/Salzinnes_staffLevel"]
    parser = argparse.ArgumentParser(prog="Evaluation metrics for AMNLT")
    parser.add_argument("-s", "--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--per_sample_metric", type=str, default=None, help="Name of a file for per sample metrics JSON file")
    parser.add_argument("--n_samples", type=int, default=0, help="Number of samples that will be used (first n samples) - 0=all samples")
    parser.add_argument("dataset", choices=supported_datasets, help="Dataset name on Huggingface")
    parser.add_argument("predictions", type=str, help="Predictions output from model")

def get_bwer(X, Y, verbose=False):
    fxv = Counter(X)
    fyv = Counter(Y)
    V = set(fxv.keys()) | set(fyv.keys())
    diff_count = 0
    for word in V:
        diff_count += abs(fxv[word] - fyv[word])
    if verbose:
        print(f"Diff count: {diff_count}")
        print(f"Length of X: {len(X)}")
        print(f"Numerator: {abs(len(X) - len(Y))}")
    diff_count = (abs(len(X) - len(Y)) + diff_count) / (2 * len(X))
    return diff_count * 100.00

def get_mer(resultado_final, dataset, dataset_name, verbose=False):
    total_mer = 0.0
    mer_values = []
    random_sample = False
    worst_mer = 0.0
    for i,(gt_text, result) in enumerate(zip(dataset["transcription_music"], resultado_final)):
        gt_text = gt_text.replace('\n', ' ')
        gt_text = gt_text.replace('(', ' ').replace(')', ' ')
        gt_text = ' '.join(gt_text.split())
        result = ' '.join(result.split())
        if dataset_name in ["solesmes", "gregosynth", "PRAIG/Solesmes_staffLevel", "PRAIG/GregoSynth_staffLevel"]:
            mer = jiwer.cer(gt_text, result)
        elif dataset_name in ["einsiedeln", "salzinnes", "PRAIG/Einsiedeln_staffLevel", "PRAIG/Salzinnes_staffLevel"]:
            mer = jiwer.wer(gt_text, result)
        else:
            raise NotImplementedError("Unknown dataset name")
        total_mer += mer
        mer_values.append(mer * 100.00)
        
        if mer > worst_mer:
            worst_mer = mer
            worst_gt = gt_text
            worst_result = result
        if random_sample:
            print(gt_text)
            print(result)
            print(mer)
            random_sample = False
    if verbose:
        print("Worst MER:" + str(worst_mer))
        print("\tGT: " + str(worst_gt))
        print("\tPrediction: " + str(worst_result) + "\n")
    mean_mer = (total_mer / len(dataset)) * 100.00
    
    # Calculate standard deviation
    mean_mer2 = sum(mer_values) / len(mer_values)
    variance = sum((x - mean_mer2) ** 2 for x in mer_values) / len(mer_values)
    std_dev = variance ** 0.5
    print(f"Mean MER: {mean_mer2:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Minimum MER: {min(mer_values):.2f}")
    print(f"Maximum MER: {max(mer_values):.2f}")
    print(f"Median MER: {sorted(mer_values)[len(mer_values) // 2]:.2f}")
    print(f"Variance: {variance:.2f}")
    
    return mean_mer

def get_cer_syler(resultado_final, dataset, verbose=False):
    total_cer = 0.0
    total_syler = 0.0
    cer_values = []
    random_sample = False
    worst_cer = 0.0
    for gt_text, result in zip(dataset["transcription_lyric"], resultado_final):
        gt_text = gt_text.replace('\n', ' ')
        gt_text = ' '.join(gt_text.split())
        result = ' '.join(result.split())
        cer = jiwer.cer(gt_text, result)
        syler = jiwer.wer(gt_text, result)
        total_cer += cer
        total_syler += syler
        cer_values.append(cer * 100.00)
        if random_sample:
            print(gt_text)
            print(result)
            print(cer)
            random_sample = False
        if cer > worst_cer:
            worst_cer = cer
            worst_gt = gt_text
            worst_result = result
    if verbose:
        print("Worst CER:" + str(worst_cer))
        print("\tGT: " + str(worst_gt))
        print("\tPrediction: " + str(worst_result) + "\n")
    mean_cer = (total_cer / len(dataset)) * 100.00
    mean_syler = (total_syler / len(dataset)) * 100.00
    return mean_cer, mean_syler

def extract_music_lyrics_solesmes_gregosynth_muaw(predictions_file, dataset, dataset_name):
    with open(predictions_file, 'r', encoding='utf-8') as transcripts_file:
        transcripts_lines = transcripts_file.readlines()
    resultado_final_music = []
    resultado_final_lyrics = []
    for transcript in transcripts_lines:
        with_m_string = ""
        without_m_string = ""
        i = 0
        while i < len(transcript):
            if transcript[i:i+3] == "<m>":
                with_m_string += transcript[i+3]
                without_m_string += ' '
                i += 4
            else:
                if transcript[i] != ")" and transcript[i] != "(":
                    without_m_string += transcript[i]
                with_m_string += ' '
                if transcript[i] != ' ':
                    i += 1
                else:
                    without_m_string += ' '
                    i += 1
        with_m_string = re.sub(r'\s+', ' ', with_m_string)
        without_m_string = re.sub(r'\s+', ' ', without_m_string)
        resultado_final_music.append(with_m_string)
        resultado_final_lyrics.append(without_m_string)
    mer = get_mer(resultado_final_music, dataset, dataset_name)
    cer, syler = get_cer_syler(resultado_final_lyrics, dataset)
    return mer, cer, syler

def extract_music_lyrics_einsiedeln_salzinnes(predictions_file, dataset, dataset_name, sorted_music_vocab):
    with open(predictions_file, 'r', encoding='utf-8') as transcripts_file:
        transcripts_lines = transcripts_file.readlines()
    resultado_final_music = []
    resultado_final_lyrics = []
    for transcript in transcripts_lines:
        found_music_tokens = []
        transcript_copy = transcript
        for token in sorted_music_vocab:
            if token in transcript_copy:
                if token in ["n", "f"]:
                    pattern = r'(?<!\()\b{}\b(?!\))'.format(re.escape(token))
                    matches = re.findall(pattern, transcript_copy)
                    if matches:
                        found_music_tokens.append(token)
                        transcript_copy = re.sub(pattern, " ", transcript_copy)
                else:
                    found_music_tokens.append(token)
                    transcript_copy = transcript_copy.replace(token, " ")
        resultado_final_lyrics.append(" ".join(transcript_copy.split()))
        i = 0
        transcript_music = []
        while i < len(transcript):
            for token in found_music_tokens:
                if transcript[i:i+len(token)] == token:
                    transcript_music.append(token)
                    i += len(token)
                    break
            else:
                i += 1
        resultado_final_music.append(" ".join(transcript_music).replace("(", "").replace(")", ""))
    mer = get_mer(resultado_final_music, dataset, dataset_name)
    cer, syler = get_cer_syler(resultado_final_lyrics, dataset)
    return mer, cer, syler

def tokenization_muaw(X, Y):
    words_X = []
    music = False
    syl = ""
    for char in X:
        if char == "(":
            if syl != "":
                words_X.append(syl)
                syl = ""
            words_X.append(char)
            music = True
        elif char == ")":
            words_X.append(char)
            music = False
        elif music:
            words_X.append(char)
        elif not music and char != " ":
            syl += char
    words_Y = []
    i = 0
    syl = ""
    while i < len(Y):
        if Y[i] == "(":
            if(syl != ""):
                words_Y.append(syl)
                syl = ""
            words_Y.append(Y[i])
        elif Y[i] == ")":
            words_Y.append(Y[i])
        elif Y[i:i+3] == "<m>":
            words_Y.append(Y[i+3])
            i += 3
        elif Y[i] != " ":
            syl += Y[i]
        i += 1
    return words_X, words_Y

def tokenization_pseudo(X, Y, sorted_music_vocab):
    X = X.replace(" ", "")
    Y = Y.replace(" ", "")
    words_X = extract_vocab(sorted_music_vocab, X)
    words_Y = extract_vocab(sorted_music_vocab, Y)
    return words_X, words_Y

def extract_vocab(sorted_music_vocab, transcript):
    lyrics_vocab = []
    found_music_tokens = []
    words = []
    transcript_copy = transcript
    for token in sorted_music_vocab:
        if token in transcript_copy:
            found_music_tokens.append(token)
            transcript_copy = transcript_copy.replace(token, " ")
    lyrics_vocab.append(" ".join(transcript_copy.split()))
    vocab = found_music_tokens + transcript_copy.split()
    sorted_vocab = sorted(vocab, key=len, reverse=True)
    i = 0
    while i < len(transcript):
        for token in sorted_vocab:
            if transcript[i:i+len(token)] == token:
                words.append(token)
                i += len(token)
                break
        else:
            i += 1
    return words

def dataset_separation(example, parser):
    lyric, music = separate_lyrics_music.separate_lyrics_music(example["transcription"], parser, filtered_symbol=" ")
    return {"transcription_lyric": lyric, "transcription_music": music}

def generate_separated_transcriptions(dataset, dataset_name, gabc_variation = None):
    if gabc_variation is None:
        if dataset_name in ["gregosynth", "PRAIG/GregoSynth_staffLevel"]:
            gabc_variation = grammars.GABC
        elif dataset_name in ["solesmes", "PRAIG/Solesmes_staffLevel"]:
            gabc_variation = grammars.S_GABC
        elif dataset_name in ["einsiedeln", "salzinnes", "PRAIG/Einsiedeln_staffLevel", "PRAIG/Salzinnes_staffLevel"]:
            gabc_variation = grammars.MEI_GABC
        else:
            raise ValueError(f"Could not infer gabc variation for the unknown dataset '{dataset_name}'")
    
    parser = GabcParser.load_parser(gabc_variation)

    new_dataset = dataset.map(functools.partial(dataset_separation, parser=parser))

    error_indices = []
    for i,example in enumerate(new_dataset):
        if example["transcription_lyric"] is None or example["transcription_music"] is None:
            error_indices.append(i)
    return error_indices, new_dataset

def metrics(predictions_file, dataset, dataset_name, n_samples, per_sample_metrics_filename=None):
    sorted_music_vocab = None
    error_indices, dataset = generate_separated_transcriptions(dataset, dataset_name)
    if len(error_indices) > 0:
        if len(error_indices) < 100:
            print(f"Could not separate lyrics and musics for the following samples: {error_indices}")
        else:
            print(f"Could not separate lyrics and musics for {len(error_indices)} samples (e.g. {error_indices[:10]}")
        print("THESE RESULTS ARE THEREFORE INVALID UNTIL THE ERRORS ARE CORRECTED")
    if dataset_name in ["einsiedeln", "salzinnes", "PRAIG/Einsiedeln_staffLevel", "PRAIG/Salzinnes_staffLevel"]:
        # Consider adding caching for the vocabulary
        music_vocab, _ = make_vocabulary(dataset_name, "new_gabc", Separation.MUSIC)
        sorted_music_vocab = sorted(music_vocab.keys(), key=len, reverse=True)
    
    if dataset_name in ["solesmes", "gregosynth", "PRAIG/Solesmes_staffLevel", "PRAIG/GregoSynth_staffLevel"]:
        mer, cer, syler = extract_music_lyrics_solesmes_gregosynth_muaw(predictions_file, dataset, dataset_name)
    elif dataset_name in ["einsiedeln", "salzinnes", "PRAIG/Einsiedeln_staffLevel", "PRAIG/Salzinnes_staffLevel"]:
        mer, cer, syler = extract_music_lyrics_einsiedeln_salzinnes(predictions_file, dataset, dataset_name, sorted_music_vocab)
    with open(predictions_file, 'r', encoding='utf-8') as transcript_file:
        predictions_lines = transcript_file.readlines()
    y_true = ["".join(s.replace('\n', ' ')) for s in dataset["transcription"]]
    y_pred = ["".join(s) for s in predictions_lines]
    per_sample = []
    N = len(y_true)
    if n_samples == 0 or n_samples > N:
        n_samples = N
    for i in range(n_samples):
        if dataset_name in ["solesmes", "gregosynth", "PRAIG/Solesmes_staffLevel", "PRAIG/GregoSynth_staffLevel"]:
            x_tokens, y_tokens = tokenization_muaw(dataset["transcription"][i].replace('\n', ' '), predictions_lines[i])
        elif dataset_name in ["einsiedeln", "salzinnes", "PRAIG/Einsiedeln_staffLevel", "PRAIG/Salzinnes_staffLevel"]:
            x_tokens, y_tokens = tokenization_pseudo(dataset["transcription"][i].replace('\n', ' '), predictions_lines[i], sorted_music_vocab)
        sample_bwer = get_bwer(x_tokens, y_tokens)
        sample_amler = jiwer.wer(" ".join(x_tokens), " ".join(y_tokens)) * 100.00
        sample_aler = (sample_amler - sample_bwer) / sample_amler if sample_amler > 0 else 0
        per_sample.append({
            "index": i,
            "gt": dataset["transcription"][i],
            "pred": predictions_lines[i],
            "bwer": sample_bwer,
            "amler": sample_amler,
            "aler": sample_aler
        })
    if per_sample_metrics_filename is not None:
        with open(per_sample_metrics_filename, "w", encoding="utf-8") as f:
            json.dump(per_sample, f, indent=2, ensure_ascii=False)
        print(f"Saved per-sample metrics in {per_sample_metrics_filename}")
    total_bwer = 0
    total_amler = 0
    for x, y in zip(y_true, y_pred):
        if dataset_name in ["solesmes", "gregosynth", "PRAIG/Solesmes_staffLevel", "PRAIG/GregoSynth_staffLevel"]:
            x_tokens, y_tokens = tokenization_muaw(x, y)
        elif dataset_name in ["einsiedeln", "salzinnes", "PRAIG/Einsiedeln_staffLevel", "PRAIG/Salzinnes_staffLevel"]:
            x_tokens, y_tokens = tokenization_pseudo(x, y, sorted_music_vocab)
        total_bwer += get_bwer(x_tokens, y_tokens)
        total_amler += jiwer.wer(" ".join(x_tokens), " ".join(y_tokens)) * 100.00
    bwer = total_bwer / len(y_true)
    amler = total_amler / len(y_true)
    aler = (amler - bwer) / amler if amler > 0 else 0
    print(f"MER: {round(mer, 2)}")
    print(f"CER: {round(cer, 2)}")
    print(f"SylER: {round(syler, 2)}")
    print(f"ALER: {round(aler, 2)}")
    print(f"\tAMLER: {round(amler, 2)}")
    print(f"\tBWER: {round(bwer, 2)}")

def main(args):
    ds = load_dataset(args.dataset, split=args.split)
    metrics(args.predictions, ds, args.dataset, args.n_samples, args.per_sample_metric)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
