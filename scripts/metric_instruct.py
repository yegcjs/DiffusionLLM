import json
from ftlangdetect import detect
from collections import defaultdict
import fire
import re
import sacrebleu

def eval_mmlu(path, data_path, result_path):
    with open(f"{data_path}/index2subject.json", "r") as f:
        index2subject = json.load(f)
    correct = defaultdict(int)
    total = defaultdict(int)
    # import ipdb; ipdb.set_trace()
    hyps, refs = {}, {}
    with open(path, "r") as f:
        for line in f:
            try:
                tag, seq = line.split('\t')
            except:
                tag, seq = line.strip(), ""
            if tag[:3]=="SRC" or tag[:4]=="STEP":
                continue
            index = int(tag.split('-')[-1])
            if tag.startswith("HYP"):
                hyps[index] = seq.strip()
            elif tag.startswith("REF"):
                refs[index] = seq.strip()

    for index in hyps:
        subject = index2subject[index]
        total[subject] += 1
        total["full"] += 1
        if hyps[index] == refs[index]:
            correct[subject] += 1
            correct["full"] += 1

    dump_result = {
        subject: {
            "correct": correct[subject],
            "total": total[subject],
            "accuracy": correct[subject] / total[subject]
        }
        for subject in total
    }
    acc = [item["accuracy"] for subject, item in dump_result.items() if subject != "full"]
    acc = sum(acc) / len(acc)
    dump_result["full_macro"] = {
            "accuracy": acc
        }
    with open(result_path, "w") as f:
        json.dump(dump_result, f, indent=4)
    print("mmlu acc", dump_result["full"]["accuracy"])    


def eval_bbh_nlp(path, data_path, result_path):
    with open(f"{data_path}/index2subject.json", "r") as f:
        index2subject = json.load(f)
    correct = defaultdict(int)
    total = defaultdict(int)
    # import ipdb; ipdb.set_trace()
    hyps, refs = {}, {}
    with open(path, "r") as f:
        for line in f:
            try:
                tag, seq = line.split('\t')
            except:
                tag, seq = line.strip(), ""
            if tag[:3]=="SRC" or tag[:4]=="STEP":
                continue
            index = int(tag.split('-')[-1])
            if tag.startswith("HYP"):
                hyps[index] = seq.strip()
            elif tag.startswith("REF"):
                refs[index] = seq.strip()

    for index in hyps:
        subject = index2subject[index]
        total[subject] += 1
        total["full"] += 1
        if hyps[index] == refs[index]:
            correct[subject] += 1
            correct["full"] += 1

    dump_result = {
        subject: {
            "correct": correct[subject],
            "total": total[subject],
            "accuracy": correct[subject] / total[subject]
        }
        for subject in total
    }
    acc = [item["accuracy"] for subject, item in dump_result.items() if subject != "full"]
    acc = sum(acc) / len(acc)
    dump_result["full_macro"] = {
            "accuracy": acc
        }
    with open(result_path, "w") as f:
        json.dump(dump_result, f, indent=4)
    print("bbh nlp acc", dump_result["full_macro"]["accuracy"])    

def eval_gsm(path, data_path, result_path):
    hyps, refs = {}, {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("HYP") or line.startswith("REF"):
                tag, sent = line.split('\t')
                index = int(tag.split('-')[-1])
                try:
                    number = int(re.findall(r'(\d+(,\d+)*)', sent)[-1][0].replace(',', ''))
                except:
                    number = -0x7fffffff
                if line.startswith("HYP"):
                    hyps[index] = number
                else:
                    refs[index] = number 
    correct, total = 0, len(hyps)
    for index in hyps:
        if hyps[index] == refs[index]:
            print(index, "is correct")
            correct += 1
    print(correct / total)
        
def eval_translate(path, data_path, result_path):
    
    tokenize = "13a" if "deen" in data_path else "intl"
    
    hyps, refs = {}, {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("HYP") or line.startswith("REF"):
                tag, sent = line.split('\t')
                index = int(tag.split('-')[-1])
                if line.startswith("HYP"):
                    hyps[index] = sent.lower()
                else:
                    refs[index] = sent.lower()
    
    print(sacrebleu.corpus_bleu([hyps[index] for index in hyps], [[refs[index] for index in hyps]], tokenize=tokenize))

def eval_tydiqa(path, data_path, result_path):
    with open(f"{data_path}/flan2022_index2lang.json", "r") as f:
        index2lang = json.load(f)
    correct = defaultdict(int)
    total = defaultdict(int)
    # import ipdb; ipdb.set_trace()
    hyps, refs = {}, {}
    with open(path, "r") as f:
        for line in f:
            try:
                tag, seq = line.split('\t')
            except:
                tag, seq = line.strip(), ""
            if tag[:3]=="SRC" or tag[:4]=="STEP":
                continue
            index = int(tag.split('-')[-1])
            if tag.startswith("HYP"):
                hyps[index] = seq.strip()
            elif tag.startswith("REF"):
                refs[index] = seq.strip()
    
    lang_correct = 0
    for index in hyps:
        subject = index2lang[index]
        total[subject] += 1
        total["full"] += 1
        if detect(hyps[index])["lang"] == detect(refs[index])["lang"]:
            lang_correct += 1
        
        if hyps[index] == refs[index]:
            correct[subject] += 1
            correct["full"] += 1

    dump_result = {
        subject: {
            "correct": correct[subject],
            "total": total[subject],
            "accuracy": correct[subject] / total[subject]
        }
        for subject in total
    }
    acc = [item["accuracy"] for subject, item in dump_result.items() if subject != "full"]
    acc = sum(acc) / len(acc)
    dump_result["full_macro"] = {
            "accuracy": acc
    }
    with open(result_path, "w") as f:
        json.dump(dump_result, f, indent=4)
    print("tydiqa acc", dump_result["full"]["accuracy"], "lang acc:", lang_correct / len(hyps))    

def main(data, path, data_path, result_path):
    {
        "mmlu": eval_mmlu,
        "mgsm": eval_gsm,
        "translate": eval_translate,
        "tydiqa": eval_tydiqa,
        "bbh-nlp": eval_bbh_nlp
    }[data](path, data_path, result_path)

if __name__ == '__main__':
    # try:
    fire.Fire(main)
    # except:
    #     pass