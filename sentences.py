from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import pipeline
from spellchecker import SpellChecker

def fix_text(text):
    spell = SpellChecker()
    corrected_word = spell.correction(text)

    tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
    model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")

    input_text = 'Fix grammatical errors in this sentence: ' + corrected_word

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=256)
    edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(edited_text)

if __name__ == '__main__':
    # test
    fix_text('hi Hack MIT! Tis is Gesture, an ASL trnzlation ap. We aree sO exsited to b here!')
