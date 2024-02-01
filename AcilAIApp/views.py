from django.shortcuts import render
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification,pipeline,AutoModelForMaskedLM
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import ElectraTokenizer, ElectraForSequenceClassification



model2 = BertForSequenceClassification.from_pretrained("NurDrmz/AcilAI_Model", num_labels=10)
tokenizer2 = BertTokenizer.from_pretrained("NurDrmz/AcilAI_Model")
# Ayrıca, kategori etiketlerini de bir değişkene atayalım
labels = ["Barinma", "Yemek", "Giysi", "Su", "Lojistik", "Elektronik", "Yagma", "Kurtarma", "Saglik", "Alakasiz"]
device = torch.device ("cuda:0" if torch.cuda.is_available () else "cpu")


# Hugging Face modeli ve tokenizer'ını yükle
model_name = 'NurDrmz/AcilAI_Electra_Model'
tokenizer = ElectraTokenizer.from_pretrained(model_name)
model = ElectraForSequenceClassification.from_pretrained(model_name)


def process_input_text(text_input, tokenizer):
    inputs = tokenizer(text_input, return_tensors="pt")
    return inputs

def make_prediction(text_input, tokenizer, model):
    inputs = process_input_text(text_input, tokenizer)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class

def predictor(request):
    if request.method == 'POST':
        print("Predictor1 function called.")

        text_input = request.POST.get('text_input', '')
        
        # Giriş metni için işleme fonksiyonunu kullanarak hazırlık yap
        inputs = process_input_text(text_input, tokenizer)

        # Modeli kullanarak tahmin işlemlerini yap
        outputs = model(**inputs)
        predicted_scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].tolist()

        # Modelin kategorilerini tanımlayın
        class_labels = ['Alakasiz', 'Barinma', 'Elektronik', 'Giysi', 'Kurtarma', 'Lojistik', 'Saglik', 'Su', 'Yagma', 'Yemek']

        # Skorları kategorilere atayarak bir sözlük oluşturun
        prediction_dict = dict(zip(class_labels, predicted_scores))

        # Ağırlıkları büyükten küçüğe sıralayın
        sorted_predictions = sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)

        # Sıralı tahmin sonucunu gönder
        context = {'sorted_predictions': sorted_predictions, 'text_input': text_input}
        print(context)
        return render(request, 'main.html', context)

    return render(request, 'main.html')
#///////////////////////////////////////////////////////////

# Metin girişini analiz eden fonksiyonunuzu değiştirmedim
def analyze_text_input2(text_input, model2, tokenizer2, device, labels):
    # Veriyi tokenize et
    tokenized_input = tokenizer2(text_input, truncation=True, padding=True, return_tensors="pt")
    input_ids = tokenized_input["input_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)
    model2.to (device)
    with torch.no_grad():
        outputs = model2(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    if len(probabilities.shape) == 1:  # Tek bir örnek için
        predicted_category = torch.argmax(probabilities).item()
        category_probabilities = probabilities.cpu().numpy().tolist()
    else:  # Mini-batch için
        predicted_category = torch.argmax(probabilities, dim=-1).item()
        category_probabilities = probabilities.cpu().numpy().tolist()[0]
    return predicted_category, category_probabilities

# Çıktıyı biçimlendiren fonksiyonunuzu da değiştirmedim
def format_output(labels, probabilities):
    formatted_output = ""
    for label, prob in zip(labels, probabilities):
        formatted_output += f"{label}:{prob:.3f}\n"
    return formatted_output

# Predictor fonksiyonunuzu ise şöyle güncelledim:
def predictor2(request):
    if request.method == 'POST':
        print("Predictor2 function called.")

        text_input = request.POST.get('text_input', '')
        # Metin girişini analiz eden fonksiyonu çağıralım
        predicted_category, category_probabilities = analyze_text_input2(text_input, model2, tokenizer2, device, labels)
        # Çıktıyı biçimlendiren fonksiyonu çağıralım
        formatted_output = format_output(labels, category_probabilities)
        print(formatted_output)
        # Çıktıyı bir sözlüğe dönüştürelim
        prediction_dict = dict(zip(labels, category_probabilities))
        print(prediction_dict)
        # Sözlüğü skora göre sıralayalım
        sorted_predictions2 = sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)
        context = {'sorted_predictions2': sorted_predictions2}
        print(context)
        # Sonucu main.html sayfasına gönderelim
        return render(request, 'main.html', context={'sorted_predictions2': sorted_predictions2, 'text_input': text_input})

#/////////////////////////////////


