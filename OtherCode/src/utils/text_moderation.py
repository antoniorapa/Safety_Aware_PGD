from captum.attr import LayerIntegratedGradients, TokenReferenceBase, FeatureAblation
import torch
from openai import OpenAI
import numpy as np
import time
from transformers import pipeline
from abc import ABC, abstractmethod
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, BertForSequenceClassification

class BaseTextModeration(ABC):
    def __init__(self, model_name, mode='ablation',api_key=None,client=None, threshold=0.5, n_eval=100):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.categories = []
        self.mode = mode
        self.threshold=threshold
        self.api_key=api_key
        self.client = client
        self.api_call_count = 0
        self.n_eval=n_eval


    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def get_model_predictions(self, text):
        pass

    @abstractmethod
    def custom_forward_func(self, text_ids):
        pass

    @abstractmethod
    def interpret_text(self,text):
        pass

    def get_token_attributions(self, text_ids, target_label):
        # Create a feature mask that groups tokens belonging to the same word
        tokens = self.tokenizer.convert_ids_to_tokens(text_ids.squeeze().cpu().numpy())

        feature_mask = self.get_feature_mask(tokens)
        ablator = FeatureAblation(self.custom_forward_func)
        attributions = ablator.attribute(text_ids,
                                         baselines=self.tokenizer.pad_token_id,
                                         target=target_label,
                                         feature_mask=feature_mask)

        return attributions.squeeze().cpu().numpy(), tokens, feature_mask.squeeze().cpu().numpy()

    def add_special_tokens(self, batch_tensor, start_token, end_token):

        start_tensor = torch.tensor([start_token], device=batch_tensor.device).expand(batch_tensor.size(0), 1)
        end_tensor = torch.tensor([end_token], device=batch_tensor.device).expand(batch_tensor.size(0), 1)
        modified_batch = torch.cat((start_tensor, batch_tensor, end_tensor), dim=1)

        return modified_batch

    def group_tokens(self, tokens, groups, scores):
        grouped_tokens = []
        grouped_scores = []
        current_group = groups[0]
        current_scores = scores[0]
        current_tokens = []

        for token, group, score in zip(tokens, groups, scores):
            token = self.clean_token(token)
            if group == current_group:
                current_tokens.append(token)
            else:
                grouped_tokens.append("".join(current_tokens))
                grouped_scores.append(current_scores)
                current_group = group
                current_scores = score
                current_tokens = [token]

        # Append the last group
        grouped_tokens.append("".join(current_tokens))
        grouped_scores.append(current_scores)

        return grouped_tokens, grouped_scores

    def clean_token(self, token):
        return token.replace('Ġ', '').replace('##', '')

class DebertaTextModeration(BaseTextModeration):
    def load_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()
        self.categories = ["hate", "hate/threatening", "harassment", "OK", "sexual", "sexual/minors", "self-harm", "violence", "violence/graphic"]

    def get_model_predictions(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        return probs.squeeze()

    def custom_forward_func(self, text_ids):
        input_ids = self.add_special_tokens(text_ids, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        return probs

    def interpret_text(self, text):

        probabilities = self.get_model_predictions(text)
        target_label = torch.argmax(probabilities, dim=-1).item()

        label_prob_pairs = list(zip(range(len(probabilities)), self.categories, probabilities))
        label_prob_pairs.sort(key=lambda item: item[2], reverse=True)
        results = []
        for class_index, class_label, class_score in label_prob_pairs:
            if class_index == target_label:
                text_ids = self.tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').to(self.device)
                attributions, tokens, groups = self.get_token_attributions(text_ids, class_index)
                word_tokens, scores = self.group_tokens(tokens, groups, attributions)

                results.append({
                    "class_index": class_index,
                    "class_label": class_label,
                    "class_score": class_score.item(),
                    "class_attributions": [
                        {"token": token, "attribution": score}
                        for token, score in zip(word_tokens, scores)
                    ]
                })

        return results

    def get_feature_mask(self, tokens):
        word_groups = []
        current_group = 0
        feature_mask = []

        for token in tokens:
            if token.startswith("Ġ") and len(feature_mask) > 0:  # New word starts
                current_group += 1
            feature_mask.append(current_group)

        return torch.tensor([feature_mask], device=self.device)

class BertTextModeration(BaseTextModeration):
    def load_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=18).to(self.device)
        self.model.eval()
        self.categories = ['harassment', 'harassment_threatening', 'hate', 'hate_threatening',
                           'self_harm', 'self_harm_instructions', 'self_harm_intent', 'sexual',
                           'sexual_minors', 'violence', 'violence_graphic', 'self-harm',
                           'sexual/minors', 'hate/threatening', 'violence/graphic',
                           'self-harm/intent', 'self-harm/instructions', 'harassment/threatening']

    def get_model_predictions(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(self.device)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        predictions = torch.sigmoid(outputs.logits).squeeze()
        return predictions

    def custom_forward_func(self, text_ids):
        input_ids = self.add_special_tokens(text_ids, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        return probs

    def interpret_text(self, text):

        probabilities = self.get_model_predictions(text)
        target_label = torch.argmax(probabilities, dim=-1).item()

        label_prob_pairs = list(zip(range(len(probabilities)), self.categories, probabilities))
        label_prob_pairs.sort(key=lambda item: item[2], reverse=True)
        results = []
        for class_index, class_label, class_score in label_prob_pairs:
            if class_score > self.threshold:
                text_ids = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=False,
                    max_length=128,
                    return_token_type_ids=False,
                    truncation=True,
                    return_attention_mask=False,
                    return_tensors='pt'
                ).to(self.device)['input_ids']

                attributions, tokens, groups = self.get_token_attributions(text_ids, class_index)

                word_tokens, scores = self.group_tokens(tokens, groups, attributions)

                results.append({
                    "class_index": class_index,
                    "class_label": class_label,
                    "class_score": class_score.item(),
                    "class_attributions": [
                        {"token": token, "attribution": score}
                        for token, score in zip(word_tokens, scores)
                    ]
                })

        return results

    def get_feature_mask(self,tokens):
        word_groups = []
        current_group = 0
        feature_mask = []

        for token in tokens:
            if token.startswith("##"):
                feature_mask.append(current_group)
            else:
                current_group += 1
                feature_mask.append(current_group)

        return torch.tensor([feature_mask], device=self.device)

class OpenAITextModeration(BaseTextModeration):
    def load_model(self):
        self.model = None
        self.tokenizer = None
        self.categories = ['SFW', 'NSFW']
        if self.client is None and not self.api_key is None:
            self.client = OpenAI(api_key=self.api_key)
        elif self.client is None and  self.api_key is None:
            raise("Provide API key or client")

    def apply_mask(self,words, mask, pad_token="<pad>"):
        mask = mask.squeeze(0)  # Rimuove la dimensione extra se necessario
        masked_words = [word if mask[i] == 1 else pad_token for i, word in enumerate(words)]
        return masked_words

    def custom_forward_func(self, mask, text, true_categories):
        words = text.split(' ')
        masked_words = self.apply_mask(words, mask, pad_token="<pad>")
        precessed_text=' '.join(masked_words)
        rate_limit_per_minute = 1000
        delay = 60.0 / rate_limit_per_minute
        time.sleep(delay)
        response = self.client.moderations.create(input=precessed_text).results[0]
        self.api_call_count += 1
        scores = [response.category_scores.dict()[key] for key in true_categories]
        average_score = sum(scores) / len(scores) if scores else 0
        probs = torch.tensor([1 - average_score, average_score]).unsqueeze(0)
        return probs

    def get_model_predictions(self, text):
        response = self.client.moderations.create(input=text).results[0]
        self.api_call_count += 1
        flagged = response.flagged
        probs = torch.tensor([0, 1]) if flagged else torch.tensor([1, 0])
        return probs

    def get_token_attributions(self, word_ids, target_label, text, true_categories):

        ablator = FeatureAblation(self.custom_forward_func)
        attributions = ablator.attribute(word_ids,
                                         baselines=0,
                                         target=target_label,
                                         additional_forward_args=(text,true_categories))

        return attributions.squeeze().cpu().numpy()

    def interpret_text(self, text):

        words = text.split(' ')
        response = self.client.moderations.create(input=text).results[0]
        class_attribution = []
        target_label = 0
        attributions=None
        probabilities = torch.tensor([1, 0])
        if response.flagged:
            true_categories = [key for key, value in response.categories.dict().items() if value]
            scores = [response.category_scores.dict()[key] for key in true_categories]
            average_score = sum(scores) / len(scores) if scores else 0
            probabilities = torch.tensor([1-average_score, average_score])
            target_label = 1
            words_ids = torch.ones((1, len(words)), dtype=torch.int)
            attributions = self.get_token_attributions(words_ids, target_label, text, true_categories)
            for i, attribution in enumerate(attributions):
                class_attribution.append(
                    {"token": words[i], "attribution": attribution}
                )

        results = []
        results.append({
            "class_index": target_label,
            "class_label": self.categories[target_label],
            "class_score": probabilities[target_label],
            "class_attributions": class_attribution,
            "critical_words": words[np.argmax(attributions)] if target_label == 1 else None
        })
        return results

class DistilRobertaTextModeration(BaseTextModeration):
    def load_model(self):
        #self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)

        #self.model.eval()
        self.model = pipeline("text-classification", model=self.model_name, device=self.device)
        self.categories = ['SFW', 'NSFW']

    def apply_mask(self,words, mask, pad_token="<pad>"):
        mask = mask.squeeze(0)  # Rimuove la dimensione extra se necessario
        masked_words = [word if mask[i] == 1 else pad_token for i, word in enumerate(words)]
        return masked_words

    def custom_forward_func(self, mask, text):
        words = text.split(' ')
        masked_words = self.apply_mask(words, mask, pad_token="<pad>")
        precessed_text=' '.join(masked_words)
        output = self.model(precessed_text)
        label = output[0]['label']
        score = output[0]['score']
        probs = torch.tensor([1-score,score]).unsqueeze(0) if label == self.categories[1] else torch.tensor([score,1-score]).unsqueeze(0)
        return probs

    def get_model_predictions(self, text):
        output = self.model(text)
        label = output[0]['label']
        score = output[0]['score']
        probs = torch.tensor([1 - score, score]) if label == self.categories[1] else torch.tensor([score, 1 - score])
        return probs

    def get_token_attributions(self, word_ids, target_label,text):

        ablator = FeatureAblation(self.custom_forward_func)
        attributions = ablator.attribute(word_ids,
                                         baselines=0,
                                         target=target_label,
                                         additional_forward_args=text)

        return attributions.squeeze().cpu().numpy()


    def interpret_text(self, text):
        words=text.split(' ')
        probabilities = self.get_model_predictions(text)

        target_label = torch.argmax(probabilities, dim=-1).item()

        class_attribution = []
        results = []
        attributions = None
        if target_label == 1:
            words_ids = torch.ones((1, len(words)), dtype=torch.int)
            attributions = self.get_token_attributions(words_ids, target_label, text)
            for i, attribution in enumerate(attributions):
                class_attribution.append(
                    {"token": words[i], "attribution": attribution}
                )
        results.append({
            "class_index": target_label,
            "class_label": self.categories[target_label],
            "class_score": probabilities[target_label],
            "class_attributions": class_attribution,
            "critical_words": words[np.argmax(attributions)] if target_label== 1 else None
        })
        return results


