use std::collections::{HashMap, HashSet};
use std::f64;
use regex::Regex;
use rustc_serialize::json;

// Default smoothing for each MarkovClassification
static DEFAULT_SMOOTHING: f64 = 1.0f64;

/// Define one classification, like "positive" or "negative" on a sentimental classification
#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
struct MarkovClassification {
    // The class that MarkovClassifier will return
    label: String,
    // Total number of trained samples
    num_examples: u64,
    // TODO: Doc
    num_words: u64,
    // The provability of the classifier
    probability: f64,
    // The base provability of each word (this will be the minimum provability of each word)
    default_word_probability: f64,
    // The counter of all ngrams (unigrams, bigrams and trigrams)
    // The index value is the ngram and the second is a tuple where
    // 0: number of times the word apears in the training sample
    // 1: TODO Document this
    ngrams: HashMap<Vec<String>, (u64, f64)>,
}

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct MarkovClassifier {
    // All classifiers.
    // key: label name
    // value: The classifier
    classifications: HashMap<String,MarkovClassification>,
    // The total number of trained samples for all classifiers
    num_examples: u64,
    // Used to smooth the result (check linear regression on wikipedia)
    smoothing: f64,
    // The whole vocabulary of ngrams
    vocab: HashSet<Vec<String>>,
    /// Max number of ngram tuple, if it's 3, the ngrams generated will be
    // ("I"), ("I", "like"), ("I", "like", "pizza")
    max_ngrams: u64,
}

impl MarkovClassification {
    /// Create and return a new MarkovClassification.
    /// Usage:
    /// let classification = MarkovClassification::new("positive");
    pub fn new(label: &String) -> MarkovClassification {
        MarkovClassification {
            label: label.clone(),
            num_examples: 0u64,
            num_words: 0u64,
            probability: 0.0f64,
            default_word_probability: 0.0f64,
            ngrams: HashMap::new(),
        }
    }

    /// Add 1 ngram to the vocabulary
    pub fn add_ngram(&mut self, ngram: &Vec<String>) {
        self.num_words += ngram.len() as u64;
        // Check if the ngram has the given ngram
        if self.ngrams.contains_key(ngram) {
            // If exist in the current ngram, then add 1 to the number of appearances
            self.ngrams.get_mut(ngram).unwrap().0 += 1;
        } else {
            // If not exist, create it with 1 appearance and 0 of TODO: what is second value?
            self.ngrams.insert(ngram.clone(), (1, 0.0f64));
        }
    }

    /// Train the classifier
    /// vocab: A bigger vocab to our vocab. Used to compare
    pub fn train(&mut self, vocab: &HashSet<Vec<String>>, total_examples: u64, smoothing: f64) {
        // the probability of this classification
        self.probability = self.num_examples as f64 / total_examples as f64;

        // the probability of any word that has not been seen in a document
        // labeled with this classification's label
        self.default_word_probability = smoothing /
            (self.num_words as f64 + smoothing * vocab.len() as f64);
        for ngram in vocab.iter() {
            if self.ngrams.contains_key(ngram) {
                let mut prev_ngram = ngram.clone();
                // Discard last word in the ngram
                prev_ngram.pop();
                // get the total number of the previous ngram
                let count_prev_ngram = match self.ngrams.get(&*prev_ngram) {
                    Some(&(value, _)) => value,
                    None => 0,
                };

                let mut word_entry = self.ngrams.get_mut(ngram).unwrap();
                let word_count = word_entry.0;

                let mut p_word_given_label = 0f64;
                // If the previous ngram has value, the provability of appearance of the current ngram is
                // count(ngram) + smooth / count(ngram-1) + smooth * vocab.len()
                if count_prev_ngram > 0 {
                    p_word_given_label = (word_count as f64 + smoothing) /
                    (count_prev_ngram as f64 + smoothing * vocab.len() as f64);
                } else {
                    p_word_given_label =
                        (word_count as f64 + smoothing) /
                        (self.num_words as f64 + smoothing * vocab.len() as f64);
                }

                word_entry.1 = p_word_given_label;
            }
        }
    }

    /// Calculate the provability/score of the giving document based on the trained data
    /// document: A vector of ngrams
    /// vocab: A set of ngrams (the vocab of a bigger training set that includs the cureent classifier)
    pub fn score_document(&self, document: &Vec<Vec<String>>, vocab: &HashSet<Vec<String>>) -> f64 {
        let mut total = 0.0f64;
        for ngram in document.iter() {
            if vocab.contains(ngram) {
                let word_probability = match self.ngrams.get(ngram) {
                    Some( &(_, p) ) => p,
                    None => self.default_word_probability,
                };
                total += word_probability.ln();
            }
        }
        // self.probability.ln() + total
        total
    }
}

/// # MarkovClassifier
/// Uage:
/// let classifier = MarkovClassifier::new(3)
/// let dataset = [
///    ("I like fruits", "like"),
///    ("I hate animals", "hate"),
///    ("I like sports", "like")
///    // ...
/// ]
/// for document in dataset {
///     classifier.add_document(document);
/// }
/// classifier.train();
/// classifier.classify("I hate chocolate!"); // "hate"
impl MarkovClassifier {
    pub fn new(ngrams_number: u64) -> MarkovClassifier {
        MarkovClassifier {
            vocab: HashSet::new(),
            num_examples: 0u64,
            smoothing: DEFAULT_SMOOTHING,
            classifications: HashMap::new(),
            max_ngrams: ngrams_number,
        }
    }

    /// Takes a document that has been tokenized into a vector of strings
    /// and a label and adds the document to the list of documents that the
    /// classifier is aware of and will train on next time the `train()` method is called
    pub fn add_document_tokenized(&mut self, document: &Vec<Vec<String>>, label: &String) {
        // Do nothing if there is no ngram
        if document.len() == 0 { return; }

        // make sure the classification already exists
        if !self.classifications.contains_key(label) {
            let c = MarkovClassification::new(label);
            self.classifications.insert(label.clone(), c);
        }

        // Get the classification for the current sample
        let mut classification = self.classifications.get_mut(label).unwrap();

        // Add all ngrams to the classification ngrams count and vocabulary
        for word in document.iter() {
            classification.add_ngram(word);
            self.vocab.insert(word.clone());
        }

        // Add 1 to the number of samples trained
        self.num_examples += 1;
        classification.num_examples += 1;
    }

    // splits a String on whitespaces
    pub fn tokenize(document: &String, ngrams: &u64) -> Vec<Vec<String>> {
        let re = Regex::new(r"[\s,\.;!?]+").unwrap();
        // This will be the return
        let mut vocab: Vec<Vec<String>> = Vec::new();
        // store the current ngram
        let mut current_ngram:Vec<String> = vec!["*".to_owned(); *ngrams as usize];
        // Store the current
        let mut ngram_idx = 1;
        for word in re.split(document) {
            // Left shift and push the new word to the unigram
            current_ngram.remove(0);
            current_ngram.push(word.to_owned().clone());
            // Add all unigrams now
            // (same as ngrams.times do in ruby)
            for x in (0..(*ngrams as usize)) {
                // we need to add ngrams times an ngram, if 3, we need to add 3 ngrams
                // the first one with 1 word, the second with 2 words and the last with 3 words
                let mut c_ngram: Vec<String> = Vec::new();
                // generate the current ngram to tokenize
                for idx in (0..x + 1 as usize) {
                    c_ngram.push(current_ngram.get(idx).unwrap().clone());
                }
                // once we generate the ngram, we push it to the tokenized result
                vocab.push(c_ngram);
            }
        }
        vocab
    }

    /// Takes a document and a label and tokenizes the document by
    /// breaking on whitespace characters. The document is added to the list
    /// of documents that the classifier is aware of and will train on next time
    /// the `train()` method is called
    pub fn add_document(&mut self, document: &String, label: &String) {
        let max = self.max_ngrams;
        self.add_document_tokenized(&MarkovClassifier::tokenize(document, &max), label);
    }

    /// Takes an unlabeled document and tokenizes it by breaking on spaces and
    /// then computes a classifying label for the document
    pub fn classify(&self, document: &String) -> String {
        let max = self.max_ngrams;
        self.classify_tokenized(&MarkovClassifier::tokenize(document, &max))
    }

    /// Takes an unlabeled document that has been tokenized into a vector of strings
    /// and then computes a classifying label for the document
    pub fn classify_tokenized(&self, document: &Vec<Vec<String>>) -> String {
        let mut max_score = f64::NEG_INFINITY;
        let mut max_classification = None;

        for classification in self.classifications.values() {

            let score = classification.score_document(document, &self.vocab);
            if score > max_score {
                max_classification = Some(classification);
                max_score = score;
            }
        }

        max_classification.expect("no classification found").label.clone()
    }

    /// Similar to classify but instead of returning a single label, returns all
    /// labels and the probabilities of each one given the document
    pub fn get_document_probabilities_tokenized(&self, document: &Vec<Vec<String>>) -> Vec<(String, f64)> {

        let all_probs:Vec<(String, f64)> = self.classifications.values().map(|classification| {
            let score = classification.score_document(document, &self.vocab);
            (classification.label.clone(), score)
        }).collect();

        let total_prob = all_probs.iter()
            .map(|&(_, s)| s)
            .fold(0.0, |acc, s| acc + s);

        all_probs.into_iter().map(|(c, s)| (c, 1.0 - s/total_prob) ).collect()
    }

    /// Similar to classify but instead of returning a single label, returns all
    /// labels and the probabilities of each one given the document
    pub fn get_document_probabilities(&self, document: &String) -> Vec<(String, f64)> {
        let max = self.max_ngrams;
        self.get_document_probabilities_tokenized(&MarkovClassifier::tokenize(document, &max))
    }

    /// Trains the classifier on the documents that have been observed so far
    pub fn train(&mut self) {
        for (_, classification) in self.classifications.iter_mut() {
            classification.train(&self.vocab, self.num_examples, self.smoothing);
        }
    }

    /// Encodes the classifier as a JSON string.
    pub fn to_json(&self) -> String {
        json::encode(self).ok().expect("encoding JSON failed")
    }

    /// Builds a new classifier from a JSON string
    pub fn from_json(encoded: &str) -> MarkovClassifier {
        let classifier: MarkovClassifier = json::decode(encoded).ok().expect("decoding JSON failed");
        classifier
    }
}
