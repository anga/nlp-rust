
/*
Markov:
Fixed size:
  size N
*/

mod dictionary;
// use dictionary;
use std::collections::HashMap;
#[allow(unused_imports)]
use std::str;
#[allow(unused_imports)]
use std::option::{Option};
use std::string::String;
extern crate regex;
use regex::Regex;

/// Markov chain implementation for string.
struct Markov {
    perplexity: f64,
    unigram: HashMap<String, u64>,
    bigram: HashMap<(String, String),u64>,
    trigram: HashMap<(String,String,String), u64>,
    num_training_words: u64,
    training_sentences: Vec<String>,
}

impl Markov {
    fn new() -> Markov {
        Markov {
            perplexity: 0f64,
            unigram: HashMap::new(),
            bigram: HashMap::new(),
            trigram: HashMap::new(),
            num_training_words: 0u64,
            training_sentences: Vec::new()
        }
    }
    // Crea los n-gramas requeridos
    fn train<'a, S: Into<&'a str>>(&mut self, text: S ) {

        // clone the str to be sure
        let cloned_text = text.into().clone();

        // Iterate over each sentence
        for sentence_capture in Regex::new(r"[^\.\\\n\?!¿¡,]+").unwrap().captures_iter(cloned_text) {
            // uvw word
            let mut uword = "*";
            let mut vword = "*";
            let mut wword = "*";
            let sentence = match sentence_capture.at(0) {
                Some(string) => string,
                None => "",
            };
            self.add_unigram("*");
            self.add_bigram("*", "*");
            // Iterate over each word
            for word in Regex::new(r"([a-zA-Z0-9]+)").unwrap().captures_iter(&sentence) {
                self.num_training_words += 1;
                // is the first word
                if uword == "*" {
                    uword = match word.at(0) {
                        Some(stri) => stri,
                        None => "$END$",
                    };
                    self.add_unigram(uword);
                    self.add_bigram("*", uword);
                    self.add_trigram("*", "*", uword);
                } else if vword == "*" { // it's the second word
                    vword = match word.at(0) {
                        Some(stri) => stri,
                        None => "$END$",
                    };
                    self.add_unigram(vword);
                    self.add_bigram(uword, vword);
                    self.add_trigram("*", uword, vword);
                } else {
                    wword = match word.at(0) {
                        Some(stri) => stri,
                        None => "$END$",
                    };
                    self.add_unigram(wword);
                    self.add_bigram(vword, wword);
                    self.add_trigram(uword, vword, wword);
                    uword = vword.clone();
                    vword = wword.clone();
                }
            }
            wword = "$END$";
            self.add_unigram(wword);
            self.add_bigram(vword, wword);
            self.add_trigram(uword, vword, wword);
        }
        self.training_sentences.push(cloned_text.to_string());
        ;
    }

    /// Store the dictionary into a file
    fn export_disk(&self, file_path: String) {
        ; //Save to disk
    }
    /// Calculate the training model quality. Bigger is better
    /// Return value goes from 0.0 to 1.0
    //  NOTE: This is a really basic calculaiton. It's only a averrage of all provabilities of
    //  our corpus
    fn model_quality(&self) -> f64 {
        let mut p = 0f64;
        for sentence in self.training_sentences.iter() {
            p += self.estimate(sentence.as_str());
        }
        p/self.training_sentences.len() as f64
        // p = 2f64.powf((-p/self.num_training_words as f64)*p);

        // 2.pow(-(1/self.num_training_words))
        // p
    }

    // FIXME: Move to another structure
    fn linear_interpolation(&self, text: &str) -> f64 {
        let mut sum = 0f64;
        let mut sum_u = 0f64;
        let mut sum_b = 0f64;
        let mut sum_t = 0f64;
        // Iterate over each sentence.
        // FIXME: the dot can not be a sentence delimitor like Copmpany Co. or any other.
        for sentence_capture in Regex::new(r"[^\.\\\n\?!¿¡,]+").unwrap().captures_iter(text) {
            let mut uword = "*";
            let mut vword = "*";
            let mut wword = "*";
            let mut words = 0u64;
            let sentence = match sentence_capture.at(0) {
                Some(string) => string,
                None => "",
            };
            // Iterate over each word in the sentence
            for word in Regex::new(r"([a-zA-Z0-9]+)").unwrap().captures_iter(&sentence) {
                // get the current word
                wword = match word.at(0) {
                    Some(stri) => stri,
                    None => "$END$",
                };
                // The next code does this:
                // If we have the next text: "The doc runs"
                // u, v, w = *
                // first we reads "The" and assign it to "w"
                // u, v = *
                // w = The
                // calculate the sum...
                // and move the values. Then at the end of the for we have:
                // u = *
                // v, w = The
                // Next we reads "dog" and assign it to w
                // u = *
                // v = The
                // w = doc
                // and so on...
                let (ur,br,tr) = self.count_trigrams(uword,vword,wword); // Maximum likelihood estimate
                println!("{:?}", (ur,br,tr));
                sum_u += ur;
                sum_b += br;
                sum_t += tr;
                uword = vword.clone();
                vword = wword.clone();
            }
        }
        0.3333f64*sum_t + 0.3333f64*sum_b + 0.3333f64*sum_u // this is <= 1 all the time
    }

    /// Ad an unigram into the dictionary
    fn add_unigram(&mut self, word: &str) {
        let mut value = self.unigram.entry(word.to_string()).or_insert(0);
        *value += 1;
    }

    /// Ad an bigram into the dictionary
    fn add_bigram(&mut self, word1: &str, word2: &str) {
        let mut value = self.bigram.entry((word1.to_string(),word2.to_string())).or_insert(0);
        *value += 1;

    }

    /// Ad an trigram into the dictionary
    fn add_trigram(&mut self, word1: &str, word2: &str, word3: &str) {
        let mut value = self.trigram.entry((word1.to_string(),word2.to_string(),word3.to_string())).or_insert(0);
        *value += 1;
    }

    /// Get the value of the unigram or 0 if does not exit
    fn get_unigram(&self, uword: &str) -> u64 {
        match self.unigram.get(uword) {
            Some(value) => *value,
            None => 0u64,
        }
    }

    /// Get the value of the bigram or 0 if does not exit
    fn get_bigram(&self, uword: &str, vword: &str) -> u64 {
        match self.bigram.get(&(uword.to_string(), vword.to_string())) {
            Some(value) => *value,
            None => 0u64,
        }
    }

    /// Get the value of the trigram or 0 if does not exit
    fn get_trigram(&self, uword: &str, vword: &str, wword: &str) -> u64 {
        match self.trigram.get(&(uword.to_string(), vword.to_string(), wword.to_string())) {
            Some(value) => *value,
            None => 0u64,
        }
    }

    /// Retuns the result of each count or 0 if not exist for
    /// p(u,v,w) = count(u,v,w) / count(u,v)
    /// p(v,w) = count(v,w) / count(v)
    /// p(w) = count(w) / total_words_in_training_data
    fn count_trigrams(&self, uword: &str, vword: &str, wword: &str) -> (f64,f64,f64) {
        let uvw = self.get_trigram(uword,vword,wword);
        let uv = self.get_bigram(uword,vword);
        let vw = self.get_bigram(vword,wword);
        let w = self.get_unigram(wword);
        let v = self.get_unigram(vword);

        let mut tr = 0f64;
        let mut br = 0f64;
        let mut ur = 0f64;
        if uv > 0 {
            tr = uvw as f64/uv as f64;
        }
        if v > 0 {
            br = vw as f64/v as f64;
        }
        if self.num_training_words > 0 {
            ur = w as f64 / self.num_training_words as f64;
        }
        (ur,br,tr)
    }

    /// Simple smooth MLE for the trigram
    /// Calculate the maximum likelihood estimate of the trigram.
    /// interpolation = 0.33*p(u,v,w) + 0.33*p(v,w) + 0.33*p(w)
    fn ss_mle(&self, uword: &str, vword: &str, wword: &str) -> (f64,f64,f64) {
        let (ur,br,tr) = self.count_trigrams(uword,vword,wword);
        ((1f64/3f64)*ur, (1f64/3f64)*br, (1f64/3f64)*tr)
    }

    /// estimation for a giving string
    fn estimate(&self,sentence: &str) -> f64 {
        let mut uword = "*";
        let mut vword = "*";
        let mut wword = "*";
        let mut sum_u = 0f64;
        let mut sum_b = 0f64;
        let mut sum_t = 0f64;
        let mut words = 0u64;
        let mut missing_words = 0;
        for sentence_capture in Regex::new(r"[^\.\\\n\?!¿¡,]+").unwrap().captures_iter(&sentence) {
            let sentence_ = match sentence_capture.at(0) {
                Some(string) => string,
                None => "",
            };
            // Iterate over each word in the sentence
            for word in Regex::new(r"([a-zA-Z0-9]+)").unwrap().captures_iter(&sentence_) {

                    words += 1;
                // get the current word
                wword = match word.at(0) {
                    Some(stri) => stri,
                    None => "$END$",
                };
                let mut alpha1 = 0f64;
                let mut alpha2 = 0f64;
                let mut alpha3 = 0f64;
                let ct = self.get_trigram(uword, vword, wword);
                if ct > 0 {
                    alpha1 = 0.75; // 75% importance to the trigram
                }
                alpha2 = (1.0f64 - alpha1)*0.75f64; // 75% of the rest
                alpha3 = 1.0f64 - alpha1 - alpha2 ;
                // calculate alphas

                // The next code does this:
                // If we have the next text: "The doc runs"
                // u, v, w = *
                // first we reads "The" and assign it to "w"
                // u, v = *
                // w = The
                // calculate the sum...
                // and move the values. Then at the end of the for we have:
                // u = *
                // v, w = The
                // Next we reads "dog" and assign it to w
                // u = *
                // v = The
                // w = doc
                // and so on...
                let (ur,br,tr) = self.count_trigrams(uword,vword,wword); // Maximum likelihood estimate
                if self.get_unigram(wword) == 0 {
                    missing_words += 1;
                }
                sum_u += alpha3*ur;
                sum_b += alpha2*br;
                sum_t += alpha1*tr;
                uword = vword.clone();
                vword = wword.clone();
            }
        }
        // Sigmoid(Sumatory(Qml(trigram) + Qml(bigrama) + Qml(unigram)))
        // And because Sigmoid(0) => 0.5 then we rest 1/num_words_in_given_sentence*2
        // If there is no words knowed in the given sentence, then 1/num_words_in_given_sentence*2
        // will be 0.5 and 0.5 - 0.5 = 0
        ( 1f64 /
            (1f64 + ( (-(sum_u + sum_b + sum_t)).exp() ))
        ) - (missing_words as f64 / (2f64*words as f64))
    }


}

#[test]
fn add_unigram_works() {
    let mut markov = Markov::new();
    markov.add_unigram("add");
    markov.add_unigram("add");
    assert!(markov.unigram.len() == 1);
    assert!(markov.get_unigram("add") == 2u64);
}

#[test]
fn add_bigram_works() {
    let mut markov = Markov::new();
    markov.add_bigram("the", "cat");
    markov.add_bigram("the", "cat");
    assert!(markov.bigram.len() == 1);
    assert!(markov.get_bigram("the", "cat") == 2u64);
}

#[test]
fn add_trigram_works() {
    let mut markov = Markov::new();
    markov.add_trigram("the", "cat", "runs");
    markov.add_trigram("the", "cat", "runs");
    assert!(markov.trigram.len() == 1);
    assert!(markov.get_trigram("the","cat","runs") == 2u64);
}

#[test]
fn test_train() {
    let mut markov = Markov::new();
    markov.train("the cat runs\nthe cat eats\nthe dog runs");
    assert!(markov.unigram.len() == 7);
    assert!(markov.bigram.len() == 9);
    assert!(markov.trigram.len() == 9);
    // Test all unigrams

    assert!(markov.get_unigram("the") == 3u64);
    assert!(markov.get_unigram("cat") == 2u64);
    assert!(markov.get_unigram("runs") == 2u64);
    assert!(markov.get_unigram("dog") == 1u64);
    assert!(markov.get_unigram("eats") == 1u64);

    // test all bigrams
    assert!(markov.get_bigram("*", "the") == 3u64);
    assert!(markov.get_bigram("the", "cat") == 2u64);
    assert!(markov.get_bigram("cat", "runs") == 1u64);
    assert!(markov.get_bigram("cat", "eats") == 1u64);
    assert!(markov.get_bigram("the", "dog") == 1u64);
    assert!(markov.get_bigram("dog", "runs") == 1u64)
    ;

    // test all trigrams
    assert!(markov.get_trigram("*", "the", "cat") == 2u64);
    assert!(markov.get_trigram("the", "cat", "runs") == 1u64);
    assert!(markov.get_trigram("the", "cat", "eats") == 1u64);
    assert!(markov.get_trigram("*", "the", "dog") == 1u64);
    assert!(markov.get_trigram("the", "cat", "runs") == 1u64);
}

#[test]
fn test_estimate() {
    let mut markov = Markov::new();
    markov.train("The cat runs. The cat jump. The cat fly. The dog fly. The dog jump");
    markov.train("The cat eats a mouse.");
    markov.train("The dog eats food.");
    markov.train("The dog eats runs.");
    markov.train("The mouse eats chesee.");
    // No we try a new sentence that is not in the training data
    assert!(markov.estimate("The cat eats pasta") >= 0.70f64);
    assert!(markov.estimate("I like pastas") >= 0.0f64);
}

#[test]
fn test_model_quality() {
    let mut markov = Markov::new();
    markov.train("The cat runs. The cat jump. The cat fly. The dog fly. The dog jump");
    markov.train("The cat eats a mouse.");
    markov.train("The dog eats food.");
    markov.train("The dog eats runs.");
    markov.train("The mouse eats chesee.");
    assert!(markov.model_quality() >= 0.9f64);
}
