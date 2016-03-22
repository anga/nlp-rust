
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
use std::cmp::Ordering;
use std::string::String;
extern crate regex;
use regex::Regex;


struct Markov<'a> {
    perplexity: f64,
    unigram: HashMap<&'a str, u64>,
    bigram: HashMap<(&'a str, &'a str),u64>,
    trigram: HashMap<(&'a str,&'a str,&'a str), u64>,
    num_training_words: u64,
    training_sentences: Vec<String>,
}

impl<'a> Markov<'a> {
    fn new() -> Markov<'a> {
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
    fn train(&mut self, text: &'a str ) {

        // Iterate over each sentence
        for sentence_capture in Regex::new(r"[^\.\\\n\?!¿¡,]+").unwrap().captures_iter(text) {
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
        self.training_sentences.push(text.to_string());
        ;
    }
    fn export_disk(&self, file_path: String) {
        ; //Save to disk
    }
    // Mientras más bajo, mejor.
    // 2^( -(1/M)*sum[  log_2(p(x)) ] )
    // M = Numero total de palabras (con repetición)
    // p(x) = provabilidad de una sentencia dentro del conjunto de ejemplo/entrenamiento
    fn perplexity(&self) -> f64 {
        let mut p = 1f64;
        for sentence in self.training_sentences.iter() {
            p += (self.lineal_interpolation(sentence.as_str())).log2();
        }
        p = 2f64.powf(-p/self.num_training_words as f64);

        // 2.pow(-(1/self.num_training_words))
        p
    }
    fn lineal_interpolation(&self, text: &'a str) -> f64 {
        let mut sum = 0f64;
        let mut sum_u = 0f64;
        let mut sum_b = 0f64;
        let mut sum_t = 0f64;
        for sentence_capture in Regex::new(r"[^\.\\\n\?!¿¡,]+").unwrap().captures_iter(text) {
            let mut uword = "*";
            let mut vword = "*";
            let mut wword = "*";
            let sentence = match sentence_capture.at(0) {
                Some(string) => string,
                None => "",
            };
            for word in Regex::new(r"([a-zA-Z0-9]+)").unwrap().captures_iter(&sentence) {
                // is the first word
                if uword == "*" {
                    uword = match word.at(0) {
                        Some(stri) => stri,
                        None => "$END$",
                    };
                    let (ur,br,tr) = self.calculate_trigram("*", "*", uword);
                    let cp = match self.trigram.get(&("*","*",uword)) {
                        Some(value) => *value as f64,
                        None => 0f64,
                    };
                    //sum += (-cp*r.log2());
                    sum_u += ur;
                    sum_b += br;
                    sum_t += tr;
                    // Nothing until trigram
                } else if vword == "*" { // it's the second word
                    vword = match word.at(0) {
                        Some(stri) => stri,
                        None => "$END$",
                    };
                    let (ur,br,tr) = self.calculate_trigram("*", uword, vword);
                    let cp = match self.trigram.get(&("*",uword,vword)) {
                        Some(value) => *value as f64,
                        None => 0f64,
                    };
                    //sum += (-cp*r.log2());
                    sum_u += ur;
                    sum_b += br;
                    sum_t += tr;
                    // Nothing until trigram
                } else {
                    wword = match word.at(0) {
                        Some(stri) => stri,
                        None => "$END$",
                    };
                    let (ur,br,tr) = self.calculate_trigram(uword, vword, wword);
                    let cp = match self.trigram.get(&(uword,vword,wword)) {
                        Some(value) => *value as f64,
                        None => 0f64,
                    };
                    //sum += (-cp*r.log2());
                    sum_u += ur;
                    sum_b += br;
                    sum_t += tr;
                    uword = vword.clone();
                    vword = wword.clone();
                }
            }
        }
        0.3333f64*sum_t + 0.3333f64*sum_b + 0.3333f64*sum_u
    }

    fn add_unigram(&mut self, word: &'a str) {
        let mut value = self.unigram.entry(word).or_insert(0);
        *value += 1;
    }

    fn add_bigram(&mut self, word1: &'a str, word2: &'a str) {
        let mut value = self.bigram.entry((word1,word2)).or_insert(0);
        *value += 1;

    }

    fn add_trigram(&mut self, word1: &'a str, word2: &'a str, word3: &'a str) {
        let mut value = self.trigram.entry((word1,word2,word3)).or_insert(0);
        *value += 1;
    }

    /// Calculate the provability of the trigram using lineal interpolation.
    /// p(u,v,w) = count(u,v,w) / count(u,v)
    /// p(v,w) = count(v,w) / count(v)
    /// p(w) = count(w) / total_words_in_training_data
    /// interpolation = p(u,v,w) + p(v,w) + p(w)
    fn calculate_trigram(&self, uword: &'a str, vword: &'a str, wword: &'a str) -> (f64,f64,f64) {
        let uvw = match self.trigram.get(&(uword,vword,wword)) {
            Some(value) => *value,
            None => 0u64,
        };
        let uv = match self.bigram.get(&(uword,vword)) {
            Some(value) => *value,
            None => 0u64,
        };
        let vw = match self.bigram.get(&(vword,wword)) {
            Some(value) => *value,
            None => 0u64,
        };
        let w = match self.unigram.get(wword) {
            Some(value) => *value,
            None => 0u64,
        };
        let v = match self.unigram.get(vword) {
            Some(value) => *value,
            None => 0u64,
        };
        let mut tr = 0f64;
        let mut br = 0f64;
        let mut ur = 0f64;
        if uv > 0 {
            tr = (uvw as f64/uv as f64);
        }
        if v > 0 {
            br = (vw as f64/v as f64);
        }
        if self.num_training_words > 0 {
            ur = (w as f64 / self.num_training_words as f64);
        }
        println!("PPPPPPP {:?}/{:?} + {:?}/{:?} + {:?}/{:?}",uvw,uv,vw,w,w,self.num_training_words);
        println!("{:?}", (uword,vword,wword));
        (ur,br,tr)
    }

}

#[test]
fn add_unigram_works() {
    let mut markov = Markov::new();
    markov.add_unigram("add");
    markov.add_unigram("add");
    assert!(markov.unigram.len() == 1);
    assert!(markov.unigram[&("add")] == 2u64);
}

#[test]
fn add_bigram_works() {
    let mut markov = Markov::new();
    markov.add_bigram("the", "cat");
    markov.add_bigram("the", "cat");
    assert!(markov.bigram.len() == 1);
    assert!(markov.bigram[&("the","cat")] == 2u64);
}

#[test]
fn add_trigram_works() {
    let mut markov = Markov::new();
    markov.add_trigram("the", "cat", "runs");
    markov.add_trigram("the", "cat", "runs");
    assert!(markov.trigram.len() == 1);
    assert!(markov.trigram[&("the","cat","runs")] == 2u64);
}

#[test]
fn test_train() {
    let mut markov = Markov::new();
    markov.train("the cat runs\nthe cat eats\nthe dog runs");

    assert!(markov.unigram.len() == 6);
    assert!(markov.bigram.len() == 8);
    assert!(markov.trigram.len() == 5);
    // Test all unigrams

    assert!(markov.unigram.get("the") == Some(&3u64));
    assert!(markov.unigram.get("cat") == Some(&2u64));
    assert!(markov.unigram.get("runs") == Some(&2u64));
    assert!(markov.unigram.get("dog") == Some(&1u64));
    assert!(markov.unigram.get("eats") == Some(&1u64));

    // test all bigrams
    assert!(markov.bigram.get(&("*", "the")) == Some(&3u64));
    assert!(markov.bigram.get(&("the", "cat")) == Some(&2u64));
    assert!(markov.bigram.get(&("cat", "runs")) == Some(&1u64));
    assert!(markov.bigram.get(&("cat", "eats")) == Some(&1u64));
    assert!(markov.bigram.get(&("the", "dog")) == Some(&1u64));
    assert!(markov.bigram.get(&("dog", "runs")) == Some(&1u64));

    // test all trigrams
    assert!(markov.trigram.get(&("*", "the", "cat")) == Some(&2u64));
    assert!(markov.trigram.get(&("the", "cat", "runs")) == Some(&1u64));
    assert!(markov.trigram.get(&("the", "cat", "eats")) == Some(&1u64));
    assert!(markov.trigram.get(&("*", "the", "dog")) == Some(&1u64));
    assert!(markov.trigram.get(&("the", "cat", "runs")) == Some(&1u64));
}

#[test]
fn test_perplexity() {

    let mut markov = Markov::new();
    markov.train("the cat runs\nthe cat eats\nthe dog runs");
    println!("YYYYYYY {:?}\n{:?}\n{:?}", markov.unigram, markov.bigram, markov.trigram);
    println!("LLL {:?}", markov.perplexity());
    assert!(false);
}
