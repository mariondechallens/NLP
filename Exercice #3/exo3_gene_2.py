# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 09:49:00 2019

@author: Admin
"""

# generating utterances 
# https://lizadaly.com/brobot/    
        
# Sentences we'll respond with if the user greeted us

import logging        
from textblob import TextBlob

logging.basicConfig()

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)        
        
        
GREETING_KEYWORDS = ("hello", "hi", "greetings", "sup", "what's up",)

GREETING_RESPONSES = ["'sup bro", "hey", "*nods*", "hey you get my snap?"]

FILTER_WORDS = set([

    "skank",

    "wetback",

    "bitch",

    "cunt",

    "dick",

    "douchebag",

    "dyke",

    "fag",

    "nigger",

    "tranny",

    "trannies",

    "paki",

    "pussy",

    "retard",

    "slut",

    "titt",

    "tits",

    "wop",

    "whore",

    "chink",

    "fatass",

    "shemale",

    "nigga",

    "daygo",

    "dego",

    "dago",

    "gook",

    "kike",

    "kraut",

    "spic",

    "twat",

    "lesbo",

    "homo",

    "fatso",

    "lardass",

    "jap",

    "biatch",

    "tard",

    "gimp",

    "gyp",

    "chinaman",

    "chinamen",

    "golliwog",

    "crip",

    "raghead",

    "negro",

    "hooker"])

def check_for_greeting(sentence):
    """If any of the words in the user's input was a greeting, return a greeting response"""
    for word in sentence.words:
        if word.lower() in GREETING_KEYWORDS:
            return random.choice(GREETING_RESPONSES)        
        




def find_verb(sent):

    """Pick a candidate verb for the sentence."""

    verb = None

    pos = None

    for word, part_of_speech in sent.pos_tags:

        if part_of_speech.startswith('VB'):  # This is a verb

            verb = word

            pos = part_of_speech

            break

    return verb, pos


def find_noun(sent):

    """Given a sentence, find the best candidate noun."""

    noun = None



    if not noun:

        for w, p in sent.pos_tags:

            if p == 'NN':  # This is a noun

                noun = w

                break

    if noun:

        logger.info("Found noun: %s", noun)



    return noun



def find_adjective(sent):

    """Given a sentence, find the best candidate adjective."""

    adj = None

    for w, p in sent.pos_tags:

        if p == 'JJ':  # This is an adjective

            adj = w

            break

    return adj

def find_pronoun(sent):
    """Given a sentence, find a preferred pronoun to respond with. Returns None if no candidate
    pronoun is found in the input"""
    pronoun = None

    for word, part_of_speech in sent.pos_tags:
        # Disambiguate pronouns
        if part_of_speech == 'PRP' and word.lower() == 'you':
            pronoun = 'I'
        elif part_of_speech == 'PRP' and word == 'I':
            # If the user mentioned themselves, then they will definitely be the pronoun
            pronoun = 'You'
    return pronoun

def find_candidate_parts_of_speech(parsed):

    """Given a parsed input, find the best pronoun, direct noun, adjective, and verb to match their input.

    Returns a tuple of pronoun, noun, adjective, verb any of which may be None if there was no good match"""

    pronoun = None

    noun = None

    adjective = None

    verb = None

    for sent in parsed.sentences:

        pronoun = find_pronoun(sent)

        noun = find_noun(sent)

        adjective = find_adjective(sent)

        verb = find_verb(sent)

    logger.info("Pronoun=%s, noun=%s, adjective=%s, verb=%s", pronoun, noun, adjective, verb)

    return pronoun, noun, adjective, verb





def check_for_comment_about_bot(pronoun, noun, adjective):
    """Check if the user's input was about the bot itself, in which case try to fashion a response
    that feels right based on their input. Returns the new best sentence, or None."""
    resp = None
    if pronoun == 'I' and (noun or adjective):
        if noun:
            if random.choice((True, False)):
                resp = random.choice(SELF_VERBS_WITH_NOUN_CAPS_PLURAL).format(**{'noun': noun.pluralize().capitalize()})
            else:
                resp = random.choice(SELF_VERBS_WITH_NOUN_LOWER).format(**{'noun': noun})
        else:
            resp = random.choice(SELF_VERBS_WITH_ADJECTIVE).format(**{'adjective': adjective})
    return resp

# Template for responses that include a direct noun which is indefinite/uncountable
SELF_VERBS_WITH_NOUN_CAPS_PLURAL = [
    "My last startup totally crushed the {noun} vertical",
    "Were you aware I was a serial entrepreneur in the {noun} sector?",
    "My startup is Uber for {noun}",
    "I really consider myself an expert on {noun}",
]

SELF_VERBS_WITH_NOUN_LOWER = [
    "Yeah but I know a lot about {noun}",
    "My bros always ask me about {noun}",
]

SELF_VERBS_WITH_ADJECTIVE = [
    "I'm personally building the {adjective} Economy",
    "I consider myself to be a {adjective}preneur",
]
    
NONE_RESPONSES = [

    "uh whatever",

    "meet me at the foosball table, bro?",

    "code hard bro",

    "want to bro down and crush code?",

    "I'd like to add you to my professional network on LinkedIn",

    "Have you closed your seed round, dog?",

]

# end



# start:example-self.py

# If the user tries to tell us something about ourselves, use one of these responses

COMMENTS_ABOUT_SELF = [

    "You're just jealous",

    "I worked really hard on that",

    "My Klout score is {}".format(random.randint(100, 500)),

]

class UnacceptableUtteranceException(Exception):

    """Raise this (uncaught) exception if the response was going to trigger our blacklist"""

    pass



def starts_with_vowel(word):

    """Check for pronoun compability -- 'a' vs. 'an'"""

    return True if word[0] in 'aeiou' else False

def broback(sentence):

    """Main program loop: select a response for the input sentence and return it"""

    logger.info("Broback: respond to %s", sentence)

    resp = respond(sentence)

    return resp

def preprocess_text(sentence):

    """Handle some weird edge cases in parsing, like 'i' needing to be capitalized

    to be correctly identified as a pronoun"""

    cleaned = []

    words = sentence.split(' ')

    for w in words:

        if w == 'i':

            w = 'I'

        if w == "i'm":

            w = "I'm"

        cleaned.append(w)



    return ' '.join(cleaned)

 
 
def respond(sentence):
    """Parse the user's inbound sentence and find candidate terms that make up a best-fit response"""
    cleaned = preprocess_text(sentence)
    parsed = TextBlob(cleaned)

    # Loop through all the sentences, if more than one. This will help extract the most relevant
    # response text even across multiple sentences (for example if there was no obvious direct noun
    # in one sentence
    pronoun, noun, adjective, verb = find_candidate_parts_of_speech(parsed)

    # If we said something about the bot and used some kind of direct noun, construct the
    # sentence around that, discarding the other candidates
    resp = check_for_comment_about_bot(pronoun, noun, adjective)

    # If we just greeted the bot, we'll use a return greeting
    if not resp:
        resp = check_for_greeting(parsed)

    if not resp:
        # If we didn't override the final sentence, try to construct a new one:
        if not pronoun:
            resp = random.choice(NONE_RESPONSES)
        elif pronoun == 'I' and not verb:
            resp = random.choice(COMMENTS_ABOUT_SELF)
        else:
            resp = construct_response(pronoun, noun, verb)

    # If we got through all that with nothing, use a random response
    if not resp:
        resp = random.choice(NONE_RESPONSES)

    logger.info("Returning phrase '%s'", resp)
    # Check that we're not going to say anything obviously offensive
    filter_response(resp)

    return resp
        
def construct_response(pronoun, noun, verb):
    """No special cases matched, so we're going to try to construct a full sentence that uses as much
    of the user's input as possible"""
    resp = []

    if pronoun:
        resp.append(pronoun)

    # We always respond in the present tense, and the pronoun will always either be a passthrough
    # from the user, or 'you' or 'I', in which case we might need to change the tense for some
    # irregular verbs.
    if verb:
        verb_word = verb[0]
        if verb_word in ('be', 'am', 'is', "'m"):  # This would be an excellent place to use lemmas!
            if pronoun.lower() == 'you':
                # The bot will always tell the person they aren't whatever they said they were
                resp.append("aren't really")
            else:
                resp.append(verb_word)
    if noun:
        pronoun = "an" if starts_with_vowel(noun) else "a"
        resp.append(pronoun + " " + noun)

    resp.append(random.choice(("tho", "bro", "lol", "bruh", "smh", "")))

    return " ".join(resp)

def filter_response(resp):
    """Don't allow any words to match our filter list"""
    tokenized = resp.split(' ')
    for word in tokenized:
        if '@' in word or '#' in word or '!' in word:
            raise UnacceptableUtteranceException()
        for s in FILTER_WORDS:
            if word.lower().startswith(s):
                raise UnacceptableUtteranceException()
       
        
###### tests
import random
import pytest

random.seed(0)



def test_random_utterance():

    """An utterance which is unparsable should return one of the random responses"""

    sent = "abcd"  # Something unparseable
    broback(sent)

    #assert resp == NONE_RESPONSES[-2]  #"I'd like to add you to my professional network on LinkedIn"

test_random_utterance()

def test_basic_greetings():

    """The bot should respond sensibly to common greeting words"""

    sent = "hello"
    broback(sent)

    #assert resp == GREETING_RESPONSES[1] #'hey'

test_basic_greetings()




def test_contains_reference_to_user():

    """An utterance where the user mentions themselves generally should specifically return a phrase starting with 'You'"""

    sent = "I went to dinner"
    broback(sent)

    #assert resp.startswith("You ")

test_contains_reference_to_user()




def test_negs_user():

    """An utterance where the user says 'I am' <something> should specifically tell them they aren't that thing"""

    sent = "I am good at Python programming"
    broback(sent)

    #assert resp.startswith("You aren't really")

    sent = "I'm good at Python programming"
    broback(sent)

    #assert resp.startswith("You aren't really")

    sent = "i'm good at Python programming"
    broback(sent)

    #assert resp.startswith("You aren't really")
test_negs_user()

def test_contains_reference_to_bot():

    """An utterance where the user directs something at the bot itself should return a canned response"""

    sent = "You are lame"
    broback(sent)

    #assert 'lame' in resp
test_contains_reference_to_bot()

def test_reuses_subject():

    """If the user tells us about some kind of subject, we should mention it in our response"""

    sent = "I am a capable programmer"
    broback(sent)

    #assert "programmer" in resp

test_reuses_subject()



def test_strip_offensive_words():

    """Don't allow the bot to respond with anything obviously offensive"""

    # To avoid including an offensive word in the test set, add a harmless word temporarily

    FILTER_WORDS.add('snakeperson')

    sent = "I am a snakeperson"

    with pytest.raises(UnacceptableUtteranceException):

        broback(sent)

test_strip_offensive_words()



def test_strip_punctuation():

    """Removing most punctuation is one way to ensure that the bot doesn't include hashtags or @-signs, which are potential vectors for harrassment"""

    sent = "I am a #snakeperson"

    with pytest.raises(UnacceptableUtteranceException):

        broback(sent)



    sent = "@you are funny"

    with pytest.raises(UnacceptableUtteranceException):

        broback(sent)

test_strip_punctuation()

def test_unicode():

    """Bros love internationalization"""

    broback(u"☃")  # Unicode snowman              
    
test_unicode()    



