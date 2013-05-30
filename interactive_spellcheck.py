import enchant
import enchant.checker
from enchant.checker.CmdLineChecker import CmdLineChecker
from enchant.tokenize import get_tokenizer, basic_tokenize, URLFilter

def do_check(checker,to_check):
    for text in to_check:
        checker.set_text(text)
        cmdline_checker = CmdLineChecker()
        cmdline_checker.set_checker(checker)
        cmdline_checker.run()
        to_check[to_check.index(text)] = checker.get_text()

class InteractiveSpellchecker(object):
    def __init__(self):
        self.checker = enchant.checker.SpellChecker("en_US")
        self.cmdline_checker = CmdLineChecker()
        self.cmdline_checker.set_checker(self.checker)
        self.result = []
    
    def process_text(self, text):
        """
        accepts: [String] text input
        returns: [List] list of lower-case tokens with URLs filtered out
        """
        try:
            del self.result[:]
            to_check = [] 
            for (word,pos) in basic_tokenize(text): 
                if '@' not in word and 'RT' not in word: to_check.append(word) 
            tknzr = get_tokenizer("en_US",filters=[URLFilter])
            return [word for (word,pos) in tknzr(' '.join(to_check))]
        except UnicodeEncodeError: pass

    def do_check(self,word): 
        self.checker.set_text(word)
        self.cmdline_checker.run()
        correct = self.checker.get_text().lower()
        if '#' not in correct:
            self.result.extend(correct.split())
