import os

import regex
from underthesea import word_tokenize, pos_tag, sent_tokenize


class TextProcessor:
    EMOJI_FILE_PATH = 'meta_data/emojicon.txt'
    TEENCODE_FILE_PATH = 'meta_data/teencode.txt'
    TYPO_WORD_FILE_PATH = 'meta_data/wrong-word.txt'
    STOPWORDS_FILE_PATH = 'meta_data/vietnamese-stopwords.txt'
    EN_VI_FILE_PATH = 'meta_data/english-vnmese.txt'

    def __init__(self):
        self.current_path = os.path.dirname(__file__)

    @staticmethod
    def load_replace_component(file_name):
        """Load replace word from file to dict for replace

        :param file_name: path to file include replace component
        :return:
        """
        file = open(file_name, 'r', encoding="utf8")
        replace_lst = file.read().split('\n')
        replace_dict = {}
        for line in replace_lst:
            key, value = line.split('\t')
            replace_dict[key] = str(value)
        file.close()
        return replace_dict

    @staticmethod
    def load_remove_component(file_name):
        """Load remove words from file to list

        :param file_name: path to file include remove words
        :return:
        """
        file = open(file_name, 'r', encoding="utf8")
        remove_lst = file.read().split('\n')
        file.close()
        return remove_lst

    def replace_emoji_to_text(self, sentence, emoji_dict=None):
        """Replace emoji icon by text

        :param sentence: sentence to replace emoji
        :param emoji_dict: dict emoji with string for mapping
        :return: clean sentence without emoji icon
        """
        default_path = os.path.join(self.current_path, self.EMOJI_FILE_PATH)
        emoji_dict_default = self.load_replace_component(default_path)
        if emoji_dict:
            emoji_dict_default.update(emoji_dict)
        clean_sentence = ''.join(emoji_dict_default[word] if word in emoji_dict_default
                                 else word for word in list(sentence))
        clean_sentence = regex.sub(r'\s+', ' ', clean_sentence).strip()
        return clean_sentence

    def replace_teencode_to_text(self, sentence, teencode_dict=None):
        """Replace teencode by text

        :param sentence: sentence to replace emoji
        :param teencode_dict: dict teencode with string for mapping
        :return: clean sentence without teencode
        """
        default_path = os.path.join(self.current_path, self.TEENCODE_FILE_PATH)
        teencode_dict_default = self.load_replace_component(default_path)
        if teencode_dict:
            teencode_dict_default.update(teencode_dict)
        clean_sentence = ' '.join(teencode_dict_default[word] if word in teencode_dict_default
                                  else word for word in sentence.split())
        clean_sentence = regex.sub(r'\s+', ' ', clean_sentence).strip()
        return clean_sentence

    def translate_english_to_vietnam(self, sentence, eng_vie_dict=None):
        """Replace english by vietnamese text

        :param sentence: sentence to replace emoji
        :param eng_vie_dict: dict english to vietnamese with string for mapping
        :return: clean sentence without english
        """
        default_path = os.path.join(self.current_path, self.EN_VI_FILE_PATH)
        eng_vie_dict_default = self.load_replace_component(default_path)
        if eng_vie_dict:
            eng_vie_dict_default.update(eng_vie_dict)
        clean_sentence = ' '.join(eng_vie_dict_default[word] if word in eng_vie_dict_default
                                  else word for word in sentence.split())
        clean_sentence = regex.sub(r'\s+', ' ', clean_sentence).strip()
        return clean_sentence

    @staticmethod
    def remove_punctuation_number(sentence):
        """Remove punctuation and number out of sentence

        :param sentence: input string
        :return:
        """
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        clean_sentence = ' '.join(regex.findall(pattern, sentence))
        clean_sentence = regex.sub(r'\s+', ' ', clean_sentence).strip()
        return clean_sentence

    def remove_typo_tokens(self, sentence, typo_word_lst=None):
        """Remove typo tokens from sentence

        :param sentence: input text
        :param typo_word_lst: list typo word for remove
        :return:
        """
        default_path = os.path.join(self.current_path, self.TYPO_WORD_FILE_PATH)
        typo_word_default = self.load_remove_component(default_path)
        if typo_word_lst:
            typo_word_default += typo_word_lst
        clean_sentence = ' '.join('' if word in typo_word_default else word for word in sentence.split())
        clean_sentence = regex.sub(r'\s+', ' ', clean_sentence).strip()
        return clean_sentence

    def remove_stopword(self, sentence, stopwords=None):
        """Remove stopwords from sentence

        :param sentence:  input text
        :param stopwords: list stop word for remove
        :return:
        """
        default_path = os.path.join(self.current_path, self.STOPWORDS_FILE_PATH)
        stopwords_default = self.load_remove_component(default_path)
        if stopwords:
            stopwords_default += stopwords
        clean_sentence = ' '.join('' if word in stopwords_default else word for word in sentence.split())
        clean_sentence = regex.sub(r'\s+', ' ', clean_sentence).strip()
        return clean_sentence

    @staticmethod
    def prepare_unicode_mapper():
        """Make dictionary mapping from others encoding to unicode

        :return:
        """
        mapper = {}
        char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
            '|')
        charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
            '|')
        for _char1252, _charutf8 in zip(char1252, charutf8):
            mapper[_char1252] = _charutf8
        return mapper

    def covert_unicode(self, sentence):
        """Convert text from others encoding to unicode

        :param sentence: string input
        :return: clean text with unicode encoding
        """
        mapper = self.prepare_unicode_mapper()
        return regex.sub(
            r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
            lambda x: mapper[x.group()], sentence)

    @staticmethod
    def process_special_word(text, special_word='không'):
        """Handle special word by combine with previous word

        :param text: input string
        :param special_word: special word to process
        :return:
        """
        new_text = ''
        text_lst = text.split()
        i = 0
        if special_word in text_lst:
            while i <= len(text_lst) - 1:
                word = text_lst[i]
                if word == special_word:
                    next_idx = i + 1
                    if next_idx <= len(text_lst) - 1:
                        word = word + '_' + text_lst[next_idx]
                    i = next_idx + 1
                else:
                    i = i + 1
                new_text = new_text + word + ' '
        else:
            new_text = text
        return new_text.strip()

    def process_postag_thesea(self, text, lst_word_type=None):
        """Filter word types of sentence

        :param text: input string
        :param lst_word_type: word types to filter, default is ['A', 'AB', 'V', 'VB', 'VY', 'R']
        :return:
        """
        if lst_word_type is None:
            lst_word_type = ['A', 'AB', 'V', 'VB', 'VY', 'R']
        new_document = ''
        for sentence in sent_tokenize(text):
            sentence = sentence.replace('.', '')
            sentence = ' '.join(word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(
                self.process_special_word(word_tokenize(sentence, format="text"))))
            new_document = new_document + sentence + ' '
        new_document = regex.sub(r'\s+', ' ', new_document).strip()
        return new_document

    def process_text(self, document, emoji_dict=None, teen_dict=None, wrong_list=None):
        """Combine cleaning text process to return clean text

        :param document: text input
        :param emoji_dict: additional emoji dict
        :param teen_dict: additional teen dict
        :param wrong_list: additional typo tokens
        :return: clean text
        """
        new_sentence = ''
        for sentence in sent_tokenize(document):
            sentence = self.replace_emoji_to_text(sentence, emoji_dict)
            sentence = self.replace_teencode_to_text(sentence, teen_dict)
            sentence = self.remove_punctuation_number(sentence)
            sentence = self.remove_typo_tokens(sentence, wrong_list)
            new_sentence = new_sentence + sentence + '. '
        document = regex.sub(r'\s+', ' ', document).strip()
        return document
