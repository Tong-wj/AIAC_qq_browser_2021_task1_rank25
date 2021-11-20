from lichee import plugin
from .bert_common import BertTextFieldParserCommon


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "bert_text")
class BertTextFieldParser(BertTextFieldParserCommon):
    """field parser for BERT input

    """
    def __init__(self):
        super().__init__()
        self.parser_name = "bert_text"


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "lsh_bert_text")
class LshBertTextFieldParser(BertTextFieldParserCommon):
    """field parser for BERT input

    """
    def __init__(self):
        super().__init__()
        self.parser_name = "lsh_bert_text"


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "rsh_bert_text")
class RshBertTextFieldParser(BertTextFieldParserCommon):
    """field parser for BERT input

    """
    def __init__(self):
        super().__init__()
        self.parser_name = "rsh_bert_text"
