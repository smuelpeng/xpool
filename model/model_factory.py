from config.base_config import Config
from model.clip_baseline import CLIPBaseline
from model.clip_transformer import CLIPTransformer
from model.clip_abr_transformer import CLIPABRTransformer

class ModelFactory:
    @staticmethod
    def get_model(config: Config):
        if config.arch == 'clip_baseline':
            return CLIPBaseline(config)
        elif config.arch == 'clip_transformer':
            return CLIPTransformer(config)
        elif config.arch == 'clip_abr_transformer':
            return CLIPABRTransformer(config)
        else:
            raise NotImplemented
