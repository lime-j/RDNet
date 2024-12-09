import importlib

from models.arch import *

from models.cls_model_eval_nocls_reg import ClsModel as ClsReg


def make_model(name: str):

    model = ClsReg()
    return model
