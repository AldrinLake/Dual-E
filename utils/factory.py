def get_model(model_name, args):
    name = model_name.lower()
    if name == "dual_e":
        from models.Dual_E import Dual_E
        return Dual_E(args)
    else:
        assert 0
