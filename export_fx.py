
import torch

import yaml
import argparse

from models.yolo import ModelForNPMC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7_training.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='cfg/training/yolov7.yaml', help='model.yaml path')
    parser.add_argument('--save-path', type=str, default='traced_yolov7_training.pt', help='save path for the traced model')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    
    weights = torch.load(args.weights)['model'].state_dict()
    model = ModelForNPMC(args.cfg, nc=cfg['nc'])
    model.load_state_dict(weights)

    graph = torch.fx.Tracer().trace(model)
    traced_model = torch.fx.GraphModule(model, graph)

    x = torch.randn(1,3,640,640)

    for original_output, traced_output in zip(model(x), traced_model(x)):
        assert torch.allclose(original_output,traced_output), "inference result is not equal!"

    torch.save(traced_model, args.save_path)