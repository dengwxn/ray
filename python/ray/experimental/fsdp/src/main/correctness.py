from argparse import ArgumentParser
import torch

from ..core.linear.model import LinearModel


def compare_models(
    layer_size: int,
    num_layers: int,
    num_units: int,
    ray_model_file: str,
    torch_model_file: str,
) -> None:
    # Load Ray model
    ray_model = LinearModel(layer_size, num_layers, num_units, torch.device("cpu"))
    ray_model_state_dict = torch.load(ray_model_file, map_location="cpu")
    ray_model.load_state_dict(ray_model_state_dict)
    # Load Torch model
    torch_model = LinearModel(layer_size, num_layers, num_units, torch.device("cpu"))
    torch_model_state_dict = torch.load(torch_model_file, map_location="cpu")
    torch_model.load_state_dict(torch_model_state_dict)
    # Compare models
    ray_weights = ray_model.fetch_weights()
    torch_weights = torch_model.fetch_weights()
    max_diff = 0
    max_diff_layer = -1
    for layer, (ray_weight, torch_weight) in enumerate(zip(ray_weights, torch_weights)):
        if not torch.allclose(ray_weight, torch_weight):
            diff = torch.max(torch.abs(ray_weight - torch_weight)).item()
            print(f"Layer {layer} diff: {diff}")
            if diff > max_diff:
                max_diff = diff
                max_diff_layer = layer
    print(f"Max diff {max_diff} at layer {max_diff_layer}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--layer-size", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-units", type=int, required=True)
    parser.add_argument("--ray-model", type=str, required=True)
    parser.add_argument("--torch-model", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    compare_models(
        args.layer_size,
        args.num_layers,
        args.num_units,
        args.ray_model,
        args.torch_model,
    )


if __name__ == "__main__":
    main()
