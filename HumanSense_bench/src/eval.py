from utils.data_execution import load_data
from model.modelclass import Model
import argparse

def main(args):
    print(args.data_file)
    data = load_data(args.data_file)
    model = Model()
    from benchmark.Benchmark import HSPIBench
    benchmark = HSPIBench(data)

    model_classes = {
    "Qwen2-VL-7B-Instruct": ("model.Qwen2VL", "Qwen2VL"),
    "Qwen-25-VL-7B-Instruct": ("model.Qwen25VL", "Qwen25VL"),
    "Qwen25-Omni-7B": ("model.Qwen25omini", "Qwen25omini"),
    "Qwen25-Omni-3B": ("model.Qwen25omini3B", "Qwen25omini"), 
    "rivideo-omni7B": ("model.rivideo_omni", "rivideo_omni"),
    "rivideo-7B": ("model.rivideo", "rivideo"),
    "InternVL3-8B-Instruct": ("model.InternVL3", "InternVL"),
    "LLaVA-Next-Video-7B": ("model.LLaVANeXTVideo", "LLaVANextVideo7"),
    "Qwen2-Audio-7B-Instruct": ("model.Qwen2audio", "Qwen2audio7B"),
    "LLaVA-OneVision": ("model.LLaVAOneVision", "LLaVAOneVision"),
    "MiniCPM-V-2_6": ("model.MiniCPMV", "MiniCPMV"),
    "LongVA": ("model.LongVA", "LongVA"),
    "VideoLLaMA3": ("model.VideoLLaMA3", "VideoLLaMA3"),
    "IXC25omniLive": ("model.IXComini", "IXC25omniLive"),
    "gpt4o": ("model.gpt4o", "gpt4o"),
    "ola": ("model.ola_omni", "ola_omni"),
    }
      
    model = None
    if args.model_name in model_classes:
        module_name, class_name = model_classes[args.model_name]
        try:
            module = __import__(module_name, fromlist=[class_name])
            ModelClass = getattr(module, class_name)
            model = ModelClass()
        except (ImportError, AttributeError) as e:
            print(f"Error loading model '{args.model_name}': {e}")
            exit(1)
    else:
        print(f"Unsupported model name: {args.model_name}")
        exit(1)


    benchmark.eval(data, model, args)         

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="",  help="Path to the data file")
    parser.add_argument("--model_name", type=str, default="Qwen25-Omni-7B",  help="Name of the model")
    parser.add_argument("--resume", action="store_true", help="Whether to resume from checkpoint")
    parser.add_argument("--tmp_video_dir", type=str, default="data/",  help="Name of the model")
    parser.add_argument("--output_file", type=str, default="",  help="Path to the output file")
    parser.add_argument("--audio", action="store_true", help="Whether to use audio_in_video")
    parser.add_argument("--think", action="store_true", help="Whether to think_in_inference")
    parser.add_argument("--think_omini", action="store_true", help="Whether to think_in_inference in omini")
    parser.add_argument("--asr", action="store_true", help="Whether to use asr in some talking videos")
    parser.add_argument("--audio_eval", action="store_true", help="Whether to use audio_in_video")
    args = parser.parse_args()
    
    if args.resume:
        args.data_file = args.output_file
    main(args)