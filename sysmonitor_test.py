import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CUDA], profile_memory=True,record_shapes=True) as prof:
    with record_function("model_inference"):
        print("start")
        model(inputs)
        print("end")

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))