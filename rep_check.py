import torch

a = torch.load('/root/workspace/NPMC_yolov7/compressed_traced_yolov7_training.pt').cpu().eval()
b = torch.load('/root/workspace/NPMC_yolov7/compressed_traced_yolov7_training_rep_0418_test.pt').cpu().eval()
# a = torch.load('/root/workspace/yolov7_training_models/yolov7_exported_l2norm_10.pt').cpu().eval()
# b = torch.load('/root/workspace/yolov7_training_models/yolov7_exported_l2norm_10_rep_test_0418.pt').cpu().eval()

x = torch.randn(1,3,640,640)
for a_, b_ in zip(a(x),b(x)):
    print(torch.allclose(a_,b_))
    print(torch.sum((a_-b_)**2))