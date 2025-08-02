import torch

print("="*50)
print("RTX 5060 Ti PyTorch Test")
print("="*50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Supported architectures: {torch.cuda.get_arch_list()}")

# Check sm_120
if 'sm_120' in torch.cuda.get_arch_list():
    print("\nsm_120 is supported!")
    
# Test GPU computation
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print("GPU computation successful!")
    print("Your RTX 5060 Ti is fully working with PyTorch!")
except Exception as e:
    print(f"Error: {e}")

print("="*50)
