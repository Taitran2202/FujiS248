from enum import Enum


class ONNXProvider(Enum):
    CUDA=0
    TensorRT=1
    CPU=2
class ModelState(Enum):
    Unloaded=0
    Loaded=1
    Error=2
    NotFound=3
class ONNXRuntime:
    State = ModelState.Unloaded
    Provider = ONNXProvider.CUDA
    def CreateProviderOption(self,directory):
        providers = None
        if self.Provider==ONNXProvider.CUDA:
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                })
            ]
                
        elif self.Provider==ONNXProvider.TensorRT:
            providers = [
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    "trt_fp16_enable":True,
                    "trt_engine_cache_enable":True,
                    'trt_max_workspace_size': 2147483648,
                    'trt_fp16_enable': True,
                    "trt_engine_cache_path":directory
                })]
        elif self.Provider==ONNXProvider.CPU:
            providers = ['GPUExecutionProvider','CPUExecutionProvider']

        return providers
    def LoadRecipe():
        return False