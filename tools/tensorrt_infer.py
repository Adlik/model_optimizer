"""
TensorRT inference function
"""
import pycuda.driver as cuda  # pylint: disable=import-error
import tensorrt as trt  # pylint: disable=import-error


class HostDeviceMem:
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    """Allocates all buffers required for an engine, i.e. host/device inputs/outputs."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    """
    This function is generalized for single inputs/outputs. Inputs and outputs are expected to be
    lists of HostDeviceMem objects.
    Asynchronously execute inference on a batch. This method only works for execution contexts built
    from networks with no implicit batch dimension.
    """
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return outputs[0].host.reshape(*context.get_binding_shape(1))
