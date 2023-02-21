import torch
import torch_tensorrt

from network_swinir import SwinIR as net


def load_model():
    model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
    param_key_g = 'params_ema'
    # Download: https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth
    pretrained_model = torch.load('003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
    model.load_state_dict(pretrained_model[param_key_g]
                          if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model


def compile_tensorrt_model(dtype):
    model = load_model().eval().cuda()

    inputs = [torch_tensorrt.Input([1, 3, 128, 128], dtype=dtype)]
    enabled_precisions = {dtype}

    with torch.no_grad():
        # traced_model = torch.jit.trace(model.forward, torch.rand(1, 3, 128, 128).cuda())
        traced_model = torch.jit.freeze(torch.jit.script(model))

        with open('code.txt', 'w', encoding='utf-8') as f:
            f.write(traced_model.code)

        with open('graph.txt', 'w', encoding='utf-8') as f:
            f.write(str(traced_model.graph))

        compiled_model = torch_tensorrt.compile(traced_model, inputs=inputs, enabled_precisions=enabled_precisions,
                                                truncate_long_and_double=True)
        torch.jit.save(compiled_model, "swinir_trt_ts_module_float32.ts")


compile_tensorrt_model(torch.float)
