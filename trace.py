# import yaml
import os
import time
import numpy as np
import torch
import coremltools as ct
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb
from aia_trans import dual_aia_trans_merge_crm, aia_complex_trans_mag, aia_complex_trans_ri

@register_torch_op
def atan2(context, node):
    inputs = _get_inputs(context, node)
    y = inputs[0]
    x = inputs[1]

    # Add a small value to all zeros in order to avoid division by zero
    epsilon = 1.0e-12
    x = mb.select(cond=mb.equal(x=x, y=0.0), a=mb.add(x=x, y=epsilon), b=x)
    y = mb.select(cond=mb.equal(x=y, y=0.0), a=mb.add(x=y, y=epsilon), b=y)

    angle = mb.select(cond=mb.greater(x=x, y=0.0),
                      a=mb.atan(x=mb.real_div(x=y, y=x)),
                      b=mb.fill(shape=x.shape, value=0.0))

    angle = mb.select(cond=mb.logical_and(x=mb.less(x=x, y=0.0), y=mb.greater_equal(x=y, y=0.0)),
                      a=mb.add(x=mb.atan(x=mb.real_div(x=y, y=x)), y=np.pi),
                      b=angle)

    angle = mb.select(cond=mb.logical_and(x=mb.less(x=x, y=0.0), y=mb.less(x=y, y=0.0)),
                      a=mb.sub(x=mb.atan(x=mb.real_div(x=y, y=x)), y=np.pi),
                      b=angle)

    angle = mb.select(cond=mb.logical_and(x=mb.equal(x=x, y=0.0), y=mb.greater(x=y, y=0.0)),
                      a=mb.mul(x=mb.mul(x=0.5, y=np.pi), y=mb.fill(shape=x.shape, value=1.0)),
                      b=angle)

    angle = mb.select(cond=mb.logical_and(x=mb.equal(x=x, y=0.0), y=mb.less(x=y, y=0.0)),
                      a=mb.mul(x=mb.mul(x=-0.5, y=np.pi), y=mb.fill(shape=x.shape, value=1.0)),
                      b=angle)

    angle = mb.select(cond=mb.logical_and(x=mb.equal(x=x, y=0.0), y=mb.equal(x=y, y=0.0)),
                      a=mb.fill(shape=x.shape, value=0.0),
                      b=angle)

    context.add(angle, torch_name=node.name)

# Load model pre-trained on DNS300 dataset
model_path = "./BEST_MODEL/aia_merge_dns300_conti.pth.tar"
model = dual_aia_trans_merge_crm()
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

with torch.no_grad():
    # Create random input
    input = torch.rand([1, 2, 401, 161])

    # Create the output directory if needed
    os.makedirs("coreml", exist_ok=True)

    # Trace model
    traced_model = torch.jit.trace(model, input)
    torch.jit.save(traced_model, "coreml/dbt-net_aia_merge_dns300.pt")

    # Convert model to CoreML
    ml_model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=input.shape)])
    ml_model.save("coreml/dbt-net_aia_merge_dns300.mlpackage")

    # Measure prediction time for original model
    t0 = time.process_time()
    output = model(input)
    t1 = time.process_time()
    print("Prediction time for original model: ", t1 - t0)

    # Measure prediction time for traced model
    t0 = time.process_time()
    output = traced_model(input)
    t1 = time.process_time()
    print("Prediction time for traced model: ", t1 - t0)

    # Measure prediction time for CoreML model
    t0 = time.process_time()
    ml_output = ml_model.predict({"x_1": input.numpy()})
    t1 = time.process_time()
    print("Prediction time for CoreML model: ", t1 - t0)
