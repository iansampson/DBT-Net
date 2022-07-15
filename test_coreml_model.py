import numpy
import torch
import coremltools as ct
from aia_trans import dual_aia_trans_merge_crm, aia_complex_trans_mag, aia_complex_trans_ri

# Load CoreML model
ct_model = ct.models.model.MLModel('dpt-net_aia_merge_dns300.mlpackage')
print(ct_model.input_description)
print(ct_model.output_description)

# Load PyTorch model pre-trained on DNS300 dataset
model_path = "./BEST_MODEL/aia_merge_dns300_conti.pth.tar"
torch_model = dual_aia_trans_merge_crm()
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
torch_model.load_state_dict(checkpoint)
torch_model.eval()

with torch.no_grad():
  input = torch.rand([1, 2, 401, 161])

  torch_output = torch_model(input)
  ct_output = ct_model.predict({'x_1': input.numpy()})

  print(torch_output)
  print(ct_output)
