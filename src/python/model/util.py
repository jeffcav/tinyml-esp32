import torch

def train(model, device, train_loader, optimizer, loss_fn=torch.nn.functional.cross_entropy):
    model.train()

    epoch_loss = 0
    n_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # prepare
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # compute
        output = model(data)
        loss = loss_fn(output, target)

        # record
        epoch_loss += loss.item()
        n_samples += output.size(0)

        # adjust
        loss.backward()
        optimizer.step()

    return epoch_loss, n_samples

def test(model, device, test_loader):
    with torch.no_grad():
        model.train(False)
        num_correct = 0
        num_samples = 0

        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            scores = model.forward(x)
            _, y_out = scores.max(1)
            
            num_correct += (y_out == y).sum()
            num_samples += y_out.size(0)
        
        acc = float(num_correct) / float(num_samples)
    return acc

def get_weights(model_state_dict, layer_idx):
    if f'layers.{layer_idx}.weight' in model_state_dict.keys():
        weights = model_state_dict[f'layers.{layer_idx}.weight'].numpy()
    else:
        raise KeyError(f'layers.{layer_idx}.weight')

    return weights

def get_bias(model_state_dict, layer_idx):
    if f'layers.{layer_idx}.bias' in model_state_dict.keys():
        bias = model_state_dict[f'layers.{layer_idx}.bias'].numpy()
    else:
        raise KeyError(f'layers.{layer_idx}.bias')

    return bias

def get_scale(qmodel_state_dict, layer_idx):
    w_scale = qmodel_state_dict[f'{layer_idx}._packed_params._packed_params'][0][0].q_scale()
    out_scale = qmodel_state_dict[f'{layer_idx}.scale']
    return w_scale, out_scale.numpy()

def get_zero(qmodel_state_dict, layer_idx):
    w_zero = qmodel_state_dict[f'{layer_idx}._packed_params._packed_params'][0][0].q_zero_point()
    out_zero = qmodel_state_dict[f'{layer_idx}.zero_point']
    return w_zero, out_zero.numpy()

def get_input_qparams(qmodel_state_dict):
    scale = qmodel_state_dict[f'0.scale'][0].numpy()
    zero_point = qmodel_state_dict[f'0.zero_point'][0].numpy()

    return scale, zero_point
